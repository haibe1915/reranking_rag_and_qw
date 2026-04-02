import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "uit_viquad2" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "uit_viquad2" / "processed"
EVAL_DIR = ROOT_DIR / "evaluation"

PARQUET_BY_SPLIT = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _extract_primary_answer(answers: Any) -> str:
    if not isinstance(answers, dict):
        return ""

    texts = answers.get("text")
    if isinstance(texts, list) and texts:
        return _to_clean_str(texts[0])

    if isinstance(texts, str):
        return _to_clean_str(texts)

    # Hugging Face parquet often stores nested list fields as numpy arrays.
    try:
        if len(texts) > 0:
            return _to_clean_str(texts[0])
    except TypeError:
        return ""

    return ""


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    primary_answer = _extract_primary_answer(row.get("answers"))
    if not primary_answer:
        primary_answer = _extract_primary_answer(row.get("plausible_answers"))

    normalized = {
        "id": _to_clean_str(row.get("id")),
        "uit_id": _to_clean_str(row.get("uit_id")),
        "title": _to_clean_str(row.get("title")),
        "context": _to_clean_str(row.get("context")),
        "query": _to_clean_str(row.get("question")),
        "ground_truth": primary_answer,
        "is_impossible": bool(row.get("is_impossible", False)),
    }
    return normalized


def load_and_normalize_split(split: str) -> List[Dict[str, Any]]:
    parquet_name = PARQUET_BY_SPLIT[split]
    parquet_path = RAW_DIR / parquet_name
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    records = df.to_dict(orient="records")
    normalized = [_normalize_row(r) for r in records]
    return normalized


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_baseline_dataset(rows: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
    # Keep only answerable examples for baseline EM/F1 style evaluation.
    answerable = [r for r in rows if r["query"] and r["ground_truth"]]
    baseline = answerable[:size]

    for idx, item in enumerate(baseline, start=1):
        item["id"] = idx

    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess UIT-ViQuAD2 parquet files")
    parser.add_argument(
        "--baseline-split",
        choices=["train", "validation"],
        default="validation",
        help="Split dùng để tạo file evaluation/baseline_dataset_from_uit_viquad2.json",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=200,
        help="Số lượng mẫu cho baseline dataset",
    )
    args = parser.parse_args()

    all_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split in PARQUET_BY_SPLIT:
        rows = load_and_normalize_split(split)
        all_rows_by_split[split] = rows

        out_jsonl = PROCESSED_DIR / f"{split}.jsonl"
        write_jsonl(out_jsonl, rows)
        print(f"[{split}] Wrote {len(rows)} rows -> {out_jsonl}")

    baseline_rows = build_baseline_dataset(all_rows_by_split[args.baseline_split], args.baseline_size)
    baseline_out = EVAL_DIR / "baseline_dataset_from_uit_viquad2.json"
    write_json(baseline_out, baseline_rows)
    print(f"[baseline] Wrote {len(baseline_rows)} rows -> {baseline_out}")


if __name__ == "__main__":
    main()
