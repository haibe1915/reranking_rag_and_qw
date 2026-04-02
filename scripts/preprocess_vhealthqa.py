import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "vhealthqa" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "vhealthqa" / "processed"
EVAL_DIR = ROOT_DIR / "evaluation"

CSV_BY_SPLIT = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv",
}


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    link = _to_clean_str(row.get("link"))
    return {
        "id": _to_clean_str(row.get("id")),
        "uit_id": "",
        "title": "vhealthqa",
        "context": link,
        "query": _to_clean_str(row.get("question")),
        "ground_truth": _to_clean_str(row.get("answer")),
        "is_impossible": False,
        "link": link,
    }


def load_and_normalize_split(split: str) -> List[Dict[str, Any]]:
    csv_name = CSV_BY_SPLIT[split]
    csv_path = RAW_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Khong tim thay file csv: {csv_path}")

    df = pd.read_csv(csv_path)
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
    answerable = [r for r in rows if r["query"] and r["ground_truth"]]
    baseline = [
        {
            "id": idx,
            "query": row["query"],
            "ground_truth": row["ground_truth"],
        }
        for idx, row in enumerate(answerable[:size], start=1)
    ]
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess VHealthQA CSV files")
    parser.add_argument(
        "--baseline-split",
        choices=["train", "validation"],
        default="train",
        help="Split dung de tao file evaluation/baseline_dataset_from_vhealthqa.json",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=200,
        help="So luong mau cho baseline dataset",
    )
    args = parser.parse_args()

    all_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split in CSV_BY_SPLIT:
        rows = load_and_normalize_split(split)
        all_rows_by_split[split] = rows

        out_jsonl = PROCESSED_DIR / f"{split}.jsonl"
        write_jsonl(out_jsonl, rows)
        print(f"[{split}] Wrote {len(rows)} rows -> {out_jsonl}")

    baseline_rows = build_baseline_dataset(all_rows_by_split[args.baseline_split], args.baseline_size)
    baseline_out = EVAL_DIR / "baseline_dataset_from_vhealthqa.json"
    write_json(baseline_out, baseline_rows)
    print(f"[baseline] Wrote {len(baseline_rows)} rows -> {baseline_out}")


if __name__ == "__main__":
    main()
