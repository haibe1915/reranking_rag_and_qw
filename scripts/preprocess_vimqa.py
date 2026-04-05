import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "vimqa" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "vimqa" / "processed"
EVAL_DIR = ROOT_DIR / "evaluation"
DATASET_TAG = "vimqa"

RAW_BY_SPLIT = {
    "train": "vimqa_train.json",
    "validation": "vimqa_dev.json",
    "test": "vimqa_test.json",
}


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_whitespace(text: str) -> str:
    # Collapse all whitespace, including newlines and tabs, into single spaces.
    return re.sub(r"\s+", " ", text).strip()


def _flatten_context(context: Any) -> str:
    if not isinstance(context, list):
        return _normalize_whitespace(_to_clean_str(context))

    chunks: List[str] = []
    for item in context:
        if isinstance(item, list) and len(item) >= 2:
            title = _to_clean_str(item[0])
            body = item[1]
            if isinstance(body, list):
                body_text = " ".join(_to_clean_str(x) for x in body if _to_clean_str(x))
            else:
                body_text = _to_clean_str(body)
            combined = " ".join(part for part in [title, body_text] if part)
            if combined:
                chunks.append(combined)
        else:
            text = _to_clean_str(item)
            if text:
                chunks.append(text)

    return _normalize_whitespace(" ".join(chunks))


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": _to_clean_str(row.get("_id")),
        "title": "",
        "context": _flatten_context(row.get("context")),
        "query": _to_clean_str(row.get("question")),
        "ground_truth": _to_clean_str(row.get("answer")),
        "is_impossible": False,
        "supporting_facts": row.get("supporting_facts", []),
    }


def _normalize_id_token(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", _to_clean_str(value).lower()).strip("_")
    return token or "default"


def assign_standard_ids(rows: List[Dict[str, Any]], split: str, group_key: str = "") -> None:
    split_token = _normalize_id_token(split)
    for idx, row in enumerate(rows, start=1):
        if group_key:
            group_token = _normalize_id_token(row.get(group_key))
            if group_token != "default":
                row["id"] = f"{DATASET_TAG}:{group_token}:{split_token}:{idx:06d}"
                continue
        row["id"] = f"{DATASET_TAG}:{split_token}:{idx:06d}"


def load_and_normalize_split(split: str) -> List[Dict[str, Any]]:
    raw_name = RAW_BY_SPLIT[split]
    raw_path = RAW_DIR / raw_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Khong tim thay file raw: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    normalized = [_normalize_row(r) for r in rows]
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
    parser = argparse.ArgumentParser(description="Preprocess VIMQA raw files")
    parser.add_argument(
        "--baseline-split",
        choices=["train", "validation"],
        default="validation",
        help="Split dung de tao file evaluation/baseline_dataset_from_vimqa.json",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=200,
        help="So luong mau cho baseline dataset",
    )
    args = parser.parse_args()

    all_rows_by_split: Dict[str, List[Dict[str, Any]]] = {}

    for split in RAW_BY_SPLIT:
        rows = load_and_normalize_split(split)
        assign_standard_ids(rows, split=split)
        all_rows_by_split[split] = rows

        out_jsonl = PROCESSED_DIR / f"{split}.jsonl"
        write_jsonl(out_jsonl, rows)
        print(f"[{split}] Wrote {len(rows)} rows -> {out_jsonl}")

    baseline_rows = build_baseline_dataset(all_rows_by_split[args.baseline_split], args.baseline_size)
    baseline_out = EVAL_DIR / "baseline_dataset_from_vimqa.json"
    write_json(baseline_out, baseline_rows)
    print(f"[baseline] Wrote {len(baseline_rows)} rows -> {baseline_out}")


if __name__ == "__main__":
    main()
