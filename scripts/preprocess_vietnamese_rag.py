import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "vietnamese_rag" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "vietnamese_rag" / "processed"
EVAL_DIR = ROOT_DIR / "evaluation"


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_context(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(_to_clean_str(x) for x in value if _to_clean_str(x))
    return _to_clean_str(value)


def _extract_text_from_array_repr(raw_value: str) -> str:
    match = re.search(r"'text'\s*:\s*array\(\[(.*?)\]", raw_value, re.S)
    if not match:
        return ""

    content = match.group(1).strip()
    if not content:
        return ""

    try:
        parsed = ast.literal_eval(f"[{content}]")
        if parsed:
            return _to_clean_str(parsed[0])
    except (ValueError, SyntaxError):
        pass

    str_match = re.search(r"'([^']+)'", content)
    if str_match:
        return _to_clean_str(str_match.group(1))

    return ""


def _extract_primary_answer(value: Any) -> str:
    if isinstance(value, dict):
        texts = value.get("text")
        if isinstance(texts, list) and texts:
            return _to_clean_str(texts[0])
        if isinstance(texts, str):
            return _to_clean_str(texts)
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        return _extract_text_from_array_repr(stripped)

    return ""


def _normalize_bkai_or_legal(row: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "id": "",
        "uit_id": "",
        "title": source,
        "context": _join_context(row.get("context")),
        "query": _to_clean_str(row.get("question")),
        "ground_truth": _to_clean_str(row.get("answer")),
        "is_impossible": False,
        "source": source,
    }


def _normalize_vi_rag(row: Dict[str, Any]) -> Dict[str, Any]:
    context = _to_clean_str(row.get("revised_claims"))
    if not context:
        context = _to_clean_str(row.get("field"))

    return {
        "id": "",
        "uit_id": "",
        "title": _to_clean_str(row.get("spec_field")) or "vi_rag",
        "context": context,
        "query": _to_clean_str(row.get("question")),
        "ground_truth": _to_clean_str(row.get("revised_answer")),
        "is_impossible": False,
        "source": "vi_RAG",
    }


def _normalize_rag_viquad(row: Dict[str, Any]) -> Dict[str, Any]:
    primary_answer = _extract_primary_answer(row.get("answers"))
    is_impossible = str(row.get("is_impossible", "False")).lower() == "true"
    if not primary_answer:
        primary_answer = _extract_primary_answer(row.get("plausible_answers"))

    return {
        "id": _to_clean_str(row.get("id")),
        "uit_id": _to_clean_str(row.get("uit_id")),
        "title": _to_clean_str(row.get("title")),
        "context": _to_clean_str(row.get("contexts")),
        "query": _to_clean_str(row.get("question")),
        "ground_truth": primary_answer,
        "is_impossible": is_impossible,
        "source": "rag_viQuAD",
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_and_normalize_all() -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []

    bkai_path = RAW_DIR / "modified_data_BKAI.jsonl"
    for row in _read_jsonl(bkai_path):
        all_rows.append(_normalize_bkai_or_legal(row, source="modified_data_BKAI"))

    legal_path = RAW_DIR / "modify_legal_corpus.json"
    for row in _read_jsonl(legal_path):
        all_rows.append(_normalize_bkai_or_legal(row, source="modify_legal_corpus"))

    rag_viquad_path = RAW_DIR / "rag_viQuAD.json"
    with rag_viquad_path.open("r", encoding="utf-8") as f:
        rag_rows = json.load(f)
    all_rows.extend(_normalize_rag_viquad(r) for r in rag_rows)

    vi_rag_path = RAW_DIR / "vi_RAG.json"
    with vi_rag_path.open("r", encoding="utf-8") as f:
        vi_rag_rows = json.load(f)
    all_rows.extend(_normalize_vi_rag(r) for r in vi_rag_rows)

    return all_rows


def split_rows(rows: List[Dict[str, Any]], seed: int, train_ratio: float, val_ratio: float) -> Dict[str, List[Dict[str, Any]]]:
    random.Random(seed).shuffle(rows)

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": rows[:train_end],
        "validation": rows[train_end:val_end],
        "test": rows[val_end:],
    }


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
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese RAG raw files")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho viec shuffle truoc khi split")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ty le train")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ty le validation")
    parser.add_argument(
        "--baseline-split",
        choices=["train", "validation"],
        default="validation",
        help="Split dung de tao file evaluation/baseline_dataset_from_vietnamese_rag.json",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=200,
        help="So luong mau cho baseline dataset",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Can dam bao 0 < train_ratio, val_ratio va train_ratio + val_ratio < 1")

    all_rows = load_and_normalize_all()
    splits = split_rows(all_rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    for split_name, rows in splits.items():
        out_jsonl = PROCESSED_DIR / f"{split_name}.jsonl"
        write_jsonl(out_jsonl, rows)
        print(f"[{split_name}] Wrote {len(rows)} rows -> {out_jsonl}")

    baseline_rows = build_baseline_dataset(splits[args.baseline_split], args.baseline_size)
    baseline_out = EVAL_DIR / "baseline_dataset_from_vietnamese_rag.json"
    write_json(baseline_out, baseline_rows)
    print(f"[baseline] Wrote {len(baseline_rows)} rows -> {baseline_out}")


if __name__ == "__main__":
    main()
