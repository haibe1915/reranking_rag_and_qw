import json
import os
from statistics import mean

import pytest

from main import run_rag_pipeline

BASELINE_DATA = os.path.join(os.path.dirname(__file__), "..", "evaluation", "baseline_dataset.json")
BASELINE_RESULT = os.path.join(os.path.dirname(__file__), "..", "evaluation", "baseline_result.json")


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def compute_aggregates(metrics_list):
    ems = [m.get("em", 0.0) for m in metrics_list]
    f1s = [m.get("f1", 0.0) for m in metrics_list]
    return {"avg_em": mean(ems) if ems else 0.0, "avg_f1": mean(f1s) if f1s else 0.0}


def test_record_baseline(tmp_path):
    """Run the pipeline on the small baseline dataset, compute avg EM/F1,
    and write the baseline result JSON. This test always passes but ensures
    the baseline result file is produced for tracking.
    """
    dataset = load_dataset(BASELINE_DATA)
    per_example = []

    for item in dataset:
        query = item["query"]
        ground_truth = item["ground_truth"]
        # run_rag_pipeline returns (answer, metrics) when ground_truth provided
        answer, metrics = run_rag_pipeline(query, ground_truth)
        per_example.append({
            "id": item.get("id"),
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "metrics": metrics,
        })

    aggregates = compute_aggregates([p["metrics"] for p in per_example])
    results = {"per_example": per_example, "aggregates": aggregates}

    # Write to the project's evaluation folder (not the tmp path) so baseline is tracked
    save_results(BASELINE_RESULT, results)

    # Basic assertions: file created and aggregates keys present
    assert os.path.exists(BASELINE_RESULT)
    with open(BASELINE_RESULT, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "aggregates" in loaded
    assert "avg_em" in loaded["aggregates"]
    assert "avg_f1" in loaded["aggregates"]
