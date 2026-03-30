import json
import os
import sys
from statistics import mean

# Ensure project root is on sys.path so imports like `from main import ...` work
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import run_rag_pipeline
BASELINE_DATA = os.path.join(BASE_DIR, "evaluation", "baseline_dataset.json")
BASELINE_RESULT = os.path.join(BASE_DIR, "evaluation", "baseline_result.json")


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


def main():
    dataset = load_dataset(BASELINE_DATA)
    per_example = []

    for item in dataset:
        query = item["query"]
        ground_truth = item["ground_truth"]
        print(f"Running: {query}")
        answer, metrics = run_rag_pipeline(query, ground_truth)
        print(f"Answer: {answer}")
        print(f"Metrics: {metrics}\n")
        per_example.append({
            "id": item.get("id"),
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "metrics": metrics,
        })

    aggregates = compute_aggregates([p["metrics"] for p in per_example])
    results = {"per_example": per_example, "aggregates": aggregates}

    save_results(BASELINE_RESULT, results)
    print("Baseline aggregates:")
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))
    print(f"Saved baseline result to: {BASELINE_RESULT}")


if __name__ == "__main__":
    main()
