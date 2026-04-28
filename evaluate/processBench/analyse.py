import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

INPUT_FILE="# INPUT_FILE"
DATA_FILE="# DATA_FILE"
SAVE_WRONG="# SAVE_WRONG_FILE"
SAVE_IGNORED="# SAVE_IGNORED_FILE"

def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = p.read_text(encoding="utf-8")
    text_stripped = text.lstrip()

    if text_stripped.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"File {path} parsed as JSON but is not a list")

    results: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except Exception as e:
            raise ValueError(f"Failed to parse line {i} in {path} as JSON: {e}")
    return results


def find_first_error_step(sample_steps: List[Dict[str, Any]], threshold: float) -> Optional[int]:
    """Return the first step_index predicted as incorrect by avg_score."""
    for step in sample_steps:
        step_index = step.get("step_index")
        avg_score = step.get("avg_score")
        if avg_score is None:
            continue
        if avg_score <= threshold:
            return step_index
    return None


def evaluate_results(
    results_data: List[Dict[str, Any]],
    math_label_map: Dict[str, int],
    threshold: float,
) -> Tuple[
    List[int],
    List[int],
    List[str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]:
    """Evaluate predictions and return metrics inputs and sample details."""
    samples_by_orig: Dict[str, List[Dict[str, Any]]] = {}
    for sample in results_data:
        orig_id = sample.get("orig_id")
        if orig_id is None:
            continue
        samples_by_orig.setdefault(orig_id, []).append(sample)

    predictions: List[int] = []
    labels: List[int] = []
    orig_ids: List[str] = []
    ignored_samples: List[Dict[str, Any]] = []
    sample_steps_map: Dict[str, List[Dict[str, Any]]] = {}

    for orig_id, sample_steps in samples_by_orig.items():
        try:
            sample_steps_sorted = sorted(sample_steps, key=lambda s: int(s.get("step_index", 0)))
        except Exception:
            sample_steps_sorted = sorted(sample_steps, key=lambda s: s.get("step_index", 0))

        has_missing_score = any(step.get("avg_score") is None for step in sample_steps_sorted)

        first_error_step = find_first_error_step(sample_steps_sorted, threshold)
        pred = first_error_step if first_error_step is not None else -1

        if has_missing_score and first_error_step is None:
            ignored_samples.append(
                {
                    "orig_id": orig_id,
                    "reason": "missing_avg_score_without_detected_error",
                    "steps": sample_steps_sorted,
                }
            )
            continue

        if orig_id not in math_label_map:
            continue

        label = math_label_map[orig_id]

        predictions.append(pred)
        labels.append(label)
        orig_ids.append(orig_id)
        sample_steps_map[orig_id] = sample_steps_sorted

    return predictions, labels, orig_ids, ignored_samples, sample_steps_map, samples_by_orig


def main() -> None:
    # Only keep a single input_file variable for results JSON path.
    input_file = INPUT_FILE
    results_path = Path(input_file)
    data_path = Path(DATA_FILE)
    threshold = 0.5
    save_wrong = SAVE_WRONG
    save_ignored = SAVE_IGNORED

    print("Loading files...")
    print(f"Results: {results_path}")
    print(f"Data:    {data_path}")

    results_data = load_json(str(results_path))
    math_data = load_json(str(data_path))

    print(f"Results steps: {len(results_data)}")
    print(f"Data samples:  {len(math_data)}")

    math_label_map = {sample["id"]: sample["label"] for sample in math_data if "id" in sample and "label" in sample}
    print(f"Label map size: {len(math_label_map)}")

    predictions, labels, orig_ids, ignored_samples, sample_steps_map, _ = evaluate_results(
        results_data, math_label_map, threshold
    )

    print("\nEvaluation summary:")
    print(f"  Valid samples:  {len(predictions)}")
    print(f"  Ignored samples: {len(ignored_samples)}")

    error_samples = [
        (orig_id, pred, label)
        for orig_id, pred, label in zip(orig_ids, predictions, labels)
        if label != -1
    ]
    if error_samples:
        first_error_step_correct = sum(pred == label for _, pred, label in error_samples)
        first_error_step_accuracy = first_error_step_correct / len(error_samples)
        wrong_error_samples = [
            {
                "orig_id": orig_id,
                "predicted_first_error_step": pred,
                "true_first_error_step": label,
                "error_type": "first_error_step_mismatch",
                "steps": sample_steps_map.get(orig_id, []),
            }
            for orig_id, pred, label in error_samples
            if pred != label
        ]
    else:
        first_error_step_correct = 0
        first_error_step_accuracy = 0.0
        wrong_error_samples = []

    correct_samples = [
        (orig_id, pred, label)
        for orig_id, pred, label in zip(orig_ids, predictions, labels)
        if label == -1
    ]
    if correct_samples:
        no_error_correct = sum(pred == -1 for _, pred, _ in correct_samples)
        no_error_accuracy = no_error_correct / len(correct_samples)
        wrong_correct_samples = [
            {
                "orig_id": orig_id,
                "predicted_first_error_step": pred,
                "true_first_error_step": -1,
                "error_type": "false_positive_error_detection",
                "steps": sample_steps_map.get(orig_id, []),
            }
            for orig_id, pred, _ in correct_samples
            if pred != -1
        ]
    else:
        no_error_correct = 0
        no_error_accuracy = 0.0
        wrong_correct_samples = []

    if first_error_step_accuracy + no_error_accuracy > 0:
        f1 = (
            2
            * first_error_step_accuracy
            * no_error_accuracy
            / (first_error_step_accuracy + no_error_accuracy)
        )
    else:
        f1 = 0.0

    print("\n=== Results (sample-level, first-error evaluation) ===")
    print(f"Error samples: {len(error_samples)}")
    print(f"  Correct first-error predictions: {first_error_step_correct}")
    print(f"  First-error accuracy: {first_error_step_accuracy:.4f}")
    print(f"  Wrong samples: {len(wrong_error_samples)}")
    print(f"\nNo-error samples: {len(correct_samples)}")
    print(f"  Correct no-error predictions: {no_error_correct}")
    print(f"  No-error accuracy: {no_error_accuracy:.4f}")
    print(f"  Wrong samples: {len(wrong_correct_samples)}")
    print(f"\nF1 (harmonic mean of the two accuracies): {f1:.4f}")

    wrong_samples = wrong_error_samples + wrong_correct_samples
    if save_wrong:
        wrong_path = Path(save_wrong)
        with open(wrong_path, "w", encoding="utf-8") as f:
            json.dump(wrong_samples, f, ensure_ascii=False, indent=2)
        print(f"\nSaved wrong samples: {wrong_path} (count: {len(wrong_samples)})")

    if save_ignored:
        ignored_path = Path(save_ignored)
        with open(ignored_path, "w", encoding="utf-8") as f:
            json.dump(ignored_samples, f, ensure_ascii=False, indent=2)
        print(f"Saved ignored samples: {ignored_path} (count: {len(ignored_samples)})")


if __name__ == "__main__":
    main()
