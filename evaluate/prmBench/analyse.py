#!/usr/bin/env python3
"""Statistical analysis script for PRMBench inference results.

This script loads the generated JSON results, evaluates the Process Reward 
Model's performance (Accuracy, Precision, Recall, F1), and outputs both 
overall statistics and per-classification statistics.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Configuration Parameters
INPUT_FILE = "output/result.json"
REWARD_THRESHOLD = 0.5



def calculate_metrics(results: List[Dict[str, Any]], threshold: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Calculates evaluation metrics based on predictions and ground truth.

    Args:
        results: A list of result dictionaries containing 'is_error' and 'reward'.
        threshold: The reward threshold above which a step is predicted as an error.

    Returns:
        A tuple containing:
            - A dictionary of aggregated metrics (accuracy, precision, f1, etc.)
            - A list of detailed judgment records for deep dives.
    """
    tp = fp = tn = fn = 0
    detailed_judgments: List[Dict[str, Any]] = []

    for result in results:
        is_error = result.get("is_error", False)
        reward = result.get("reward", 0.5)
        predicted_error = reward < threshold

        if (not is_error) and (not predicted_error):
            tp += 1
            judgment_type = "TP"
            judgment_desc = "Correctly predicted as error"
        elif is_error and (not predicted_error):
            fp += 1
            judgment_type = "FP"
            judgment_desc = "Incorrectly predicted as error (False Alarm)"
        elif is_error and predicted_error:
            tn += 1
            judgment_type = "TN"
            judgment_desc = "Correctly predicted as correct"
        else:
            fn += 1
            judgment_type = "FN"
            judgment_desc = "Incorrectly predicted as correct (Missed Error)"

        detailed_judgments.append({
            "original_id": result.get("original_id", "unknown"),
            "step_id": result.get("step_id", 0),
            "is_error": is_error,
            "reward": reward,
            "predicted_error": predicted_error,
            "judgment_type": judgment_type,
            "judgment_desc": judgment_desc,
        })

    total = len(results)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    negative_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if (negative_precision + negative_recall) > 0
        else 0.0
    )

    metrics = {
        "total_samples": total,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "negative_precision": negative_precision,
        "negative_recall": negative_recall,
        "negative_f1_score": negative_f1,
        "Avg_F1": (f1+negative_f1)/2
    }
    
    return metrics, detailed_judgments


def print_report(title: str, metrics: Dict[str, Any]) -> None:
    """Prints a beautifully formatted metric report to the console."""
    print(f"\n--- {title.upper()} ---")
    print(f"Total Samples : {metrics['total_samples']}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1 Score      : {metrics['f1_score']:.4f}")
    print(f"Neg Precision : {metrics['negative_precision']:.4f}")
    print(f"Neg Recall    : {metrics['negative_recall']:.4f}")
    print(f"Neg F1 Score  : {metrics['negative_f1_score']:.4f}")
    print(f"Avg F1 Score  : {metrics['Avg_F1']:.4f}")
    print(f"Confusion Matrix: TP={metrics['true_positives']} | FP={metrics['false_positives']} | TN={metrics['true_negatives']} | FN={metrics['false_negatives']}")


def main() -> None:
    """Main execution function for the analysis script."""
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        print(f"Error: Input file not found -> {INPUT_FILE}")
        return

    print(f"Loading results from: {INPUT_FILE}")
    with open(input_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
        
    if not all_results:
        print("Error: The results file is empty.")
        return

    # Calculate Overall Metrics
    overall_metrics, overall_details = calculate_metrics(all_results, REWARD_THRESHOLD)
    print_report("Overall Statistics", overall_metrics)

    # Group data by classification for fine-grained analysis
    grouped_results = defaultdict(list)
    for res in all_results:
        cls = res.get("classification", "unclassified")
        grouped_results[cls].append(res)

    # Calculate and print per-classification metrics
    classification_reports = {}
    if len(grouped_results) > 1 or "unclassified" not in grouped_results:
        for cls, cls_results in grouped_results.items():
            cls_metrics, _ = calculate_metrics(cls_results, REWARD_THRESHOLD)
            classification_reports[cls] = cls_metrics
            print_report(f"Category: {cls}", cls_metrics)

    
if __name__ == "__main__":
    main()