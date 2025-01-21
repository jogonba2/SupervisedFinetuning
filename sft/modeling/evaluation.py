from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from evaluate import load as load_metric
from torchmetrics.classification import MulticlassCalibrationError

from ..utils import get_logger, log, read_json

_logger = get_logger(__name__)


def evaluate(
    references: list[str],
    predictions: list[str],
    probabilities: list[list[float]],
    metric_name: str,
    eval_args: Dict[str, Any],
) -> dict:
    """
    Evaluates the performance of a model.

    Args:
        references (list[str]): list of references
        predictions (list[str]): list predictions
        metric_name (str): name of the metric in the HF hub
        eval_args (Dict): Dictionary of args that will be applied on the evaluation metrics.

    Returns:
        dict: results from the HF metric
    """
    # Compute HF metric
    metric = load_metric(metric_name)

    results = metric.compute(
        predictions=predictions,
        references=references,
        **eval_args,
    )

    # Compute ECE if required
    if probabilities is not None:
        label_set = sorted(list(set(references)))
        label_idxs = {label: idx for idx, label in enumerate(label_set)}
        targets = torch.LongTensor([label_idxs[label] for label in references])
        probs = torch.Tensor(probabilities)
        mcce = MulticlassCalibrationError(
            num_classes=len(label_set),
            n_bins=eval_args.get("ece_bins", 10),
            norm=eval_args.get("ece_norm", "l1"),
        )
        results["ece"] = mcce(probs, targets).item()

    log(
        _logger.info,
        f"This is how a prediction looks like: {predictions[0]}",
        "blue",
    )
    log(
        _logger.info,
        f"This is how a reference looks like: {references[0]}",
        "blue",
    )

    return results


def aggregate_results(
    results_path: Path, file_name: str = "aggregated_results"
) -> None:
    """
    Aggregates the results of the json files in `results_path`
    and stores them in json and markdown formats.

    Args:
        results_path (Path): path with a set of json files with results
        file_name (str): name of the file where to store the aggregated results
    """
    rows = []
    for f in results_path.glob("*.json"):
        d = read_json(f)
        row = {
            "dataset": d["config"]["dataset"]["path"],
            "model": d["config"]["model"]["name"],
            "task_type": d["config"]["dataset"]["task_type"],
            "shots": d["config"]["dataset"]["shots"],
            "test_samples": d["config"]["dataset"].get("max_n_test", "all"),
            "max_input_length": d["config"]["prompting"]["max_input_length"],
            "max_output_length": d["config"]["prompting"]["max_output_length"],
            "max_seq_length": d["config"]["inference"]["max_seq_length"],
            "icl_training": d["config"]["training"].get("icl_examples", 0),
            "icl_inference": d["config"]["inference"].get("icl_examples", 0),
            "random_seed": d["seed"],
        }
        if row["shots"] > 0:
            # Default LR from trainer if not in config
            row["learning_rate"] = d["config"]["training"].get(
                "learning_rate", 5e-5
            )
            row = {**row, **d["config"]["lora"]}

        if row["task_type"] == "classification":
            row = {**row, **d["results"]}
        else:
            row = {
                **row,
                **{
                    "rouge_1": d["results"]["ROUGE"]["rouge1"],
                    "rouge_2": d["results"]["ROUGE"]["rouge2"],
                    "rouge_L": d["results"]["ROUGE"]["rougeLsum"],
                    "bleu": d["results"]["BLEU"]["bleu"],
                    "exact_match": d["results"]["EXACT_MATCH"]["exact_match"],
                    "bert_score": d["results"]["BERT_SCORE"]["f1"],
                    "bleurt_score": d["results"]["BLEURT"]["scores"],
                },
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["dataset", "model", "shots", "icl_training", "icl_inference"]
    )
    df.to_json(f"{file_name}.json", indent=4)
    df.to_markdown(f"{file_name}.md", index=False)
