from typing import Callable

import torch
from datasets import Dataset
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...utils import clean_text, get_logger, log, truncate

_logger = get_logger(__name__)


def contextual_calibration(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompted_dataset: Dataset,
    input_columns: list[str],
    max_seq_length: int,
    max_input_length: int,
    labels: list[str],
    batch_size: int,
    logits: torch.Tensor,
    forward_fn: Callable,
    num_estimations: int,
    **kwargs,
) -> torch.Tensor:
    """
    Implements contextual calibration from the paper https://arxiv.org/pdf/2102.09690
    It uses the same number of IC examples as used for classification, that means that,
    if you use 2 IC examples for classification, 2 IC examples will be used too to
    perform the calibration.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        prompted_dataset (Dataset): a dataset with the prompts already in the `prompts` column
        input_columns (list[str]): columns of the input text to fill the prompt
        max_seq_length (int): maximum length of the whole prompt. If longer, the prompt will be truncated
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        labels (list[str]): list of labels in the dataset
        batch_size (int): samples per batch
        logits (torch.Tensor): logits to be calibrated
        forward_fn (Callable): the classification forward function to perform a forward pass with context-free inputs
        num_estimations (int): number of samples to be used for computing `pcf` (See §5 of the paper)
    Returns:
        torch.Tensor: calibrated probabilities
    """
    # Prepare a dataset with N/A inputs
    na_dataset = prompted_dataset.select(range(num_estimations))

    # By replacing the input columns with N/A from right to left
    def replace_input_prompt(example):
        prompt = example["prompts"]
        for input_column in input_columns[::-1]:
            str_to_replace = truncate(
                clean_text(example[input_column]), tokenizer, max_input_length
            )
            prompt = prompt[::-1].replace(str_to_replace[::-1], "A/N", 1)[::-1]
        return {"prompts": prompt}

    na_dataset = na_dataset.map(replace_input_prompt)
    log(
        _logger.info,
        f"This is how one prompt for contextual calibration looks like: {na_dataset['prompts'][0]}",
        "yellow",
    )

    # Tokenize inputs
    tokenized_dataset = na_dataset.map(
        lambda examples: tokenizer(
            examples["prompts"], truncation=True, max_length=max_seq_length
        ),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.select_columns(
        ["input_ids", "attention_mask"]
    )

    # Get the probs (averaged across `num_estimations`)
    na_logits = forward_fn(
        model, tokenizer, tokenized_dataset, labels, batch_size
    )
    pcf = F.softmax(na_logits, dim=-1).mean(dim=0)

    # Compute W=diag(p)^-1 (diagonal C x C) and diagonalize
    W = torch.linalg.inv(torch.eye(pcf.shape[0]) * pcf).diag()
    log(_logger.info, f"`pcf` of contextual calibration is: {W}", "green")

    # Calibrate the classification probs and recompute softmax (B x C)
    calibrated_probs = F.softmax(W * F.softmax(logits, dim=-1))
    return calibrated_probs
