from typing import Callable

import torch
from datasets import Dataset, concatenate_datasets
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...utils import (
    clean_text,
    get_logger,
    get_random_string,
    get_vocab,
    log,
    truncate,
)

_logger = get_logger(__name__)


def domain_calibration(
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
    language: str,
    **kwargs,
) -> torch.Tensor:
    """
    Implements domain calibration from the paper https://arxiv.org/pdf/2305.19148
    It uses the same number of IC examples as used for classification, that means that,
    if you use 2 IC examples for classification, 2 IC examples will be used too to
    perform the calibration.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        prompted_dataset (Dataset): a dataset with the prompts already in in the `prompts` column
        input_columns (list[str]): columns of the input text to fill the prompt
        max_seq_length (int): maximum length of the whole prompt. If longer, the prompt will be truncated
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        labels (list[str]): list of labels in the dataset
        batch_size (int): samples per batch
        logits (torch.Tensor): logits to be calibrated
        forward_fn (Callable): the classification forward function to perform a forward pass with context-free inputs
        num_estimations (int): number of samples to be used for computing `hat_p` (See Eq. 2 of the paper)
        language (str): language for the Spacy tokenizer use to build the random strings
    Returns:
        torch.Tensor: calibrated probabilities
    """
    # Get the vocabulary of the dataset
    tokens: list[str] = sum(
        [prompted_dataset[input_column] for input_column in input_columns], []
    )
    vocab = get_vocab(
        tokens,
        language,
        n_process=kwargs.get("n_process", 4),
        batch_size=kwargs.get("spacy_batch_size", 2000),
    )

    # Build the random strings
    random_strings = [
        get_random_string(vocab, tokenizer, max_input_length)
        for _ in range(num_estimations * len(input_columns))
    ]

    # Select one sample and repeat it `num_estimation` times to
    # share the same in-context examples for all the random strings
    # according to Eq 2. of https://arxiv.org/pdf/2305.19148
    random_dataset = prompted_dataset.select(range(1))
    random_dataset = concatenate_datasets(
        [random_dataset for _ in range(num_estimations)]
    )

    # Replace the input columns with the random strings from right to left
    def replace_input_prompt(example):
        prompt = example["prompts"]
        for input_column in input_columns[::-1]:
            str_to_replace = truncate(
                clean_text(example[input_column]), tokenizer, max_input_length
            )
            prompt = prompt[::-1].replace(
                str_to_replace[::-1], random_strings.pop()[::-1], 1
            )[::-1]
        return {"prompts": prompt}

    random_dataset = random_dataset.map(replace_input_prompt)
    log(
        _logger.info,
        f"This is how one prompt for domain calibration looks like: {random_dataset['prompts'][0]}",
        "yellow",
    )

    # Tokenize inputs
    tokenized_dataset = random_dataset.map(
        lambda examples: tokenizer(
            examples["prompts"], truncation=True, max_length=max_seq_length
        ),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.select_columns(
        ["input_ids", "attention_mask"]
    )

    # Get the `p_hat` probs (from Eq. 2, averaged across `num_estimations`)
    random_logits = forward_fn(
        model, tokenizer, tokenized_dataset, labels, batch_size
    )
    p_hat = F.softmax(random_logits, dim=-1).mean(dim=0)
    log(_logger.info, f"`p_hat` of domain calibration is: {p_hat}", "green")

    # Calibrate the classification probs as in Eq. 3 (B x C)
    calibrated_probs = F.softmax(F.softmax(logits, dim=-1) / p_hat)
    return calibrated_probs
