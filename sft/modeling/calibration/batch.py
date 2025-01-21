from functools import partial
from typing import Callable

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from torch.nn import functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from ...prompting.prompt import format_prompt
from ...utils import get_logger, log

_logger = get_logger(__name__)


def batch_calibration(
    logits: torch.Tensor,
    num_estimations: int,
    **kwargs,
) -> torch.Tensor:
    """
    Implements batch calibration from the paper https://arxiv.org/pdf/2309.17249
    It uses the same number of IC examples as used for classification, that means that,
    if you use 2 IC examples for classification, 2 IC examples will be used too to
    perform the calibration.
    Besides, it uses the logits of the test samples that have been passed through the
    classification forward pass.

    Args:
        logits (torch.Tensor): logits to be calibrated
        num_estimations (int): number of samples to be used for computing `pyc` (`M` in Eq. 1 of the paper).
                               If `num_estimations` <= 0, all the samples will be used to compute the marginalization,
                               then avoiding the need to compute running statistics as in Eq. 3 of the paper
    Returns:
        torch.Tensor: calibrated probabilities
    """
    # Compute pyc from Eq. 1, using the existing logits
    pyc = logits.clone()
    if num_estimations > 0:
        pyc = pyc[:num_estimations, :]
    pyc = pyc.mean(dim=0)

    log(_logger.info, f"`pyc` of batch calibration is: {pyc}", "green")

    # Calibrate the logits as in Eq. 2 with "unsupervised" batch calibration
    # Softmax is not needed, but used to be consistent w/ the other calibrations
    return F.softmax(logits - pyc, dim=-1)


def adjustable_batch_calibration(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    instruction: str,
    input_columns: list[str],
    output_column: str,
    random_seed: int,
    max_seq_length: int,
    max_input_length: int,
    max_output_length: int,
    icl_examples: int,
    labels: list[str],
    batch_size: int,
    logits: torch.Tensor,
    forward_fn: Callable,
    num_estimations: int,
    num_train_samples: int,
    random_icl: bool,
    gamma_range: list[float] = [-5, 5],
    gamma_steps: int = 50,
    **kwargs,
) -> torch.Tensor:
    """
    Implements adjustable batch calibration from the paper https://arxiv.org/pdf/2309.17249
    It uses the same number of IC examples as used for classification, that means that,
    if you use 2 IC examples for classification, 2 IC examples will be used too to
    perform the calibration.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        instruction (str): instruction of the prompt
        input_columns (list[str]): columns of the input text to fill the prompt
        max_seq_length (int): maximum length of the whole prompt. If longer, the prompt will be truncated
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        labels (list[str]): list of labels in the dataset
        batch_size (int): samples per batch
        logits (torch.Tensor): logits to be calibrated
        forward_fn (Callable): the classification forward function to perform a forward pass with context-free inputs
        num_estimations (int): number of test samples to be used for computing `pyc` (See §4 of the paper)
        num_train_samples (int): number of train samples to be used for computing `gamma` (See §4 of the paper)
        random_icl (bool): whether if the IC examples should be random
        gamma_range (list[float, float]): range of possible `gamma` values to run in the grid search
        gamma_steps (int): num of steps to take between the `gamma_range` values
    Returns:
        torch.Tensor: calibrated probabilities
    """
    assert (
        train_dataset
    ), "A train dataset is needed for adjustable batch calibration"

    # Prepare the training dataset with chat templates
    train_dataset = train_dataset.select(range(num_train_samples))

    format_prompt_fn = partial(
        format_prompt,
        instruction=instruction,
        input_columns=input_columns,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        icl_examples=icl_examples,
        task_type="classification",
        random_seed=random_seed,
        output_column=output_column,
        max_output_length=max_output_length,
        add_output=False,
        random_icl=random_icl,
    )

    prompted_dataset = train_dataset.map(
        lambda examples: {"prompts": format_prompt_fn(examples)}, batched=True
    )

    tokenized_dataset = prompted_dataset.map(
        lambda examples: tokenizer(
            examples["prompts"], truncation=True, max_length=max_seq_length
        ),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.select_columns(
        ["input_ids", "attention_mask"]
    )

    # Compute the logits of the training set
    train_logits = forward_fn(
        model, tokenizer, tokenized_dataset, labels, batch_size
    )

    # Compute pyc from Eq. 1, using the existing logits
    pyc = logits.clone()
    if num_estimations > 0:
        pyc = pyc[:num_estimations, :]
    pyc = pyc.mean(dim=0)

    log(_logger.info, f"`pyc` of adjusted batch calibration is: {pyc}", "green")

    # Run grid search to determine `gamma` on the `train_dataset`
    best_score, best_gamma = -1, gamma_range[0]
    for gamma in tqdm(
        np.linspace(*gamma_range, num=gamma_steps), desc="Computing gamma..."
    ):
        pbcl = train_logits - gamma * pyc
        label_preds = [labels[idx] for idx in pbcl.argmax(-1)]
        score = f1_score(
            y_true=train_dataset[output_column],
            y_pred=label_preds,
            average="macro",
        )
        if score > best_score:
            best_score, best_gamma = score, gamma

    log(
        _logger.info,
        f"The best `gamma` found is {best_gamma} with mf1: {best_score}",
        "green",
    )

    # Calibrate the logits as in Eq. 4 with adjusted batch calibration
    # Softmax is not needed, but used to be consistent w/ the other calibrations
    return F.softmax(logits - best_gamma * pyc, dim=-1)
