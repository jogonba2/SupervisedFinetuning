from functools import partial
from typing import Optional

import torch
from datasets import Dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..prompting.prompt import format_prompt
from .calibration import calibration_registry


def classify(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    instruction: str,
    input_columns: list[str],
    labels: list[str],
    max_input_length: int,
    batch_size: int,
    max_seq_length: int,
    icl_examples: int,
    random_seed: int,
    output_column: str,
    max_output_length: int,
    calibration_args: dict = {},
    train_dataset: Optional[Dataset] = None,
    random_icl: bool = True,
) -> dict[str, torch.Tensor | list[str]]:
    """
    Function to classify text using logits from LLMs.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        dataset (Dataset): a test dataset
        instruction (str): instruction of the prompt
        input_columns (list[str]): columns of the input text to fill the prompt
        labels (list[str]): list of labels in the dataset
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        batch_size (int): samples per batch
        max_seq_length (int): maximum length of the whole prompt. If longer, the prompt will be truncated
        icl_examples (int): number of in-context examples to add in the prompts for inference
        random_seed (int): a random seed to get in-context examples
        output_column (str): column to use as label in in-context examples
        max_output_length (int): maximum length of the output text in in-context examples
        calibration_args (dict): arguments to perform calibration if needed
        train_dataset (Optional[Dataset]): dataset with labels to be used in some calibration methods.
        random_icl (bool): whether if the IC examples should be random for each example

    Returns:
        dict[str, list[str] | torch.Tensor]: pred label and probabilities
    """
    # Format prompts with chat templates
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

    prompted_dataset = dataset.map(
        lambda examples: {"prompts": format_prompt_fn(examples)}, batched=True
    )

    # Tokenize inputs
    tokenized_dataset = prompted_dataset.map(
        lambda examples: tokenizer(
            examples["prompts"], truncation=True, max_length=max_seq_length
        ),
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.select_columns(
        ["input_ids", "attention_mask"]
    )

    # Get the logits
    logits = forward(model, tokenizer, tokenized_dataset, labels, batch_size)

    # Calibrate (all calibrations must return probabilities)
    if calibration_args:
        calibration_fn = calibration_registry[
            calibration_args["calibration_fn"]
        ]
        probs = calibration_fn(
            model=model,
            tokenizer=tokenizer,
            instruction=instruction,
            prompted_dataset=prompted_dataset,
            input_columns=input_columns,
            output_column=output_column,
            random_seed=random_seed,
            max_seq_length=max_seq_length,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            labels=labels,
            batch_size=batch_size,
            logits=logits,
            forward_fn=forward,
            train_dataset=train_dataset,
            icl_examples=icl_examples,
            random_icl=random_icl,
            **calibration_args,
        )
    # If not calibrating, forward returns logits -> compute softmax
    else:
        probs = F.softmax(logits, dim=-1)

    # Get the predictions
    preds = probs.argmax(-1)
    label_preds = [labels[idx] for idx in preds]

    return {"probs": probs, "preds": label_preds}


def forward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    tokenized_dataset: Dataset,
    labels: list[str],
    batch_size: int,
) -> torch.Tensor:
    """
    Efficient forward pass for classification.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        tokenizer_dataset (Dataset): an already tokenized test dataset
        labels (list[str]): list of labels in the dataset
        batch_size (int): samples per batch
    Returns:
        torch.Tensor: logits outputed by the model
    """
    # Tokenize the labels
    # Temporary set padding to right before tokenizing and restore it
    tokenizer.padding_side = "right"
    tokenized_labels = tokenizer(labels, add_special_tokens=False, padding=True)
    tokenizer.padding_side = "left"

    # Prepare the labels
    label_ids = torch.LongTensor(tokenized_labels["input_ids"]).to(model.device)

    collator = DataCollatorWithPadding(tokenizer, padding=True)
    data_loader = DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=collator
    )

    # Forward pass for classification
    model.eval()
    with torch.inference_mode():
        output_logits = []
        for batch in tqdm(data_loader, desc="Classifying..."):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            # (batch_size, num_labels, label_ids)
            logits_per_label = torch.zeros(
                (input_ids.shape[0], *label_ids.shape)
            ).to(model.device)

            # Go label by label to compute the logits of each label token
            for i, l_ids in enumerate(label_ids):
                past_key_values = output.past_key_values
                # Compute the logit of the first label token for each sample in the batch
                logits_per_label[:, i, 0] = output.logits[:, -1, l_ids[0]]
                # Compute the logits of the remaining tokens of the label `i`
                for j in range(1, l_ids.shape[0]):
                    curr_id = l_ids[j]

                    # Skip the padding
                    if curr_id == tokenizer.pad_token_id:
                        continue

                    label_output = model(
                        input_ids=curr_id.repeat(input_ids.shape[0], 1).to(
                            model.device
                        ),
                        past_key_values=past_key_values,
                    )
                    logits_per_label[:, i, j] = label_output.logits[
                        :, -1, curr_id
                    ]
                    past_key_values = label_output.past_key_values

            # Average the logits (exclude 0 from padding)
            mask = logits_per_label != 0
            mean_logits = (logits_per_label * mask).sum(dim=-1) / mask.sum(
                dim=-1
            )

            # Accumulate outputs
            output_logits.append(mean_logits.detach().cpu())

        # Stack outputs
        output_logits = torch.vstack(output_logits)

    return output_logits
