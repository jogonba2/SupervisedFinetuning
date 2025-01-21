from functools import partial

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..prompting.prompt import format_prompt


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    instruction: str,
    input_columns: list[str],
    max_input_length: int,
    batch_size: int,
    max_seq_length: int,
    icl_examples: int,
    random_seed: int,
    output_column: str,
    max_output_length: int,
    generation_args: dict,
) -> dict[str, list[str] | torch.Tensor]:
    """
    Function to generate text using LLMs.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        dataset (Dataset): a test dataset
        instruction (str): instruction of the prompt
        input_columns (list[str]): columns of the input text to fill the prompt
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        batch_size (int): samples per batch
        max_seq_length (int): maximum length of the whole prompt. If longer, the prompt will be truncated
        icl_examples (int): number of in-context examples to add in the prompts for inference
        random_seed (int): a random seed to get icl examples
        output_column (str): column to use as label in in-context examples
        max_output_length (int): maximum length of the output text in in-context examples
        generation_args (dict): arguments to be passed to the `generate` method of HuggingFace

    Returns:
        dict[str, list[str] | torch.Tensor]: pred text and probabilities
    """
    # Format prompts with chat templates
    format_prompt_fn = partial(
        format_prompt,
        instruction=instruction,
        input_columns=input_columns,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        icl_examples=icl_examples,
        task_type="generation",
        random_seed=random_seed,
        output_column=output_column,
        max_output_length=max_output_length,
        add_output=False,
    )

    dataset = dataset.map(
        lambda examples: {"prompts": format_prompt_fn(examples)}, batched=True
    )

    # Tokenize
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["prompts"], truncation=True, max_length=max_seq_length
        ),
        batched=True,
    )
    dataset = dataset.select_columns(["input_ids", "attention_mask"])
    collator = DataCollatorWithPadding(tokenizer, padding=True)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collator
    )

    # empty prediction counter
    empty_predictions = 0

    # Generate
    model.eval()
    with torch.inference_mode():
        predictions = []
        for batch in tqdm(data_loader, desc="Generating completions..."):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            completion_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_args,
            )

            for idx, completion in enumerate(completion_ids):
                prediction = tokenizer.decode(
                    completion[len(input_ids[idx]) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                if not prediction:
                    prediction = "unknown"
                    empty_predictions += 1
                predictions.append(prediction)

    print(f"Empty predictions: {empty_predictions}")

    # TODO: implement `probs`
    return {"preds": predictions}
