from functools import partial
from typing import Literal

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from ..prompting.prompt import format_prompt
from ..prompting.utils import infer_response_prefix_ids


def fit(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    instruction: str,
    input_columns: list[str],
    output_column: str,
    max_input_length: int,
    max_output_length: int,
    icl_examples: int,
    task_type: Literal["classification", "generation"],
    random_seed: int,
    trainer_args: dict,
) -> PreTrainedModel:
    """
    Function to fit LLMs for supervised finetuning.

    Args:
        model (PreTrainedModel): a model
        tokenizer (PreTrainedTokenizer): a tokenizer
        dataset (Dataset): a training dataset
        instruction (str): instruction of the prompt
        input_columns (list[str]): columns of the input text to fill the prompt
        output_column (str): column of the output text to fill the prompt
        max_input_length (int): maximum length of the input text. If longer, the input text will be truncated
        max_output_length (int): maximum length of the output text. If longer, the output text will be truncated
        icl_examples (int): number of in-context examples to add in the prompts for training
        task_type (Literal): either `classification` or `generation` to get icl examples
        random_seed (int): a random seed to get icl examples
        trainer_args (dict): arguments to be passed to `SFTrainer` of HuggingFace
    """
    # Prepare prompting fn (chat templates)
    format_prompt_fn = partial(
        format_prompt,
        instruction=instruction,
        input_columns=input_columns,
        output_column=output_column,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        icl_examples=icl_examples,
        task_type=task_type,
        random_seed=random_seed,
        add_output=True,
    )

    # Prepare the collator
    collator = DataCollatorForCompletionOnlyLM(
        infer_response_prefix_ids(tokenizer),
        tokenizer=tokenizer,
        mlm=False,
    )

    # Disable `use_cache` when `gradient_checkpointing`
    if trainer_args.get("gradient_checkpointing", False):
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.use_cache = False

    # Prepare the trainer
    sft_config = SFTConfig(**trainer_args)
    model.train()

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=format_prompt_fn,
        data_collator=collator,
    )

    # Let's train
    trainer.train()

    # Enable use cache always at the end
    model.config.use_cache = True
    model.use_cache = True

    return model
