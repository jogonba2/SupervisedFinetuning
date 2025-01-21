from typing import Literal

from datasets import Dataset, disable_caching
from transformers import PreTrainedTokenizerBase

from ..utils import clean_text, get_logger, get_shots, log, truncate

disable_caching()

_logger = get_logger(__name__)


def format_prompt(
    examples: dict[str, list[str]],
    instruction: str,
    input_columns: list[str],
    output_column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_output_length: int,
    icl_examples: int,
    task_type: Literal["classification", "generation"],
    random_seed: int,
    add_output: bool,
    random_icl: bool,
) -> list[str]:
    """
    Format the prompt using chat template from the LLMs tokenizer.
    Allows to add in-context examples, truncate them, use multiple
    input columns, and works both for training and inference.

    Args:
        examples (dict[str, list[str]]): a batch of examples from a dataset
        instruction (str): instruction to add to the prompt
        input_columns (list[str]): input columns whose texts are used as inputs in the prompt
        output_column (str): output column to be appended to the prompt if `add_output=True`
        tokenizer (PreTrainedTokenizer): a tokenizer
        max_input_length (int): maximum length of each input text. Each input text
                                will be truncated to the maximum length.
        max_output_length (int): maximum length of each output text. The output will be
                                 truncated to the maximum length.
        icl_examples (int): number of in-context examples to add to the prompt.
                            These examples are considered inputs, so the same truncation
                            rules are applied both for the input and for these examples.
        task_type (Literal): either `classification` or `generation`
        random_seed (int): a random seed
        add_output (bool): whether to add the output of each example at the end of the prompt.
                           As rule of thumb, set `add_output=True` for training, and
                           `add_output=False` for inference.
        random_icl (bool): whether if the IC examples should be random for each example

    Returns:
        list[str]: list of formatted prompts for the `examples`
    """
    prompts = []

    for idx, output in enumerate(examples[output_column]):
        # Get texts from input columns
        inputs = [
            clean_text(examples[input_column][idx])
            for input_column in input_columns
        ]

        # Truncate inputs
        truncated_inputs = [
            truncate(input, tokenizer, max_input_length) for input in inputs
        ]

        # Join the inputs, by prepending input column names
        # if there are more than 1 input columns.
        # Otherwise, use only the text
        if len(input_columns) > 1:
            truncated_input = "\n".join(
                f"{input_column.capitalize()}: {text}"
                for input_column, text in zip(input_columns, truncated_inputs)
            )
        else:
            truncated_input = truncated_inputs[0]

        # Add instruction to the messages.
        messages = [{"role": "system", "content": instruction}]

        # Add in-context examples
        if icl_examples > 0:
            icl_messages = get_icl_messages(
                input_texts=inputs,
                dataset=Dataset.from_dict(examples),
                input_columns=input_columns,
                output_column=output_column,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                icl_examples=icl_examples,
                task_type=task_type,
                random_seed=random_seed,
                tokenizer=tokenizer,
                random_icl=random_icl,
            )
            messages += icl_messages

        # Add current input
        messages.append({"role": "user", "content": truncated_input})

        # Add output if specified, e.g., when training
        if add_output:
            truncated_output = truncate(
                clean_text(output), tokenizer, max_output_length
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": truncated_output,
                }
            )

        # Instantiate the template with the messages for this example
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=not add_output
        )

        prompts.append(formatted_prompt)

    log(
        _logger.info,
        f"This is how one prompt looks like (`add_output={add_output}`):\n{prompts[0]}",
        "blue",
    )
    return prompts


def get_icl_messages(
    input_texts: list[str],
    dataset: Dataset,
    input_columns: list[str],
    output_column: str,
    max_input_length: int,
    max_output_length: int,
    icl_examples: int,
    task_type: Literal["classification", "generation"],
    random_seed: int,
    tokenizer: PreTrainedTokenizerBase,
    random_icl: bool,
) -> list[dict[str, str]]:
    """
    Samples in-context examples, truncates, and formats them
    as a list of HF messages to be used in a chat template.

    Args:
        input_texts (list[str]): list of inputs from `input_columns`
                                 that must not appear as in-context examples
                                 to avoid leakage
        dataset (Dataset): a dataset from where to sample in-context examples
        input_columns (list[str]): input columns whose texts are used as inputs in the prompt
        output_column (str): output column to be appended to the prompt if `add_output=True`
        max_input_length (int): maximum length of each in-context example input.
                                Each example will be truncated to the maximum length.
        max_output_length (int): maximum length of each in-context example output.
                                 The output will be truncated to the maximum length.
        icl_examples (int): number of in-context examples.
        task_type (Literal): either `classification` or `generation`
        random_seed (int): a random seed
        tokenizer (PreTrainedTokenizer): a tokenizer
        random_icl (bool): whether if the IC examples should be random for each example

    Returns:
        list[dict[str, str]]: in-context examples as a list of messages, e.g.,
           [{"role": "user", "content": "This is the input of the 1st IC},
            {"role": "assistant", "content": "This is the output of the 1st IC},
            {"role": "user", "content": "This is the input of the 2nd IC},
            {"role": "assistant", "content": "This is the output of the 2nd IC}]
    """
    icl_dataset = get_shots(
        dataset=dataset,
        output_column=output_column,
        shots=icl_examples,
        task_type=task_type,
        random_seed=random_seed,
        random_icl=random_icl,
    )

    icl_messages = []

    for idx, example_output in enumerate(icl_dataset[output_column]):
        # Get texts from input columns
        inputs = [
            clean_text(icl_dataset[idx][input_column])
            for input_column in input_columns
        ]

        # Skip the in-context example if it
        # matches the example to be predicted
        if inputs == input_texts:
            continue

        # Truncate inputs and outputs
        truncated_inputs = [
            truncate(input, tokenizer, max_input_length) for input in inputs
        ]
        truncated_output = truncate(
            clean_text(example_output), tokenizer, max_output_length
        )

        # Join the inputs, by prepending input column names
        # if there are more than 1 input columns.
        # Otherwise, use only the text
        if len(input_columns) > 1:
            truncated_input = "\n".join(
                f"{input_column.capitalize()}: {text}"
                for input_column, text in zip(input_columns, truncated_inputs)
            )
        else:
            truncated_input = truncated_inputs[0]

        # Add messages
        icl_messages.append({"role": "user", "content": truncated_input})
        icl_messages.append(
            {
                "role": "assistant",
                "content": truncated_output,
            }
        )

    return icl_messages
