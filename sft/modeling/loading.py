import importlib
from typing import Optional

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from ..utils import get_logger, log

_logger = get_logger(__name__)


def prepare_quantization(quantization: Optional[dict] = {}) -> dict:
    """
    Prepares quantization args to be passed to the `from_pretrained` method.

    Args:
        quantization (Optional[dict]): dictionary with quantization config class
                                       and args to be passed to the config object.

    Returns:
        dict: a dict with quantization args prepared to call `from_pretrained`
    """
    quantization_config = {}
    if quantization:
        # Config class is mandatory
        if "config_class" not in quantization:
            raise ValueError(
                "When using quantization, you must specify the quantization"
                " config, e.g., `BitsAndBytesConfig` or `GPTQConfig` in"
                " the `config_class` key."
            )
        quantization_args = quantization.get("args", {})
        quantization_class = getattr(
            importlib.import_module("transformers"),
            quantization["config_class"],
        )
        quantization_config = {
            "quantization_config": quantization_class(**quantization_args)
        }
    return quantization_config


def load_model(
    model_name: str,
    model_args: dict = {},
    quantization: dict = {},
) -> PreTrainedModel:
    """
    Loads a model.

    Args:
        model_name (str): name or path of the model
        model_args (Optional[dict]): args to be passed to the `from_pretrained` method
        quantization (Optional[dict]): dictionary with quantization config class
                                       and args to be passed to the config object.

    Returns:
        PreTrainedModel: loaded model
    """
    # Prepare quantization config if required
    quantization_config = prepare_quantization(quantization)
    parallelize = model_args.get("parallelize")

    if parallelize:
        # split weights across gpus available
        weights_location = snapshot_download(repo_id=model_name)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        with init_empty_weights():
            # duplicate trust_remote_code to avoid errors
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **quantization_config,
            )

        # tie the weights (avoid OOM errors)
        model.tie_weights()

        # load model
        no_split_module_classes = model_args.get("no_split_module_classes", [])
        model = load_checkpoint_and_dispatch(
            model,
            weights_location,
            device_map="balanced",
            no_split_module_classes=no_split_module_classes,
        )
    else:
        # Instantiate the model (all to the first GPU, use CUDA_VISIBLE_DEVICES)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda:0",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **quantization_config,
            **model_args,
        )

    return model


def load_tokenizer(
    model_name: str, chat_template_from: Optional[str] = None
) -> PreTrainedTokenizer:
    """
    Loads a tokenizer.

    Args:
        model_name (str): name of the model.
        chat_template_from (Optional[str]): name of a model from which to use the chat template.
                             Especially useful for cases where a third-party model
                             has some bug in the template, but the original model
                             already fixed it, e.g., casperhansen/llama-3-70b-instruct-awq
                             has an issue related to the `add_generation_prompt` arg
                             in the chat template, but not meta-llama/Meta-Llama-3-70B-instruct.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if chat_template_from:
        log(
            _logger.info,
            f"Replacing the template of {model_name} by {chat_template_from}",
            "yellow",
        )
        other_tokenizer = AutoTokenizer.from_pretrained(chat_template_from)
        tokenizer.chat_template = other_tokenizer.chat_template
    return tokenizer
