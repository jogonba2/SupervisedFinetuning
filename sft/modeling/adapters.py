from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from ..initialization import custom_init_lora, initializer_registry
from ..utils import get_logger, log

_logger = get_logger(__name__)


def add_lora_to_model(model: PreTrainedModel, lora_args: dict):
    """
    Function to add LoRA adapters into a model, potentially
    initialized from custom initializations.

    Args:
        model (PreTrainedModel): a model
        lora_args (dict): arguments to be passed to LoRA instantiation
    """
    # Change `init_lora_weights` to True if the initializer is custom
    # since the weights will be replaced later
    custom_init = None
    if "init_lora_weights" in lora_args:
        if lora_args["init_lora_weights"] in initializer_registry:
            custom_init = lora_args["init_lora_weights"]
            lora_args["init_lora_weights"] = True

    lora_config = LoraConfig(**lora_args)
    model = get_peft_model(model, lora_config)

    if custom_init:
        log(
            _logger.info,
            f"Using a custom initialization for lora, with `{custom_init}`",
            "blue",
        )
        model = custom_init_lora(
            model=model,
            lora_args=lora_args,
            init_fn_name=custom_init,
        )
        lora_args["init_lora_weights"] = custom_init

    model.print_trainable_parameters()
    return model
