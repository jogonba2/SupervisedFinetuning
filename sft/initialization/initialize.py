import re

import torch
from transformers import PreTrainedModel

from ..utils import get_logger, get_nested_attr, log
from .initializers import initializer_registry
from .utils import get_from_cache, store_in_cache

_logger = get_logger(__name__)


def custom_init_lora(
    model: PreTrainedModel,
    lora_args: dict,
    init_fn_name: str,
    use_cache: bool = True,
) -> PreTrainedModel:
    """
    Initializes the LoRA weights of a model.

    Args:
        model (PreTrainedModel): a model with LoRA adapters included
        lora_args (dict): arguments of the lora adapter used
        init_fn_name (str): function name of an initializer in `initializers`
        use_cache (bool): whether the initializations will be stored/loaded from cache or not.

    Returns:
        PreTrainedModel: `model` with the LoRA weights initialized
                         using a custom initializer.
    """
    init_fn = initializer_registry[init_fn_name]

    for target_module in lora_args["target_modules"]:
        pattern = rf"base_model\.model\.model\.layers\.\d+\..*\.{target_module}\.base_layer"
        for module_idx, (name, _) in enumerate(model.named_modules()):
            match = re.fullmatch(str(pattern), str(name))
            if match:
                match_string = match.string
                _, right_side = match_string.split(".layers.")

                # Get the original weight matrix
                layer, right_side = right_side.split(".", 1)
                base_block = model.base_model.model.model.layers[int(layer)]
                base_layer = get_nested_attr(base_block, right_side)

                # Get lora A matrix
                lora_a_right_side = right_side.rsplit(".", 1)[0] + ".lora_A"
                lora_a_module = get_nested_attr(base_block, lora_a_right_side)

                # Get lora B matrix
                lora_b_right_side = right_side.rsplit(".", 1)[0] + ".lora_B"
                lora_b_module = get_nested_attr(base_block, lora_b_right_side)

                # Get initialization from cache if existing
                if use_cache:
                    lora_weights = get_from_cache(
                        model, lora_args, init_fn_name, module_idx
                    )
                else:
                    lora_weights = {}

                # Get A and B matrices from the initializer if not cached
                if not lora_weights:
                    lora_weights = init_fn(
                        base_layer,
                        lora_args["r"],
                        dtype=lora_a_module.default.weight.dtype,
                    )
                    # Store initialization in cache
                    if use_cache:
                        store_in_cache(
                            model,
                            lora_args,
                            init_fn_name,
                            lora_weights,
                            module_idx,
                        )
                else:
                    log(
                        _logger.info,
                        f"Module {module_idx} initialization loaded from cache.",
                        "blue",
                    )

                # Set LoRA A and B matrix weights
                lora_a_module.default.weight = torch.nn.Parameter(
                    lora_weights["lora_A"]
                )
                lora_b_module.default.weight = torch.nn.Parameter(
                    lora_weights["lora_B"]
                )

    return model
