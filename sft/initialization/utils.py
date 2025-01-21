import os
from pathlib import Path
from typing import TypeAlias

import torch
from awq.utils.packing_utils import dequantize_gemm
from torch import nn
from transformers import PreTrainedModel

from ..utils import hash_dict

T: TypeAlias = torch.Tensor


def dequantize(layer: nn.Module) -> T:
    """
    Dequantizes the weights of a layer.
    Only supports `bitsandbytes` (int8) and `AWQ` quantized models.

    Args:
        layer (nn.Module): a layer from a torch model

    Returns:
        T: torch tensor containing the dequantized weights
    """
    is_awq = hasattr(layer, "qweight")
    is_bitsandbytes = hasattr(layer, "weight") and hasattr(layer.weight, "SCB")

    if is_awq:
        return dequantize_gemm(
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.w_bit,
            layer.group_size,
        )
    if is_bitsandbytes:
        return (layer.weight.SCB.unsqueeze(1) * layer.weight) / 127
    return layer.weight


def get_cache_folder(
    model: PreTrainedModel, lora_args: dict, init_fn_name: str, module_idx: int
) -> str:
    """
    Get the cache folder for a model, lora args, initialization, and adapted module.

    Args:
        model (PreTrainedModel): a huggingface model
        lora_args (dict): arguments of the lora adapter used
        init_fn_name (str): name of the custom initialization function
        module_idx (int): index in the model of the adapted module

    Returns:
        str: path to the cache
    """
    cache_folder = os.environ.get("HF_DATASETS_CACHE", "./")
    lora_hash = hash_dict(lora_args)
    model_hash = hash_dict(model.config.to_dict())
    return (
        f"{cache_folder}/{model_hash}-{lora_hash}-{init_fn_name}-{module_idx}/"
    )


def get_from_cache(
    model: PreTrainedModel, lora_args: dict, init_fn_name: str, module_idx: int
) -> dict[str, T]:
    """
    Gets the initialization of the module from the cache.

    Args:
        model (PreTrainedModel): a huggingface model
        lora_args (dict): arguments of the lora adapter used
        init_fn_name (str): name of the custom initialization function
        module_idx (int): index in the model of the adapted module

    Returns:
        dict[str, T]: dictionary mapping layers to weights, e.g., {"lora_A": T, "lora_B": T}
    """
    folder_name = get_cache_folder(model, lora_args, init_fn_name, module_idx)
    path = Path(folder_name) / "tensors.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return torch.load(path)
    return {}


def store_in_cache(
    model: PreTrainedModel,
    lora_args: dict,
    init_fn_name: str,
    lora_weights: dict[str, T],
    module_idx: int,
) -> None:
    """
    Stores the initialization of the module in the cache.

    Args:
        model (PreTrainedModel): a huggingface model
        lora_args (dict): arguments of the lora adapter used
        init_fn_name (str): name of the custom initialization function
        lora_weights (dict[str, T]): dictionary mapping layers
                             to weights, e.g., {"lora_A": T, "lora_B": T}
        module_idx (int): index in the model of the adapted module
    """
    folder_name = get_cache_folder(model, lora_args, init_fn_name, module_idx)
    path = Path(folder_name) / "tensors.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lora_weights, path)
