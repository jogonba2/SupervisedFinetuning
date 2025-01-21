import torch
from torch import nn
from umap import UMAP

from ...utils import get_logger, log
from ..utils import dequantize

_logger = get_logger(__name__)


def umap(
    layer: nn.Module, rank: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """
    Initializer that returns the non-quantized weight matrix
    reduced using UMAP to initialize LoRA_A, and zero values
    to initialize LoRA_B

    Args:
        layer (nn.Module): layer from a model
        rank (int): lora rank
        dtype (torch.dtype): dtype to cast the initialization

    Returns:
        dict[str, torch.Tensor]: dictionary mapping layers
          to weights, e.g., {"lora_A": T, "lora_B": T}
    """
    # Dequantize the layer weights
    dequantized_weights = dequantize(layer).T

    # UMAP needs full precision
    # Offload the full precision matrix to CPU to free memory in GPU
    # This makes the process much slower, but safer for GPU memory.
    # Anyway, the inits are cached per model, so this has to be done
    # only the first time a initialization is used for a model.
    original_device = dequantized_weights.device
    dequantized_weights = dequantized_weights.to(dtype=torch.float32).to("cpu")

    # Compute UMAP reduction
    reducer = UMAP(n_components=rank)
    reduced = torch.from_numpy(reducer.fit_transform(dequantized_weights))

    # Setup lora_A and lora_B matrices
    lora_A = reduced.to(dtype=dtype).T.contiguous()
    lora_B = (
        torch.zeros((dequantized_weights.shape[1], rank))
        .to(dtype=dtype)
        .contiguous()
    )

    log(
        _logger.info,
        f"Built initialization for: {layer}",
        "blue",
    )

    return {
        "lora_A": lora_A.to(original_device),
        "lora_B": lora_B.to(original_device),
    }
