import torch
from torch import nn

from ...utils import get_logger, log
from ..utils import dequantize

_logger = get_logger(__name__)


def pissa(
    layer: nn.Module, rank: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """
    Simplification from https://arxiv.org/pdf/2404.02948,
    without considering the matrix of residuals and not
    sharing the random A and B matrices across the model.

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

    # SVD needs full precision
    # Offload the full precision matrix to CPU to free memory in GPU
    # This makes the process much slower, but safer for GPU memory.
    # Anyway, the inits are cached per model, so this has to be done
    # only the first time a initialization is used for a model.
    original_device = dequantized_weights.device
    dequantized_weights = dequantized_weights.to(dtype=torch.float32).to("cpu")

    # Compute SVD
    U, S, V = torch.linalg.svd(dequantized_weights, full_matrices=False)

    # Setup lora_A and lora_B matrices
    lora_A = (
        (U[:, :rank] @ torch.diag(S.sqrt())[:rank, :rank])
        .to(dtype=dtype)
        .T.contiguous()
    )
    lora_B = (
        (torch.diag(S.sqrt())[:rank, :rank] @ V[:, :rank].T)
        .to(dtype=dtype)
        .T.contiguous()
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
