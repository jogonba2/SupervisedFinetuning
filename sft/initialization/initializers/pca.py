import torch
from torch import nn

from ...utils import get_logger, log
from ..utils import dequantize

_logger = get_logger(__name__)


def pca(
    layer: nn.Module, rank: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """
    Initializer that returns the principal components from the
    non-quantized weight matrix to initialize LoRA_A, and zero values
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

    # SVD needs full precision
    # Offload the full precision matrix to CPU to free memory in GPU
    # This makes the process much slower, but safer for GPU memory.
    # Anyway, the inits are cached per model, so this has to be done
    # only the first time a initialization is used for a model.
    original_device = dequantized_weights.device
    dequantized_weights = dequantized_weights.to(dtype=torch.float32).to("cpu")

    # Compute SVD
    U, _, _ = torch.linalg.svd(dequantized_weights, full_matrices=False)

    # Setup lora_A and lora_B matrices
    lora_A = U[:, :rank]
    lora_A = lora_A.to(dtype=dtype).T.contiguous()
    lora_B = (
        torch.zeros((dequantized_weights.shape[1], rank))
        .to(dtype=dtype)
        .contiguous()
    )

    # Quantify the variance ratio
    total_var = torch.var(dequantized_weights)
    total_var_approx = torch.var(lora_A)
    cumulative_percent_variance = (total_var / total_var_approx) * 100
    log(
        _logger.info,
        f"Variance ratio covered (%): {cumulative_percent_variance.item()}",
        "blue",
    )

    return {
        "lora_A": lora_A.to(original_device),
        "lora_B": lora_B.to(original_device),
    }
