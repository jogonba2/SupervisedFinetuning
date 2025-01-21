import random

import numpy as np
import torch


def fix_seed(seed: int = 13) -> None:
    """
    Fix a global random seed in several libraries.

    Args:
        seed (int): a random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
