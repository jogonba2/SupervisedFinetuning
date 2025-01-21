# flake8: noqa
from typing import Callable

from .pca import pca
from .pissa import pissa
from .umap import umap

# Register here new initializers
initializer_registry: dict[str, Callable] = {
    "pca": pca,
    "pissa": pissa,
    "umap": umap,
}
