import gc
import hashlib
import json
from time import sleep

import torch
from accelerate.utils import release_memory


def hash_dict(d: dict) -> str:
    """
    Hashes a dict.

    Args:
        d (dict): a dictionary

    Returns:
        str: a hash of the dictionary
    """
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def flush_gpu(sleep_seconds: int = 10) -> None:
    """
    (Tries to) flush the GPU memory.

    Args:
        sleep_seconds (int): sleep seconds after cleaning memory.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    release_memory()
    sleep(sleep_seconds)


def get_nested_attr(obj, attr_chain: str):
    """
    Gets an attribute from the object `obj` given the
    required attribute chain to access it.

    Example:
        obj = nn.Module
        attr_chain = "model.layers.classifier
        result = Classifier

    Args:
        obj (object): an object
        attr_chain (str): a chain to reach the attribute

    Result:
        object: any kind of object-type attribute getted from `obj`
    """
    attrs = attr_chain.split(".")
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj
