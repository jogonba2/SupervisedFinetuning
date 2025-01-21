from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def load_dataset_from_config(path: str, args: dict) -> Dataset | DatasetDict:
    """
    Loads a HF dataset or dataset dict from a path.

    Args:
        path (str): path to the dataset
        args (dict): dict of args to be passed when calling `load_dataset`

    Returns:
        Dataset | DatasetDict: a HF dataset.
    """
    try:
        dataset = load_from_disk(path)
    except FileNotFoundError:
        dataset = load_dataset(path, **args)
    return dataset


def get_shots(
    dataset: Dataset,
    output_column: str,
    shots: int,
    task_type: Literal["classification", "generation"],
    random_seed: int,
    random_icl: bool,
) -> Dataset:
    """
    Sample shots from a dataset.
    For `generation`, select `shots` random samples.
    For `classification`, select `shots` random samples per label.

    Args:
        dataset (Dataset): a dataset
        output_column (str): column containing the outputs
        shots (int): number of shots to sample
        task_type (Literal): either `classification` or `generation`
        random_seed (int): a random seed
        random_icl (bool): whether if the IC examples should be random for each example

    Returns:
        Dataset: a dataset with sampled shots
    """
    if random_icl:
        dataset = dataset.shuffle(seed=random_seed)
    if task_type == "generation":
        return dataset.select(range(shots))
    else:
        return Dataset.from_pandas(
            dataset.to_pandas()
            .groupby(output_column)
            .sample(
                shots,
                replace=True,
                random_state=None if random_icl else random_seed,
            ),
            preserve_index=False,
        )
