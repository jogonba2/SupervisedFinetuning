from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel

from .modeling import (
    add_lora_to_model,
    classify,
    fit,
    generate,
    load_model,
    load_tokenizer,
)
from .utils import (
    fix_seed,
    get_logger,
    get_shots,
    load_dataset_from_config,
    log,
)

_logger = get_logger(__name__)


def train(config: dict, random_seed: int) -> PreTrainedModel:
    """
    Train a model given a configuration.

    Args:
        config (dict): a config dict
        random_seed (int): a random seed

    Returns:
        PreTrainedModel: a model
    """
    # Fix random seed
    fix_seed(random_seed)

    # Info the user if shots <=0 that model won't be trained
    shots = config["dataset"]["shots"]

    if shots <= 0:
        log(
            _logger.info,
            f"You passed `shots={shots}`. The model won't be trained.",
            "yellow",
        )

    # Load the dataset
    dataset = load_dataset_from_config(
        path=config["dataset"]["path"], args=config["dataset"].get("args", {})
    )

    # Sample shots for training
    if shots > 0:
        dataset["train"] = get_shots(
            dataset["train"],
            config["prompting"]["output_column"],
            shots,
            config["dataset"]["task_type"],
            random_seed,
            random_icl=True,
        )

    # Instantiate the model and the tokenizer
    model = load_model(
        model_name=config["model"]["name"],
        model_args=config["model"].get("model_args", {}),
        quantization=config["model"].get("quantization", {}),
    )

    tokenizer = load_tokenizer(
        config["model"]["name"],
        chat_template_from=config["model"].get("chat_template_from", None),
    )

    # Add LoRA adapters to the model just for few-shot settings
    if shots > 0:
        model = add_lora_to_model(model=model, lora_args=config["lora"])
        log(
            _logger.info,
            f"Trainable parameters: {model.get_nb_trainable_parameters()[0]}",
            "blue",
        )

    # Fit the model
    if shots > 0:
        model = fit(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset["train"],
            **config["prompting"],
            icl_examples=config["training"].get("icl_examples", 0),
            random_seed=random_seed,
            task_type=config["dataset"]["task_type"],
            trainer_args=config["training"]["trainer_args"],
        )

    return model


def predict(
    config: dict, random_seed: int, model: Optional[PreTrainedModel] = None
) -> Dataset:
    """
    Computes predictions with a model given a configuration.

    Args:
        config (dict): a config dict
        random_seed (int): a random seed
        model (Optional[PreTrainedModel]): model to be used. If none,
                                    the model will be loaded from the config.

    Returns:
        Dataset: the dataset of the config with a new column `predictions`
    """
    # Fix random seed
    fix_seed(random_seed)

    # Load the dataset
    dataset = load_dataset_from_config(
        path=config["dataset"]["path"], args=config["dataset"].get("args", {})
    )

    # Select test samples
    max_n_test = config["dataset"].get("max_n_test", -1)
    if max_n_test > 0:
        log(
            _logger.info,
            f"Selecting {max_n_test} from the test set samples to evaluate.",
            "yellow",
        )
        dataset["test"] = dataset["test"].select(range(max_n_test))

    # Load the model if no `model` is passed
    if model is None:
        log(
            _logger.info,
            f"Model will be loaded from {config['model']['name']}",
            "yellow",
        )
        model = load_model(
            model_name=config["model"]["name"],
            model_args=config["model"].get("model_args", {}),
            quantization=config["model"].get("quantization", {}),
        )

    tokenizer = load_tokenizer(
        config["model"]["name"],
        chat_template_from=config["model"].get("chat_template_from", None),
    )

    # Predict according the `task_type`
    task_type = config["dataset"]["task_type"]
    if task_type == "generation":
        predictions = generate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset["test"],
            instruction=config["prompting"]["instruction"],
            input_columns=config["prompting"]["input_columns"],
            max_input_length=config["prompting"]["max_input_length"],
            batch_size=config["inference"]["batch_size"],
            max_seq_length=config["inference"]["max_seq_length"],
            icl_examples=config["inference"].get("icl_examples", 0),
            random_seed=random_seed,
            output_column=config["prompting"]["output_column"],
            max_output_length=config["prompting"]["max_output_length"],
            generation_args=config["inference"]["generation_args"],
        )
    else:
        labels = sorted(
            list(set(dataset["test"][config["prompting"]["output_column"]]))
        )
        predictions = classify(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset["test"],
            instruction=config["prompting"]["instruction"],
            input_columns=config["prompting"]["input_columns"],
            labels=labels,
            max_input_length=config["prompting"]["max_input_length"],
            batch_size=config["inference"]["batch_size"],
            max_seq_length=config["inference"]["max_seq_length"],
            icl_examples=config["inference"].get("icl_examples", 0),
            random_seed=random_seed,
            output_column=config["prompting"]["output_column"],
            max_output_length=config["prompting"]["max_output_length"],
            calibration_args=config["inference"].get("calibration_args", {}),
            train_dataset=dataset["train"] if "train" in dataset else None,
            random_icl=config["inference"]["random_icl"],
        )

    output_dataset = dataset["test"].add_column(
        "predictions", predictions["preds"]
    )
    if "probs" in predictions:
        output_dataset = output_dataset.add_column(
            "probabilities", predictions["probs"].detach().cpu().tolist()
        )
    return output_dataset


def evaluate(
    config: dict,
    random_seed: int,
    model: Optional[PreTrainedModel] = None,
) -> dict:
    """
    Evaluates a model given a configuration.

    Args:
        config (dict): a config dict
        random_seed (int): a random seed
        model (Optional[PreTrainedModel]): model to be used. If none,
                                    the model will be loaded from the config.

    Returns:
        dict: the output from the `evaluator`
    """
    from .modeling.evaluation import evaluate as _evaluate

    # Predict with the model
    output_dataset = predict(
        config=config,
        random_seed=random_seed,
        model=model,
    )

    # Evaluate
    task_type = config["dataset"]["task_type"]
    references = output_dataset[config["prompting"]["output_column"]]
    predictions = output_dataset["predictions"]
    probabilities = (
        output_dataset["probabilities"]
        if "probabilities" in output_dataset.column_names
        else None
    )
    eval_args = config.get("evaluation", {})

    return _evaluate(
        references=references,
        predictions=predictions,
        probabilities=probabilities,
        metric_name=f"symanto/{task_type}_evaluator",
        eval_args=eval_args,
    )
