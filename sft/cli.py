import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import typer

from sft.modeling.evaluation import aggregate_results
from sft.utils import get_logger, get_results_path, load_configs, log, save_json

app = typer.Typer()
app = typer.Typer(pretty_exceptions_enable=False)

_logger = get_logger(__name__)


@app.command()
def train(
    config_path: Path,
    random_seed: int = 13,
) -> None:
    """
    Train a model given a configuration.
    The model checkpoint will be saved in the `output_dir`
    specified in the config, passed to the `SFTrainer`.
    Args:
        config (dict): config dict
        random_seed (int): a random seed
    """
    from .cli_utils import train as _train

    # Load the config
    config = load_configs(config_path)[0]

    # Train the model
    _train(config, random_seed)


@app.command()
def predict(
    config_path: Path,
    random_seed: int = 13,
    results_path: Path = Path("predictions"),
) -> None:
    """
    Computes predictions with a model given a configuration.

    Args:
        config (dict): config dict
        random_seed (int): a random seed
    """
    from .cli_utils import predict as _predict

    # Load the config
    config = load_configs(config_path)[0]

    # Predict with the model
    output_dataset = _predict(config, random_seed)

    # Save predictions
    output_path = get_results_path(results_path, extension=".tsv")
    output_dataset.to_csv(output_path, sep="\t")
    log(_logger.info, f"Predictions have been stored in {output_path}", "green")


@app.command()
def evaluate(
    config_path: Path,
    random_seed: int = 13,
    results_path: Path = Path("results"),
) -> None:
    """
    Evaluates a model given a configuration, using
    our internal evaluation code.

    Args:
        config (dict): config dict
        random_seed (int): a random seed
    """
    from sft.cli_utils import evaluate as _evaluate

    # Load the config
    config = load_configs(config_path)[0]

    # Evaluate the model
    results = _evaluate(config, random_seed)

    # Store results
    output = {
        "seed": random_seed,
        "results": results,
        "config": config,
    }
    output_path = get_results_path(results_path)
    save_json(output_path, output)
    log(_logger.info, f"Results have been stored in {output_path}", "green")


@app.command()
def run_experiment(
    config_path: Path,
    do_eval: bool = True,
    random_seed: int = 13,
    results_path: Path = Path("results"),
):
    """
    Run a single experiment (train + eval if specified)

    Args:
        config_path (Path): path to the config of one experiment.
        do_eval (bool): whether to evaluate the model on the test split.
        random_seed (int): random seed
        results_path (Path): folder where to save the results
    """
    from .cli_utils import evaluate as _evaluate
    from .cli_utils import train as _train

    # Load the config
    config = load_configs(config_path)[0]

    # Train the model
    model = _train(config, random_seed)

    # Evaluate the model and store results
    if do_eval:
        results = _evaluate(config, random_seed, model)
        output = {
            "seed": random_seed,
            "results": results,
            "config": config,
        }
        output_path = get_results_path(results_path)
        save_json(output_path, output)
        log(_logger.info, f"Results have been stored in {output_path}", "green")


@app.command()
def run_experiments(
    config_path: Path,
    do_eval: bool = True,
    random_seed: int = 13,
    results_path: Path = Path("results"),
):
    """
    Run single/multiple experiments using subprocess,
    ensuring the GPU memory is free after the process.

    Args:
        config (dict): config of the experiment.
                       If json file -> one experiment
                       If jsonnet file -> multiple experiments
        do_eval (bool): whether to evaluate the model on the test split.
        random_seed (int): random seed
        results_path (Path): folder where to save the results
    """
    # Load config
    configs = load_configs(config_path)

    # Run experiments using subprocess to ensure
    # GPU memory is free after each experiment
    for config in configs:
        with NamedTemporaryFile(suffix=".json") as fw:
            save_json(fw.name, config)
            try:
                flag_do_eval = "--do-eval" if do_eval else "--no-do-eval"
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "sft.cli",
                        "run-experiment",
                        fw.name,
                        flag_do_eval,
                        "--random-seed",
                        str(random_seed),
                        "--results-path",
                        results_path,
                    ]
                )
            except Exception as e:
                log(
                    _logger.error,
                    f"There was an error with this config: {config}: {e}",
                    "red",
                )

    # Aggregate results
    aggregate_results(results_path)


if __name__ == "__main__":
    app()
