from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("tune_optuna")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "examples" / "workflow_config_gcn_Alpha360.yaml"
DEFAULT_STORAGE = "sqlite:///optuna_studies/qlib_gcn_optuna.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune QlibGCN hyperparameters with Optuna.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Qlib workflow YAML to reuse for qlib_init, dataset, and base model kwargs.",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional study timeout in seconds.")
    parser.add_argument("--study-name", default="qlib-gcn-optuna", help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=DEFAULT_STORAGE,
        help="Optuna storage URI. Use 'none' for an in-memory study.",
    )
    parser.add_argument("--sampler-seed", type=int, default=42, help="TPESampler random seed.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel Optuna jobs.")
    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=None,
        help="Override model n_epochs for tuning trials. Defaults to the YAML value.",
    )
    parser.add_argument(
        "--trial-early-stop",
        type=int,
        default=None,
        help="Override model early_stop for tuning trials. Defaults to the YAML value.",
    )
    parser.add_argument("--gpu", default=None, help="Override the model GPU setting, for example 0 or -1.")
    parser.add_argument("--seed", type=int, default=None, help="Override the model seed for all trials.")
    parser.add_argument(
        "--tune-base-model",
        action="store_true",
        help="Also tune base_model between GRU and LSTM. By default the YAML value is kept.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "optuna_best_params.json",
        help="Path for the best-trial summary JSON.",
    )
    parser.add_argument(
        "--save-best-config",
        type=Path,
        default=None,
        help="Optional path for a YAML config with best model kwargs merged in.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def import_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: PyYAML. Install it with `pip install PyYAML`.") from exc
    return yaml


def import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Missing dependency: optuna. Install it with `pip install optuna`.") from exc
    return optuna


def load_config(path: Path) -> dict[str, Any]:
    yaml = import_yaml()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or "task" not in config:
        raise ValueError(f"Expected a Qlib workflow YAML with a top-level task section: {path}")
    return config


def normalize_storage_uri(storage: str | None) -> str | None:
    if storage is None:
        return None

    storage = storage.strip()
    if storage.lower() in {"", "none", "memory", "in-memory"}:
        return None

    sqlite_prefix = "sqlite:///"
    if storage.startswith(sqlite_prefix):
        sqlite_path = storage[len(sqlite_prefix) :]
        if sqlite_path and sqlite_path != ":memory:":
            path = Path(sqlite_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            return sqlite_prefix + path.as_posix()

    return storage


def prepare_qlib(config: dict[str, Any]):
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import qlib
        from qlib.utils import init_instance_by_config
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyqlib. Install it with `pip install pyqlib`.") from exc

    qlib_init = copy.deepcopy(config.get("qlib_init", {}))
    provider_uri = qlib_init.get("provider_uri")
    if isinstance(provider_uri, str):
        qlib_init["provider_uri"] = str(Path(provider_uri).expanduser())

    qlib.init(**qlib_init)
    return init_instance_by_config


def fixed_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.trial_epochs is not None:
        overrides["n_epochs"] = args.trial_epochs
    if args.trial_early_stop is not None:
        overrides["early_stop"] = args.trial_early_stop
    if args.gpu is not None:
        overrides["GPU"] = args.gpu
    if args.seed is not None:
        overrides["seed"] = args.seed
    return overrides


def suggest_model_kwargs(
    trial: Any,
    args: argparse.Namespace,
    base_kwargs: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "gcn_hidden_size": trial.suggest_categorical("gcn_hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
    }

    adj_mode = str(base_kwargs.get("adj_mode", "dynamic")).lower()
    if adj_mode == "dynamic":
        params["topk"] = trial.suggest_categorical("topk", [5, 10, 20, 30, 50])
        params["min_edge_weight"] = trial.suggest_float("min_edge_weight", 0.0, 0.3)

    if args.tune_base_model:
        params["base_model"] = trial.suggest_categorical("base_model", ["GRU", "LSTM"])

    params.update(fixed_overrides(args))
    return params


def build_model_config(task_config: dict[str, Any], model_kwargs: dict[str, Any]) -> dict[str, Any]:
    model_config = copy.deepcopy(task_config["model"])
    merged_kwargs = copy.deepcopy(model_config.get("kwargs", {}))
    merged_kwargs.update(model_kwargs)
    model_config["kwargs"] = merged_kwargs
    return model_config


def best_finite_score(scores: list[Any]) -> float | None:
    finite_scores = []
    for score in scores:
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite_scores.append(value)
    if not finite_scores:
        return None
    return max(finite_scores)


def free_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("Skipping torch cache cleanup", exc_info=True)


def build_objective(
    *,
    args: argparse.Namespace,
    task_config: dict[str, Any],
    dataset: Any,
    init_instance_by_config: Any,
    optuna_module: Any,
):
    base_kwargs = copy.deepcopy(task_config["model"].get("kwargs", {}))

    def objective(trial: Any) -> float:
        sampled_kwargs = suggest_model_kwargs(trial, args, base_kwargs)
        model_config = build_model_config(task_config, sampled_kwargs)
        evals_result: dict[str, list[float]] = {}
        model = init_instance_by_config(model_config)

        LOGGER.info("Trial %s started with params: %s", trial.number, sampled_kwargs)
        try:
            model.fit(dataset, evals_result=evals_result)
            valid_scores = evals_result.get("valid", [])
            score = best_finite_score(valid_scores)
            if score is None:
                raise optuna_module.TrialPruned("No finite validation score was produced.")

            best_epoch = getattr(model, "best_epoch", None)
            epochs_ran = len(valid_scores)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("epochs_ran", epochs_ran)
            if valid_scores:
                last_score = best_finite_score([valid_scores[-1]])
                if last_score is not None:
                    trial.set_user_attr("last_valid_score", last_score)

            report_step = int(best_epoch or epochs_ran or 0)
            trial.report(score, step=report_step)
            if trial.should_prune():
                raise optuna_module.TrialPruned()

            LOGGER.info(
                "Trial %s finished: best_valid_score=%.6f, best_epoch=%s, epochs_ran=%s",
                trial.number,
                score,
                best_epoch,
                epochs_ran,
            )
            return score
        finally:
            del model
            free_torch_cache()

    return objective


def write_best_config(config: dict[str, Any], best_model_kwargs: dict[str, Any], output_path: Path) -> None:
    yaml = import_yaml()
    tuned_config = copy.deepcopy(config)
    tuned_config["task"]["model"]["kwargs"] = best_model_kwargs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(tuned_config, file, sort_keys=False, allow_unicode=True)


def write_summary(
    *,
    args: argparse.Namespace,
    study: Any,
    config: dict[str, Any],
    storage_uri: str | None,
) -> dict[str, Any]:
    base_kwargs = copy.deepcopy(config["task"]["model"].get("kwargs", {}))
    best_model_kwargs = copy.deepcopy(base_kwargs)
    best_model_kwargs.update(study.best_trial.params)
    best_model_kwargs.update(fixed_overrides(args))

    summary = {
        "study_name": study.study_name,
        "storage": storage_uri,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "fixed_overrides": fixed_overrides(args),
        "best_model_kwargs": best_model_kwargs,
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }

    output_json = resolve_project_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
    LOGGER.info("Best-trial summary written to %s", output_json)

    if args.save_best_config is not None:
        best_config_path = resolve_project_path(args.save_best_config)
        write_best_config(config, best_model_kwargs, best_config_path)
        LOGGER.info("Best-params YAML written to %s", best_config_path)

    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.n_jobs != 1:
        LOGGER.warning("Parallel trials may contend for GPU memory and Qlib dataset caches.")

    optuna_module = import_optuna()
    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    init_instance_by_config = prepare_qlib(config)

    dataset_config = copy.deepcopy(config["task"]["dataset"])
    dataset = init_instance_by_config(dataset_config)

    storage_uri = normalize_storage_uri(args.storage)
    sampler = optuna_module.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna_module.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage_uri,
        load_if_exists=True,
    )

    objective = build_objective(
        args=args,
        task_config=config["task"],
        dataset=dataset,
        init_instance_by_config=init_instance_by_config,
        optuna_module=optuna_module,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_jobs)

    completed_trials = [trial for trial in study.trials if trial.state == optuna_module.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("No completed Optuna trials. Check the training logs and search space.")

    summary = write_summary(args=args, study=study, config=config, storage_uri=storage_uri)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
