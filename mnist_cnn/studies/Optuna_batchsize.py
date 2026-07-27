from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import optuna
from omegaconf import DictConfig, OmegaConf


ROOT_DIR = Path(__file__).resolve().parents[1]
STUDIES_DIR = ROOT_DIR / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from MNist_NN import load_config, train  # noqa: E402


def resolve_device(cfg: DictConfig):
    import torch

    device_name = os.environ.get("MNIST_DEVICE", cfg.device)
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Training Using CPU")

    print(f"Requested device: {device_name}")

    return device


def log_config_to_trial(trial: optuna.Trial, config: dict, prefix: str = "config"):
    trial.set_user_attr(prefix, json.dumps(config, sort_keys=True))

    for key, value in config.items():
        attr_name = f"{prefix}.{key}"
        if isinstance(value, dict):
            log_config_to_trial(trial, value, attr_name)
        elif isinstance(value, list):
            trial.set_user_attr(attr_name, json.dumps(value))
        else:
            trial.set_user_attr(attr_name, value)


def build_trial_config(base_cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    trial_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    trial_cfg.batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    trial_cfg.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    trial_cfg.hidden_layer_1 = trial.suggest_int("hidden_layer_1", 128, 512, step=64)
    trial_cfg.hidden_layer_2 = trial.suggest_int("hidden_layer_2", 32, trial_cfg.hidden_layer_1, step=32)
    trial_cfg.dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    trial_cfg.max_epochs = 40
    trial_cfg.use_wandb = False
    trial_cfg.use_optuna = True
    return trial_cfg


def print_trial_config(trial: optuna.Trial, trial_cfg: DictConfig):
    print("\nChosen values for this Optuna trial:")
    print(f"  trial_number: {trial.number}")
    print(f"  batch_size: {trial_cfg.batch_size}")
    print(f"  learning_rate: {trial_cfg.learning_rate}")
    print(f"  hidden_layer_1: {trial_cfg.hidden_layer_1}")
    print(f"  hidden_layer_2: {trial_cfg.hidden_layer_2}")
    print(f"  dropout_rate: {trial_cfg.dropout_rate}")
    print(f"  max_epochs: {trial_cfg.max_epochs}")
    print(f"  device: {trial_cfg.device}")


def objective(trial: optuna.Trial, base_cfg: DictConfig, device):
    trial_cfg = build_trial_config(base_cfg, trial)
    full_config = OmegaConf.to_container(trial_cfg, resolve=True)
    log_config_to_trial(trial, full_config)
    print_trial_config(trial, trial_cfg)

    accuracy, epoch, model, avg_last_10_epoch_accuracy = train(trial_cfg, device)
    trial.set_user_attr("final_accuracy", accuracy)
    trial.set_user_attr("final_epoch", epoch)
    trial.set_user_attr("avg_last_10_epoch_accuracy", avg_last_10_epoch_accuracy)
    return avg_last_10_epoch_accuracy


def run_study(cfg: DictConfig | None = None):
    if cfg is None:
        cfg = load_config()

    import torch

    device = resolve_device(cfg)
    sampler = optuna.samplers.TPESampler(
        seed=None,
        multivariate=True,
        constant_liar=True,
    )
    storage_url = f"sqlite:///{(STUDIES_DIR / 'Optuna_batchsize.db').as_posix()}"
    study = optuna.create_study(
        study_name="Optuna_batchsize",
        direction="maximize",
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True,
    )

    # Optionally mark stale RUNNING trials as FAIL so they stop biasing new suggestions.
    stale_minutes = int(os.environ.get("OPTUNA_STALE_MINUTES", "0"))
    if stale_minutes > 0:
        now = datetime.now(timezone.utc)
        for trial in study.get_trials(deepcopy=False):
            if trial.state != optuna.trial.TrialState.RUNNING:
                continue
            if trial.datetime_start is None:
                continue
            if (now - trial.datetime_start) > timedelta(minutes=stale_minutes):
                try:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                    print(
                        f"Marked stale RUNNING trial {trial.number} as FAIL "
                        f"(older than {stale_minutes} minutes)."
                    )
                except Exception as exc:
                    print(f"Could not mark trial {trial.number} as FAIL: {exc}")

    study.set_user_attr("base_config", json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True))
    study.optimize(lambda trial: objective(trial, cfg, device), n_trials=cfg.optuna_trials)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    return study


if __name__ == "__main__":
    run_study()