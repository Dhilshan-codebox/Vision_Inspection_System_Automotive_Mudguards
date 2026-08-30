"""
Phase 2: Hyperparameter Tuning Script
Utilizes Optuna to systematically search for the best hyperparameters 
to minimize False Negatives and maximize macro F1 / mAP on the validation set.
"""

import os
import yaml
import logging
from typing import Dict, Any

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.training.trainer import MudguardTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hparams_config(config_path: str = 'configs/hyperparameters.yaml') -> Dict[str, Any]:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {
        "direction": "maximize",
        "num_trials": 10,
        "search_space": {
            "learning_rate": {"low": 1e-5, "high": 1e-2},
            "weight_decay": {"low": 1e-6, "high": 1e-3},
            "batch_size": {"choices": [8, 16, 32]},
        }
    }


def objective(trial, config: Dict[str, Any]) -> float:
    """
    Optuna objective function searching over learning rate, weight decay, and batch size.
    """
    search_space = config.get('search_space', {})
    
    lr = trial.suggest_float(
        "learning_rate",
        search_space.get('learning_rate', {}).get('low', 1e-4),
        search_space.get('learning_rate', {}).get('high', 1e-2),
        log=True,
    )
    wd = trial.suggest_float(
        "weight_decay",
        search_space.get('weight_decay', {}).get('low', 1e-6),
        search_space.get('weight_decay', {}).get('high', 1e-3),
        log=True,
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        search_space.get('batch_size', {}).get('choices', [8, 16]),
    )
    
    logger.info(f"Trial {trial.number}: lr={lr}, wd={wd}, bs={batch_size}")
    
    train_config = TrainingConfig(
        learning_rate=lr,
        weight_decay=wd,
        batch_size=batch_size,
        epochs=3,
    )
    trainer = MudguardTrainer(train_config)
    results = trainer.train()
    
    return float(results.get("macro_f1", 0.0))


def run_tuning():
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed. Running single baseline trainer evaluation.")
        trainer = MudguardTrainer()
        res = trainer.train()
        logger.info(f"Baseline Trainer result: {res}")
        return res

    config = load_hparams_config()
    study = optuna.create_study(direction=config.get('direction', 'maximize'))
    study.optimize(lambda trial: objective(trial, config), n_trials=config.get('num_trials', 5))
    
    logger.info(f"Best trial value: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params


if __name__ == "__main__":
    run_tuning()
