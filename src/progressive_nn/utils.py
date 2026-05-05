#!/usr/bin/env python3
"""Main experiment script for Progressive Neural Networks."""

import logging
import os
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.progressive_nn.data import DigitsContinualLearning, SyntheticContinualLearning
from src.progressive_nn.models.core import (
    ProgressiveNeuralNetwork,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    PackNet,
    SimpleMLP
)
from src.progressive_nn.train import run_continual_learning_experiment
from src.progressive_nn.utils import get_device, set_seed


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )


def create_model(cfg: DictConfig) -> torch.nn.Module:
    """Create model based on configuration."""
    model_type = cfg.model._target_.split('.')[-1]
    
    if model_type == "ProgressiveNeuralNetwork":
        return ProgressiveNeuralNetwork(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.model.output_size
        )
    elif model_type == "ElasticWeightConsolidation":
        return ElasticWeightConsolidation(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.model.output_size,
            ewc_lambda=cfg.model.get('ewc_lambda', 1000.0)
        )
    elif model_type == "LearningWithoutForgetting":
        return LearningWithoutForgetting(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.model.output_size,
            temperature=cfg.model.get('temperature', 3.0)
        )
    elif model_type == "PackNet":
        return PackNet(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            output_size=cfg.model.output_size,
            prune_ratio=cfg.model.get('prune_ratio', 0.5)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_data(cfg: DictConfig) -> Any:
    """Create data based on configuration."""
    data_type = cfg.data._target_.split('.')[-1]
    
    if data_type == "DigitsContinualLearning":
        return DigitsContinualLearning(
            num_tasks=cfg.data.num_tasks,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state
        )
    elif data_type == "SyntheticContinualLearning":
        return SyntheticContinualLearning(
            num_tasks=cfg.data.num_tasks,
            samples_per_task=cfg.data.samples_per_task,
            input_dim=cfg.data.input_dim,
            num_classes_per_task=cfg.data.num_classes_per_task,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main experiment function."""
    # Setup
    set_seed(cfg.experiment.seed)
    setup_logging(cfg.logging.level, cfg.logging.log_dir)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Progressive Neural Networks experiment")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create data
    logger.info("Creating dataset...")
    data = create_data(cfg)
    
    # Create task loaders
    task_loaders = {}
    for task_id in range(cfg.data.num_tasks):
        train_loader, test_loader = data.get_task_dataloaders(
            task_id, batch_size=cfg.training.batch_size
        )
        task_loaders[task_id] = (train_loader, test_loader)
        logger.info(f"Task {task_id}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(cfg)
    logger.info(f"Model created: {type(model).__name__}")
    
    # Determine model type for trainer
    model_type = cfg.model._target_.split('.')[-1].lower()
    if "progressive" in model_type:
        model_type = "progressive"
    elif "elastic" in model_type:
        model_type = "ewc"
    elif "learning" in model_type:
        model_type = "lwf"
    elif "pack" in model_type:
        model_type = "packnet"
    else:
        model_type = "default"
    
    # Run experiment
    logger.info("Starting continual learning experiment...")
    results = run_continual_learning_experiment(
        model=model,
        task_loaders=task_loaders,
        model_type=model_type,
        epochs_per_task=cfg.training.epochs_per_task,
        verbose=True
    )
    
    # Log results
    logger.info("Experiment completed!")
    logger.info("Final Results:")
    for metric, value in results['continual_learning_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    os.makedirs(cfg.output.save_dir, exist_ok=True)
    results_path = os.path.join(cfg.output.save_dir, f"{cfg.experiment.name}_results.yaml")
    OmegaConf.save(results, results_path)
    logger.info(f"Results saved to {results_path}")
    
    # Save model
    model_path = os.path.join(cfg.output.save_dir, f"{cfg.experiment.name}_model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
