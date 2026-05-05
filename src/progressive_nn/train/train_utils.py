"""Training utilities for continual learning models."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from ..data import get_device, set_seed
from ..metrics import AccuracyTracker, ContinualLearningMetrics, compute_task_accuracy


class ContinualLearningTrainer:
    """Trainer for continual learning models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4) -> None:
        """Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.metrics = ContinualLearningMetrics(num_tasks=10)  # Will be updated
        self.accuracy_tracker = AccuracyTracker()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def train_task(self, train_loader: DataLoader, test_loader: DataLoader,
                   task_id: int, epochs: int = 10, verbose: bool = True) -> Dict[str, float]:
        """Train model on a specific task.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            task_id: Task identifier
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Update metrics tracker
        if hasattr(self.model, 'learn_task'):
            self.model.learn_task(task_id)
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.accuracy_tracker.reset()
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if hasattr(self.model, 'forward') and 'task_id' in self.model.forward.__code__.co_varnames:
                    outputs = self.model(batch_x, task_id)
                else:
                    outputs = self.model(batch_x)
                
                loss = self.criterion(outputs, batch_y)
                
                # Add regularization losses for specific models
                if hasattr(self.model, 'ewc_loss'):
                    ewc_loss = self.model.ewc_loss(task_id)
                    loss += ewc_loss
                
                if hasattr(self.model, 'distillation_loss'):
                    dist_loss = self.model.distillation_loss(batch_x, task_id)
                    loss += dist_loss
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                self.accuracy_tracker.update(outputs, batch_y)
            
            # Compute epoch metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = self.accuracy_tracker.get_accuracy()
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if verbose and epoch % 2 == 0:
                self.logger.info(f"Task {task_id}, Epoch {epoch+1}/{epochs}, "
                               f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Evaluate on test set
        test_accuracy = compute_task_accuracy(self.model, test_loader, task_id, self.device)
        
        # Update metrics
        self.metrics.update_task_accuracy(task_id, test_accuracy)
        
        if verbose:
            self.logger.info(f"Task {task_id} completed. Test accuracy: {test_accuracy:.4f}")
        
        return {
            'task_id': task_id,
            'final_train_loss': train_losses[-1],
            'final_train_accuracy': train_accuracies[-1],
            'test_accuracy': test_accuracy,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies
        }
    
    def evaluate_all_tasks(self, task_loaders: Dict[int, Tuple[DataLoader, DataLoader]]) -> Dict[str, float]:
        """Evaluate model on all learned tasks.
        
        Args:
            task_loaders: Dictionary mapping task_id to (train_loader, test_loader)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        task_accuracies = {}
        
        for task_id, (_, test_loader) in task_loaders.items():
            accuracy = compute_task_accuracy(self.model, test_loader, task_id, self.device)
            task_accuracies[f'task_{task_id}_accuracy'] = accuracy
            
            if verbose:
                self.logger.info(f"Task {task_id} accuracy: {accuracy:.4f}")
        
        return task_accuracies
    
    def get_continual_learning_metrics(self) -> Dict[str, float]:
        """Get continual learning specific metrics."""
        return self.metrics.compute_final_metrics()


class ProgressiveNeuralNetworkTrainer(ContinualLearningTrainer):
    """Specialized trainer for Progressive Neural Networks."""
    
    def train_task(self, train_loader: DataLoader, test_loader: DataLoader,
                   task_id: int, epochs: int = 10, verbose: bool = True) -> Dict[str, float]:
        """Train Progressive Neural Network on a specific task."""
        # Freeze previous tasks
        for prev_task in range(task_id):
            if hasattr(self.model, 'freeze_task'):
                self.model.freeze_task(prev_task)
        
        # Train current task
        return super().train_task(train_loader, test_loader, task_id, epochs, verbose)


class EWCTrainer(ContinualLearningTrainer):
    """Specialized trainer for Elastic Weight Consolidation."""
    
    def train_task(self, train_loader: DataLoader, test_loader: DataLoader,
                   task_id: int, epochs: int = 10, verbose: bool = True) -> Dict[str, float]:
        """Train EWC model on a specific task."""
        # Train normally
        metrics = super().train_task(train_loader, test_loader, task_id, epochs, verbose)
        
        # Compute Fisher information matrix after training
        if hasattr(self.model, 'compute_fisher_matrix'):
            self.model.compute_fisher_matrix(train_loader, task_id)
        
        return metrics


class LearningWithoutForgettingTrainer(ContinualLearningTrainer):
    """Specialized trainer for Learning Without Forgetting."""
    
    def train_task(self, train_loader: DataLoader, test_loader: DataLoader,
                   task_id: int, epochs: int = 10, verbose: bool = True) -> Dict[str, float]:
        """Train LwF model on a specific task."""
        # Store previous model before learning new task
        if hasattr(self.model, 'learn_task'):
            self.model.learn_task(task_id)
        
        return super().train_task(train_loader, test_loader, task_id, epochs, verbose)


def create_trainer(model: nn.Module, model_type: str = "progressive",
                   **kwargs) -> ContinualLearningTrainer:
    """Create appropriate trainer for model type.
    
    Args:
        model: Model to train
        model_type: Type of model ("progressive", "ewc", "lwf", "packnet")
        **kwargs: Additional arguments for trainer
        
    Returns:
        Appropriate trainer instance
    """
    if model_type.lower() == "progressive":
        return ProgressiveNeuralNetworkTrainer(model, **kwargs)
    elif model_type.lower() == "ewc":
        return EWCTrainer(model, **kwargs)
    elif model_type.lower() == "lwf":
        return LearningWithoutForgettingTrainer(model, **kwargs)
    else:
        return ContinualLearningTrainer(model, **kwargs)


def run_continual_learning_experiment(
    model: nn.Module,
    task_loaders: Dict[int, Tuple[DataLoader, DataLoader]],
    model_type: str = "progressive",
    epochs_per_task: int = 10,
    verbose: bool = True
) -> Dict[str, Union[float, Dict]]:
    """Run a complete continual learning experiment.
    
    Args:
        model: Model to train
        task_loaders: Dictionary mapping task_id to (train_loader, test_loader)
        model_type: Type of model
        epochs_per_task: Number of epochs per task
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing experiment results
    """
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create trainer
    trainer = create_trainer(model, model_type)
    
    # Train on each task sequentially
    task_results = {}
    
    for task_id in sorted(task_loaders.keys()):
        train_loader, test_loader = task_loaders[task_id]
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Training Task {task_id}")
            print(f"{'='*50}")
        
        # Train on current task
        task_metrics = trainer.train_task(
            train_loader, test_loader, task_id, epochs_per_task, verbose
        )
        task_results[task_id] = task_metrics
        
        # Evaluate on all previous tasks
        if verbose:
            print(f"\nEvaluating on all learned tasks after Task {task_id}:")
        
        eval_results = trainer.evaluate_all_tasks(
            {tid: loaders for tid, loaders in task_loaders.items() if tid <= task_id}
        )
        
        for metric_name, value in eval_results.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Get final continual learning metrics
    cl_metrics = trainer.get_continual_learning_metrics()
    
    return {
        'task_results': task_results,
        'continual_learning_metrics': cl_metrics,
        'final_evaluation': eval_results
    }