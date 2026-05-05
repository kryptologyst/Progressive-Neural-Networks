"""Evaluation metrics for continual learning."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class ContinualLearningMetrics:
    """Metrics for evaluating continual learning performance."""
    
    def __init__(self, num_tasks: int) -> None:
        """Initialize metrics tracker.
        
        Args:
            num_tasks: Number of tasks in the continual learning scenario
        """
        self.num_tasks = num_tasks
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        # Task-specific accuracies: [task_id][epoch] -> accuracy
        self.task_accuracies: Dict[int, List[float]] = {i: [] for i in range(self.num_tasks)}
        
        # Final accuracies after all tasks learned
        self.final_accuracies: Dict[int, float] = {}
        
        # Backward transfer: how much learning new tasks affects old tasks
        self.backward_transfer: Dict[int, float] = {}
        
        # Forward transfer: how much learning old tasks helps new tasks
        self.forward_transfer: Dict[int, float] = {}
        
        # Forgetting: how much performance degrades on old tasks
        self.forgetting: Dict[int, float] = {}
    
    def update_task_accuracy(self, task_id: int, accuracy: float) -> None:
        """Update accuracy for a specific task.
        
        Args:
            task_id: Task identifier
            accuracy: Accuracy value (0-1)
        """
        self.task_accuracies[task_id].append(accuracy)
    
    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute final continual learning metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Compute final accuracies
        for task_id in range(self.num_tasks):
            if self.task_accuracies[task_id]:
                self.final_accuracies[task_id] = self.task_accuracies[task_id][-1]
        
        # Compute backward transfer
        metrics.update(self._compute_backward_transfer())
        
        # Compute forward transfer  
        metrics.update(self._compute_forward_transfer())
        
        # Compute forgetting
        metrics.update(self._compute_forgetting())
        
        # Compute average metrics
        metrics.update(self._compute_average_metrics())
        
        return metrics
    
    def _compute_backward_transfer(self) -> Dict[str, float]:
        """Compute backward transfer for each task."""
        backward_transfer = {}
        
        for task_id in range(1, self.num_tasks):
            if task_id in self.final_accuracies and task_id - 1 in self.final_accuracies:
                # Backward transfer: how much learning task i affects task i-1
                if len(self.task_accuracies[task_id - 1]) >= 2:
                    initial_acc = self.task_accuracies[task_id - 1][0]
                    final_acc = self.task_accuracies[task_id - 1][-1]
                    backward_transfer[f"backward_transfer_task_{task_id}"] = final_acc - initial_acc
        
        return backward_transfer
    
    def _compute_forward_transfer(self) -> Dict[str, float]:
        """Compute forward transfer for each task."""
        forward_transfer = {}
        
        for task_id in range(1, self.num_tasks):
            if task_id in self.final_accuracies:
                # Forward transfer: how much learning previous tasks helps current task
                # This is approximated by comparing with random initialization
                # In practice, you'd compare with a model trained only on current task
                forward_transfer[f"forward_transfer_task_{task_id}"] = self.final_accuracies[task_id]
        
        return forward_transfer
    
    def _compute_forgetting(self) -> Dict[str, float]:
        """Compute forgetting for each task."""
        forgetting = {}
        
        for task_id in range(self.num_tasks - 1):
            if task_id in self.final_accuracies:
                # Forgetting: how much performance degrades on old tasks
                if len(self.task_accuracies[task_id]) >= 2:
                    peak_acc = max(self.task_accuracies[task_id])
                    final_acc = self.task_accuracies[task_id][-1]
                    forgetting[f"forgetting_task_{task_id}"] = peak_acc - final_acc
        
        return forgetting
    
    def _compute_average_metrics(self) -> Dict[str, float]:
        """Compute average metrics across all tasks."""
        metrics = {}
        
        if self.final_accuracies:
            metrics["average_accuracy"] = np.mean(list(self.final_accuracies.values()))
            metrics["average_backward_transfer"] = np.mean([
                v for k, v in self.__dict__.items() 
                if k.startswith("backward_transfer_task_")
            ])
            metrics["average_forward_transfer"] = np.mean([
                v for k, v in self.__dict__.items() 
                if k.startswith("forward_transfer_task_")
            ])
            metrics["average_forgetting"] = np.mean([
                v for k, v in self.__dict__.items() 
                if k.startswith("forgetting_task_")
            ])
        
        return metrics


class AccuracyTracker:
    """Track accuracy over time for continual learning."""
    
    def __init__(self) -> None:
        """Initialize accuracy tracker."""
        self.reset()
    
    def reset(self) -> None:
        """Reset tracker."""
        self.correct = 0
        self.total = 0
        self.predictions: List[int] = []
        self.targets: List[int] = []
    
    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Update tracker with new predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        """
        if predictions.dim() > 1:
            predictions = torch.argmax(predictions, dim=1)
        
        self.correct += (predictions == targets).sum().item()
        self.total += targets.size(0)
        
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.targets, self.predictions)


class LearningCurveTracker:
    """Track learning curves for continual learning."""
    
    def __init__(self) -> None:
        """Initialize learning curve tracker."""
        self.reset()
    
    def reset(self) -> None:
        """Reset tracker."""
        self.losses: List[float] = []
        self.accuracies: List[float] = []
        self.epochs: List[int] = []
    
    def update(self, epoch: int, loss: float, accuracy: float) -> None:
        """Update with new epoch data.
        
        Args:
            epoch: Current epoch
            loss: Training loss
            accuracy: Training accuracy
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def get_learning_curve(self) -> Dict[str, List]:
        """Get learning curve data."""
        return {
            "epochs": self.epochs,
            "losses": self.losses,
            "accuracies": self.accuracies
        }


def compute_task_accuracy(model, dataloader, task_id: int, device: torch.device) -> float:
    """Compute accuracy for a specific task.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader for the task
        task_id: Task identifier
        device: Device to run on
        
    Returns:
        Accuracy value (0-1)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if hasattr(model, 'forward') and 'task_id' in model.forward.__code__.co_varnames:
                outputs = model(batch_x, task_id)
            else:
                outputs = model(batch_x)
            
            if outputs.dim() > 1:
                predicted = torch.argmax(outputs, dim=1)
            else:
                predicted = (outputs > 0.5).long()
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return correct / total if total > 0 else 0.0


def compute_forgetting_measure(accuracies: List[float]) -> float:
    """Compute forgetting measure from accuracy history.
    
    Args:
        accuracies: List of accuracies over time
        
    Returns:
        Forgetting measure (higher = more forgetting)
    """
    if len(accuracies) < 2:
        return 0.0
    
    peak_accuracy = max(accuracies)
    final_accuracy = accuracies[-1]
    
    return peak_accuracy - final_accuracy


def compute_backward_transfer(accuracies_before: List[float], 
                           accuracies_after: List[float]) -> float:
    """Compute backward transfer measure.
    
    Args:
        accuracies_before: Accuracies before learning new task
        accuracies_after: Accuracies after learning new task
        
    Returns:
        Backward transfer measure (positive = positive transfer)
    """
    if not accuracies_before or not accuracies_after:
        return 0.0
    
    avg_before = np.mean(accuracies_before)
    avg_after = np.mean(accuracies_after)
    
    return avg_after - avg_before