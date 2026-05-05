"""Core models for Progressive Neural Networks and continual learning baselines."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseContinualLearner(ABC):
    """Abstract base class for continual learning models."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize the continual learner.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension for current task
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.task_id = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
    
    @abstractmethod
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass for given task."""
        pass
    
    @abstractmethod
    def learn_task(self, task_id: int) -> None:
        """Learn a new task."""
        pass
    
    @abstractmethod
    def get_task_params(self, task_id: int) -> List[nn.Parameter]:
        """Get parameters for specific task."""
        pass


class ProgressiveNeuralNetwork(BaseContinualLearner):
    """Progressive Neural Network implementation.
    
    Adds new columns for each task while keeping previous columns frozen.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize Progressive Neural Network.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension  
            output_size: Output dimension for current task
        """
        super().__init__(input_size, hidden_size, output_size)
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific columns
        self.columns: Dict[int, nn.Module] = {}
        self.lateral_connections: Dict[int, nn.Module] = {}
        
        # Initialize first task
        self._add_task_column(0)
        
    def _add_task_column(self, task_id: int) -> None:
        """Add a new column for the given task."""
        self.columns[task_id] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
        # Add lateral connections from previous tasks
        if task_id > 0:
            lateral_input_size = self.hidden_size * task_id
            self.lateral_connections[task_id] = nn.Linear(
                lateral_input_size, self.hidden_size
            )
    
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass for given task.
        
        Args:
            x: Input tensor
            task_id: Task identifier
            
        Returns:
            Output tensor for the task
        """
        if task_id not in self.columns:
            raise ValueError(f"Task {task_id} not learned yet")
        
        # Forward through backbone
        backbone_out = self.backbone(x)
        
        # Collect lateral connections from previous tasks
        lateral_inputs = [backbone_out]
        for prev_task in range(task_id):
            if prev_task in self.columns:
                prev_output = self.columns[prev_task][:-1](backbone_out)  # Exclude final layer
                lateral_inputs.append(prev_output)
        
        # Concatenate lateral connections
        if len(lateral_inputs) > 1:
            lateral_concat = torch.cat(lateral_inputs, dim=1)
            lateral_out = self.lateral_connections[task_id](lateral_concat)
            combined_input = backbone_out + lateral_out
        else:
            combined_input = backbone_out
        
        # Forward through task-specific column
        output = self.columns[task_id](combined_input)
        return output
    
    def learn_task(self, task_id: int) -> None:
        """Learn a new task by adding a new column."""
        if task_id not in self.columns:
            self._add_task_column(task_id)
        self.task_id = task_id
    
    def get_task_params(self, task_id: int) -> List[nn.Parameter]:
        """Get parameters for specific task."""
        if task_id not in self.columns:
            return []
        
        params = list(self.columns[task_id].parameters())
        if task_id in self.lateral_connections:
            params.extend(list(self.lateral_connections[task_id].parameters()))
        return params
    
    def freeze_task(self, task_id: int) -> None:
        """Freeze parameters for a specific task."""
        for param in self.get_task_params(task_id):
            param.requires_grad = False


class ElasticWeightConsolidation(BaseContinualLearner):
    """Elastic Weight Consolidation (EWC) for continual learning.
    
    Prevents catastrophic forgetting by penalizing changes to important weights.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 ewc_lambda: float = 1000.0) -> None:
        """Initialize EWC model.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension for current task
            ewc_lambda: EWC regularization strength
        """
        super().__init__(input_size, hidden_size, output_size)
        
        self.ewc_lambda = ewc_lambda
        self.fisher_matrices: Dict[int, Dict[str, Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, Tensor]] = {}
        
        # Single shared network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass."""
        return self.network(x)
    
    def learn_task(self, task_id: int) -> None:
        """Learn a new task."""
        self.task_id = task_id
    
    def get_task_params(self, task_id: int) -> List[nn.Parameter]:
        """Get all parameters (EWC uses shared network)."""
        return list(self.network.parameters())
    
    def compute_fisher_matrix(self, dataloader, task_id: int) -> None:
        """Compute Fisher information matrix for current task."""
        self.network.eval()
        fisher_matrix = {}
        
        for name, param in self.network.named_parameters():
            fisher_matrix[name] = torch.zeros_like(param)
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.network.zero_grad()
            output = self.network(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad.data ** 2
        
        # Average over batches
        for name in fisher_matrix:
            fisher_matrix[name] /= len(dataloader)
        
        self.fisher_matrices[task_id] = fisher_matrix
        
        # Store optimal parameters
        self.optimal_params[task_id] = {
            name: param.data.clone() for name, param in self.network.named_parameters()
        }
    
    def ewc_loss(self, task_id: int) -> Tensor:
        """Compute EWC regularization loss."""
        if task_id not in self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = torch.tensor(0.0, device=self.device)
        fisher_matrix = self.fisher_matrices[task_id]
        optimal_params = self.optimal_params[task_id]
        
        for name, param in self.network.named_parameters():
            if name in fisher_matrix and name in optimal_params:
                ewc_loss += (fisher_matrix[name] * (param - optimal_params[name]) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss


class LearningWithoutForgetting(BaseContinualLearner):
    """Learning Without Forgetting (LwF) for continual learning.
    
    Uses knowledge distillation to retain previous task knowledge.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 temperature: float = 3.0) -> None:
        """Initialize LwF model.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension for current task
            temperature: Temperature for knowledge distillation
        """
        super().__init__(input_size, hidden_size, output_size)
        
        self.temperature = temperature
        self.previous_models: Dict[int, nn.Module] = {}
        
        # Single shared network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass."""
        return self.network(x)
    
    def learn_task(self, task_id: int) -> None:
        """Learn a new task."""
        # Store previous model
        if task_id > 0:
            self.previous_models[task_id - 1] = self._copy_model()
        self.task_id = task_id
    
    def _copy_model(self) -> nn.Module:
        """Create a copy of current model."""
        model_copy = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
        # Copy parameters
        for param_src, param_dst in zip(self.network.parameters(), model_copy.parameters()):
            param_dst.data.copy_(param_src.data)
        
        return model_copy
    
    def get_task_params(self, task_id: int) -> List[nn.Parameter]:
        """Get all parameters (LwF uses shared network)."""
        return list(self.network.parameters())
    
    def distillation_loss(self, x: Tensor, task_id: int) -> Tensor:
        """Compute knowledge distillation loss."""
        if task_id == 0 or task_id - 1 not in self.previous_models:
            return torch.tensor(0.0, device=self.device)
        
        previous_model = self.previous_models[task_id - 1]
        previous_model.eval()
        
        with torch.no_grad():
            teacher_output = previous_model(x)
            teacher_probs = F.softmax(teacher_output / self.temperature, dim=1)
        
        student_output = self.network(x)
        student_log_probs = F.log_softmax(student_output / self.temperature, dim=1)
        
        distillation_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return distillation_loss * (self.temperature ** 2)


class PackNet(BaseContinualLearner):
    """PackNet: Learning to Pack Neural Networks for Continual Learning.
    
    Prunes and packs networks to make room for new tasks.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 prune_ratio: float = 0.5) -> None:
        """Initialize PackNet model.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension for current task
            prune_ratio: Fraction of weights to prune
        """
        super().__init__(input_size, hidden_size, output_size)
        
        self.prune_ratio = prune_ratio
        self.masks: Dict[int, Dict[str, Tensor]] = {}
        
        # Single shared network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass with task-specific masks."""
        if task_id in self.masks:
            self._apply_mask(task_id)
        
        output = self.network(x)
        
        if task_id in self.masks:
            self._remove_mask()
        
        return output
    
    def learn_task(self, task_id: int) -> None:
        """Learn a new task."""
        self.task_id = task_id
    
    def get_task_params(self, task_id: int) -> List[nn.Parameter]:
        """Get all parameters (PackNet uses shared network)."""
        return list(self.network.parameters())
    
    def prune_for_task(self, task_id: int) -> None:
        """Prune network to make room for new task."""
        mask = {}
        
        for name, param in self.network.named_parameters():
            if len(param.shape) > 1:  # Only prune weight matrices, not biases
                # Get smallest weights to prune
                flat_param = param.data.flatten()
                threshold = torch.quantile(torch.abs(flat_param), self.prune_ratio)
                
                # Create mask
                mask[name] = (torch.abs(param.data) > threshold).float()
        
        self.masks[task_id] = mask
    
    def _apply_mask(self, task_id: int) -> None:
        """Apply task-specific mask."""
        if task_id not in self.masks:
            return
        
        mask = self.masks[task_id]
        for name, param in self.network.named_parameters():
            if name in mask:
                param.data *= mask[name]
    
    def _remove_mask(self) -> None:
        """Remove current mask."""
        # This is a simplified implementation
        # In practice, you'd need to restore original parameters
        pass


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron baseline."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initialize MLP.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.network(x)
