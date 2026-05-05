"""Data loading utilities for continual learning experiments."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


class ContinualLearningDataset(Dataset):
    """Dataset for continual learning scenarios."""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, 
                 task_id: int, transform: Optional[callable] = None) -> None:
        """Initialize dataset.
        
        Args:
            data: Input data
            targets: Target labels
            task_id: Task identifier
            transform: Optional data transformation
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.task_id = task_id
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class DigitsContinualLearning:
    """Continual learning scenario using digits dataset."""
    
    def __init__(self, num_tasks: int = 3, test_size: float = 0.2, 
                 random_state: int = 42) -> None:
        """Initialize digits continual learning scenario.
        
        Args:
            num_tasks: Number of tasks to create
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.num_tasks = num_tasks
        self.test_size = test_size
        self.random_state = random_state
        
        # Load digits dataset
        digits = load_digits()
        self.X = digits.data / 16.0  # Normalize to [0, 1]
        self.y = digits.target
        
        # Split into tasks
        self._create_tasks()
    
    def _create_tasks(self) -> None:
        """Create tasks by splitting classes."""
        self.tasks: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Split classes into tasks
        classes_per_task = 10 // self.num_tasks
        remaining_classes = 10 % self.num_tasks
        
        start_class = 0
        for task_id in range(self.num_tasks):
            # Determine number of classes for this task
            if task_id < remaining_classes:
                num_classes = classes_per_task + 1
            else:
                num_classes = classes_per_task
            
            end_class = start_class + num_classes
            
            # Get data for this task
            task_mask = (self.y >= start_class) & (self.y < end_class)
            task_X = self.X[task_mask]
            task_y = self.y[task_mask] - start_class  # Relabel to start from 0
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                task_X, task_y, test_size=self.test_size, 
                random_state=self.random_state, stratify=task_y
            )
            
            self.tasks[task_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'num_classes': num_classes
            }
            
            start_class = end_class
    
    def get_task_dataloaders(self, task_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders for a task.
        
        Args:
            task_id: Task identifier
            batch_size: Batch size for dataloaders
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_data = self.tasks[task_id]
        
        # Create datasets
        train_dataset = ContinualLearningDataset(
            task_data['X_train'], task_data['y_train'], task_id
        )
        test_dataset = ContinualLearningDataset(
            task_data['X_test'], task_data['y_test'], task_id
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def get_task_info(self, task_id: int) -> Dict[str, Union[int, np.ndarray]]:
        """Get information about a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Dictionary with task information
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        return self.tasks[task_id].copy()


class SyntheticContinualLearning:
    """Synthetic continual learning scenario."""
    
    def __init__(self, num_tasks: int = 3, samples_per_task: int = 1000,
                 input_dim: int = 20, num_classes_per_task: int = 2,
                 test_size: float = 0.2, random_state: int = 42) -> None:
        """Initialize synthetic continual learning scenario.
        
        Args:
            num_tasks: Number of tasks to create
            samples_per_task: Number of samples per task
            input_dim: Input feature dimension
            num_classes_per_task: Number of classes per task
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task
        self.input_dim = input_dim
        self.num_classes_per_task = num_classes_per_task
        self.test_size = test_size
        self.random_state = random_state
        
        self.tasks: Dict[int, Dict[str, np.ndarray]] = {}
        self._create_tasks()
    
    def _create_tasks(self) -> None:
        """Create synthetic tasks."""
        np.random.seed(self.random_state)
        
        for task_id in range(self.num_tasks):
            # Generate synthetic data for this task
            X, y = make_classification(
                n_samples=self.samples_per_task,
                n_features=self.input_dim,
                n_classes=self.num_classes_per_task,
                n_redundant=0,
                n_informative=self.input_dim,
                n_clusters_per_class=1,
                random_state=self.random_state + task_id
            )
            
            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size,
                random_state=self.random_state, stratify=y
            )
            
            self.tasks[task_id] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'num_classes': self.num_classes_per_task
            }
    
    def get_task_dataloaders(self, task_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Get train and test dataloaders for a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_data = self.tasks[task_id]
        
        # Create datasets
        train_dataset = ContinualLearningDataset(
            task_data['X_train'], task_data['y_train'], task_id
        )
        test_dataset = ContinualLearningDataset(
            task_data['X_test'], task_data['y_test'], task_id
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def get_task_info(self, task_id: int) -> Dict[str, Union[int, np.ndarray]]:
        """Get information about a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        return self.tasks[task_id].copy()


def create_permuted_mnist_tasks(num_tasks: int = 5, random_state: int = 42) -> Dict[int, Dict[str, np.ndarray]]:
    """Create permuted MNIST tasks (simplified version using digits).
    
    Args:
        num_tasks: Number of tasks to create
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing task data
    """
    np.random.seed(random_state)
    
    # Load digits as base dataset
    digits = load_digits()
    X_base = digits.data.astype(np.float32)
    y_base = digits.target
    
    tasks = {}
    
    for task_id in range(num_tasks):
        # Create random permutation for this task
        permutation = np.random.permutation(X_base.shape[1])
        X_permuted = X_base[:, permutation]
        
        # Normalize
        X_permuted = X_permuted / 16.0
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_permuted, y_base, test_size=0.2,
            random_state=random_state + task_id, stratify=y_base
        )
        
        tasks[task_id] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'num_classes': 10,
            'permutation': permutation
        }
    
    return tasks


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        PyTorch device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False