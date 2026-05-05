"""Tests for Progressive Neural Networks."""

import pytest
import torch
import torch.nn as nn

from src.progressive_nn.models.core import (
    ProgressiveNeuralNetwork,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    PackNet,
    SimpleMLP
)
from src.progressive_nn.data import DigitsContinualLearning, SyntheticContinualLearning
from src.progressive_nn.metrics import ContinualLearningMetrics, AccuracyTracker
from src.progressive_nn.utils import get_device, set_seed


class TestModels:
    """Test model implementations."""
    
    def test_progressive_neural_network(self):
        """Test Progressive Neural Network."""
        model = ProgressiveNeuralNetwork(input_size=64, hidden_size=128, output_size=5)
        
        # Test forward pass
        x = torch.randn(32, 64)
        output = model(x, task_id=0)
        assert output.shape == (32, 5)
        
        # Test adding new task
        model.learn_task(1)
        output = model(x, task_id=1)
        assert output.shape == (32, 5)
    
    def test_elastic_weight_consolidation(self):
        """Test Elastic Weight Consolidation."""
        model = ElasticWeightConsolidation(input_size=64, hidden_size=128, output_size=5)
        
        x = torch.randn(32, 64)
        output = model(x, task_id=0)
        assert output.shape == (32, 5)
    
    def test_learning_without_forgetting(self):
        """Test Learning Without Forgetting."""
        model = LearningWithoutForgetting(input_size=64, hidden_size=128, output_size=5)
        
        x = torch.randn(32, 64)
        output = model(x, task_id=0)
        assert output.shape == (32, 5)
    
    def test_packnet(self):
        """Test PackNet."""
        model = PackNet(input_size=64, hidden_size=128, output_size=5)
        
        x = torch.randn(32, 64)
        output = model(x, task_id=0)
        assert output.shape == (32, 5)
    
    def test_simple_mlp(self):
        """Test Simple MLP."""
        model = SimpleMLP(input_size=64, hidden_size=128, output_size=5)
        
        x = torch.randn(32, 64)
        output = model(x)
        assert output.shape == (32, 5)


class TestData:
    """Test data loading utilities."""
    
    def test_digits_continual_learning(self):
        """Test Digits continual learning dataset."""
        data = DigitsContinualLearning(num_tasks=3, random_state=42)
        
        # Test task creation
        assert len(data.tasks) == 3
        
        # Test dataloader creation
        train_loader, test_loader = data.get_task_dataloaders(0, batch_size=32)
        
        # Test batch
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[1] == 64  # Input features
        assert batch_y.shape[0] == batch_x.shape[0]  # Batch size consistency
    
    def test_synthetic_continual_learning(self):
        """Test Synthetic continual learning dataset."""
        data = SyntheticContinualLearning(
            num_tasks=2, 
            samples_per_task=100,
            input_dim=20,
            num_classes_per_task=2,
            random_state=42
        )
        
        # Test task creation
        assert len(data.tasks) == 2
        
        # Test dataloader creation
        train_loader, test_loader = data.get_task_dataloaders(0, batch_size=16)
        
        # Test batch
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[1] == 20  # Input features
        assert batch_y.shape[0] == batch_x.shape[0]  # Batch size consistency


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_continual_learning_metrics(self):
        """Test continual learning metrics."""
        metrics = ContinualLearningMetrics(num_tasks=3)
        
        # Test metric updates
        metrics.update_task_accuracy(0, 0.8)
        metrics.update_task_accuracy(1, 0.7)
        metrics.update_task_accuracy(2, 0.9)
        
        # Test final metrics computation
        final_metrics = metrics.compute_final_metrics()
        assert 'average_accuracy' in final_metrics
    
    def test_accuracy_tracker(self):
        """Test accuracy tracker."""
        tracker = AccuracyTracker()
        
        # Test updates
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 2, 1, 1])
        
        tracker.update(predictions, targets)
        accuracy = tracker.get_accuracy()
        
        # Should be 4/5 = 0.8
        assert abs(accuracy - 0.8) < 1e-6


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't crash
        assert True


@pytest.mark.slow
class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training."""
        # Create small dataset
        data = SyntheticContinualLearning(
            num_tasks=2,
            samples_per_task=50,
            input_dim=10,
            num_classes_per_task=2,
            random_state=42
        )
        
        # Create model
        model = ProgressiveNeuralNetwork(input_size=10, hidden_size=32, output_size=2)
        
        # Create task loaders
        task_loaders = {}
        for task_id in range(2):
            train_loader, test_loader = data.get_task_dataloaders(task_id, batch_size=16)
            task_loaders[task_id] = (train_loader, test_loader)
        
        # Train on first task
        from src.progressive_nn.train import ContinualLearningTrainer
        trainer = ContinualLearningTrainer(model)
        
        train_loader, test_loader = task_loaders[0]
        metrics = trainer.train_task(train_loader, test_loader, task_id=0, epochs=2, verbose=False)
        
        assert 'test_accuracy' in metrics
        assert metrics['test_accuracy'] >= 0.0
