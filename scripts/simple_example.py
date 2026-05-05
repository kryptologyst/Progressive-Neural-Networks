#!/usr/bin/env python3
"""Simple example script for Progressive Neural Networks."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from progressive_nn.data import DigitsContinualLearning, get_device, set_seed
from progressive_nn.models.core import (
    ProgressiveNeuralNetwork,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    SimpleMLP
)
from progressive_nn.train import run_continual_learning_experiment


def main():
    """Run a simple continual learning experiment."""
    
    print("🧠 Progressive Neural Networks - Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n📊 Creating Digits dataset...")
    data = DigitsContinualLearning(num_tasks=3, random_state=42)
    
    # Create task loaders
    task_loaders = {}
    for task_id in range(3):
        train_loader, test_loader = data.get_task_dataloaders(task_id, batch_size=32)
        task_loaders[task_id] = (train_loader, test_loader)
        print(f"Task {task_id}: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")
    
    # Test different models
    models_to_test = [
        ("Progressive Neural Network", ProgressiveNeuralNetwork(64, 128, 5), "progressive"),
        ("Elastic Weight Consolidation", ElasticWeightConsolidation(64, 128, 5), "ewc"),
        ("Learning Without Forgetting", LearningWithoutForgetting(64, 128, 5), "lwf"),
        ("Simple MLP (Baseline)", SimpleMLP(64, 128, 5), "default")
    ]
    
    results = {}
    
    for model_name, model, model_type in models_to_test:
        print(f"\n🚀 Training {model_name}...")
        print("-" * 40)
        
        # Run experiment
        experiment_results = run_continual_learning_experiment(
            model=model,
            task_loaders=task_loaders,
            model_type=model_type,
            epochs_per_task=5,  # Reduced for demo
            verbose=True
        )
        
        results[model_name] = experiment_results
        
        # Print key metrics
        cl_metrics = experiment_results['continual_learning_metrics']
        print(f"\n📈 {model_name} Results:")
        print(f"  Average Accuracy: {cl_metrics.get('average_accuracy', 0):.3f}")
        print(f"  Average Forgetting: {cl_metrics.get('average_forgetting', 0):.3f}")
        print(f"  Average Backward Transfer: {cl_metrics.get('average_backward_transfer', 0):.3f}")
    
    # Summary comparison
    print("\n🏆 Summary Comparison:")
    print("=" * 50)
    print(f"{'Model':<30} {'Avg Acc':<10} {'Forgetting':<12} {'Backward Transfer':<18}")
    print("-" * 70)
    
    for model_name, experiment_results in results.items():
        cl_metrics = experiment_results['continual_learning_metrics']
        avg_acc = cl_metrics.get('average_accuracy', 0)
        forgetting = cl_metrics.get('average_forgetting', 0)
        backward_transfer = cl_metrics.get('average_backward_transfer', 0)
        
        print(f"{model_name:<30} {avg_acc:<10.3f} {forgetting:<12.3f} {backward_transfer:<18.3f}")
    
    print("\n✅ Experiment completed!")
    print("\n💡 Key Insights:")
    print("- Progressive Neural Networks should show low forgetting")
    print("- EWC uses regularization to prevent catastrophic forgetting")
    print("- LwF uses knowledge distillation to retain previous knowledge")
    print("- Simple MLP serves as a baseline showing catastrophic forgetting")
    
    print("\n⚠️  Remember: This is a research demo. Results may vary with different:")
    print("- Dataset sizes and complexity")
    print("- Model architectures and hyperparameters")
    print("- Training procedures and optimization settings")


if __name__ == "__main__":
    main()
