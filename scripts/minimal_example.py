#!/usr/bin/env python3
"""Minimal working example for Progressive Neural Networks."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SimpleProgressiveNN(nn.Module):
    """Simplified Progressive Neural Network."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific columns
        self.columns = nn.ModuleDict()
        self.task_id = 0
        
        # Initialize first task
        self._add_task_column(0)
    
    def _add_task_column(self, task_id: int):
        """Add a new column for the given task."""
        self.columns[str(task_id)] = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.output_size)
        )
    
    def forward(self, x, task_id=None):
        """Forward pass for given task."""
        if task_id is None:
            task_id = self.task_id
        
        task_id = str(task_id)
        if task_id not in self.columns:
            raise ValueError(f"Task {task_id} not learned yet")
        
        # Forward through backbone
        backbone_out = self.backbone(x)
        
        # Forward through task-specific column
        output = self.columns[task_id](backbone_out)
        return output
    
    def learn_task(self, task_id: int):
        """Learn a new task by adding a new column."""
        if str(task_id) not in self.columns:
            self._add_task_column(task_id)
        self.task_id = task_id


def create_digits_tasks(num_tasks=3, test_size=0.2, random_state=42):
    """Create digits tasks for continual learning."""
    # Load digits dataset
    digits = load_digits()
    X = digits.data / 16.0  # Normalize to [0, 1]
    y = digits.target
    
    tasks = {}
    classes_per_task = 10 // num_tasks
    remaining_classes = 10 % num_tasks
    
    start_class = 0
    for task_id in range(num_tasks):
        # Determine number of classes for this task
        if task_id < remaining_classes:
            num_classes = classes_per_task + 1
        else:
            num_classes = classes_per_task
        
        end_class = start_class + num_classes
        
        # Get data for this task
        task_mask = (y >= start_class) & (y < end_class)
        task_X = X[task_mask]
        task_y = y[task_mask] - start_class  # Relabel to start from 0
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            task_X, task_y, test_size=test_size, 
            random_state=random_state, stratify=task_y
        )
        
        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test), 
            torch.LongTensor(y_test)
        )
        
        tasks[task_id] = {
            'train_loader': DataLoader(train_dataset, batch_size=32, shuffle=True),
            'test_loader': DataLoader(test_dataset, batch_size=32, shuffle=False),
            'num_classes': num_classes
        }
        
        start_class = end_class
    
    return tasks


def train_task(model, train_loader, test_loader, task_id, epochs=5):
    """Train model on a specific task."""
    model.learn_task(task_id)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x, task_id)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if epoch % 2 == 0:
            accuracy = 100 * correct / total
            print(f"Task {task_id}, Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x, task_id)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Task {task_id} Test Accuracy: {test_accuracy:.2f}%")
    return test_accuracy


def evaluate_all_tasks(model, tasks):
    """Evaluate model on all learned tasks."""
    model.eval()
    accuracies = {}
    
    for task_id, task_data in tasks.items():
        test_loader = task_data['test_loader']
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x, task_id)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        accuracies[task_id] = accuracy
        print(f"Task {task_id} Final Accuracy: {accuracy:.2f}%")
    
    return accuracies


def main():
    """Run the progressive neural network example."""
    
    print("🧠 Progressive Neural Networks - Minimal Example")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    print("\n📊 Creating Digits dataset...")
    tasks = create_digits_tasks(num_tasks=3, random_state=42)
    
    for task_id, task_data in tasks.items():
        train_size = len(task_data['train_loader'].dataset)
        test_size = len(task_data['test_loader'].dataset)
        print(f"Task {task_id}: {train_size} train, {test_size} test samples, {task_data['num_classes']} classes")
    
    # Create model
    print("\n🏗️ Creating Progressive Neural Network...")
    model = SimpleProgressiveNN(input_size=64, hidden_size=128, output_size=5)
    
    # Train on each task sequentially
    print("\n🚀 Training on tasks sequentially...")
    task_accuracies = {}
    
    for task_id in sorted(tasks.keys()):
        print(f"\n{'='*30}")
        print(f"Training Task {task_id}")
        print(f"{'='*30}")
        
        train_loader = tasks[task_id]['train_loader']
        test_loader = tasks[task_id]['test_loader']
        
        # Train on current task
        test_acc = train_task(model, train_loader, test_loader, task_id, epochs=5)
        task_accuracies[task_id] = test_acc
        
        # Evaluate on all previous tasks
        print(f"\nEvaluating on all learned tasks after Task {task_id}:")
        final_accuracies = evaluate_all_tasks(model, {tid: data for tid, data in tasks.items() if tid <= task_id})
    
    # Summary
    print("\n🏆 Final Results Summary:")
    print("=" * 40)
    print(f"{'Task':<8} {'Accuracy':<12}")
    print("-" * 20)
    
    for task_id, accuracy in final_accuracies.items():
        print(f"Task {task_id:<3} {accuracy:<12.2f}%")
    
    avg_accuracy = np.mean(list(final_accuracies.values()))
    print(f"{'Average':<8} {avg_accuracy:<12.2f}%")
    
    print("\n✅ Experiment completed!")
    print("\n💡 Key Insights:")
    print("- Progressive Neural Networks add new columns for each task")
    print("- Previous columns remain frozen to prevent forgetting")
    print("- This approach should show low catastrophic forgetting")
    print("- Each task gets its own specialized column")
    
    print("\n⚠️  Remember: This is a research demo. Results may vary with:")
    print("- Different datasets and task complexities")
    print("- Model architectures and hyperparameters")
    print("- Training procedures and optimization settings")


if __name__ == "__main__":
    main()
