# Progressive Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of **Progressive Neural Networks** and continual learning methods for research and education. This project demonstrates how neural networks can learn new tasks sequentially without forgetting previously learned knowledge.

## ⚠️ Important Disclaimers

> **🚨 RESEARCH AND EDUCATIONAL USE ONLY**
> 
> This implementation is designed for **research, education, and demonstration purposes only**. It is **NOT intended for production use** or real-world decision making. Results shown are for illustrative purposes and may not reflect actual performance in real-world scenarios.

### Safety & Ethics Considerations

- **Not for Production**: This code should not be used in production systems without extensive validation
- **Research Context**: Continual learning models require careful evaluation and domain-specific adaptation
- **Human Oversight**: All AI systems should include human oversight and validation mechanisms
- **Bias & Fairness**: Models may exhibit biases present in training data; regular auditing recommended
- **Privacy**: Ensure compliance with data protection regulations when handling personal data

## What Are Progressive Neural Networks?

Progressive Neural Networks (PNNs) are a continual learning approach that:

- **Add new "columns"** for each new task while keeping previous columns frozen
- **Prevent catastrophic forgetting** by preserving learned representations
- **Enable knowledge transfer** through lateral connections between columns
- **Scale gracefully** as more tasks are learned

### Key Concepts

- **Continual Learning**: Learning new tasks without forgetting old ones
- **Catastrophic Forgetting**: The tendency of neural networks to forget previous tasks when learning new ones
- **Knowledge Transfer**: Using knowledge from previous tasks to help learn new tasks
- **Backward Transfer**: How learning new tasks affects performance on old tasks
- **Forward Transfer**: How learning old tasks helps performance on new tasks

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- Apple Silicon support (MPS) available

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kryptologyst/Progressive-Neural-Networks.git
   cd Progressive-Neural-Networks
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run the interactive demo:**
   ```bash
   streamlit run demo/app.py
   ```

### Basic Usage

```python
from progressive_nn.models.core import ProgressiveNeuralNetwork
from progressive_nn.data import DigitsContinualLearning
from progressive_nn.train import run_continual_learning_experiment

# Create dataset
data = DigitsContinualLearning(num_tasks=3)

# Create model
model = ProgressiveNeuralNetwork(input_size=64, hidden_size=128, output_size=5)

# Run experiment
results = run_continual_learning_experiment(model, task_loaders, "progressive")
```

## Implemented Methods

### Core Models

| Method | Description | Key Features |
|--------|-------------|--------------|
| **Progressive Neural Networks** | Original PNN implementation | Column-based architecture, lateral connections |
| **Elastic Weight Consolidation (EWC)** | Regularization-based approach | Fisher information matrix, parameter importance |
| **Learning Without Forgetting (LwF)** | Knowledge distillation approach | Teacher-student learning, soft targets |
| **PackNet** | Pruning-based approach | Dynamic pruning, task-specific masks |
| **Simple MLP** | Baseline comparison | Standard multi-layer perceptron |

### Datasets

- **Digits Dataset**: 10-class classification split into sequential tasks
- **Synthetic Dataset**: Generated classification tasks with configurable parameters
- **Permuted MNIST**: Simplified version using digits with random permutations

## Running Experiments

### Command Line Interface

```bash
# Train Progressive Neural Network on digits dataset
python scripts/train.py model=progressive data=digits

# Train EWC model with custom parameters
python scripts/train.py model=ewc data=synthetic training.epochs_per_task=15

# Train with different configurations
python scripts/train.py model=lwf data=digits training.learning_rate=0.0005
```

### Configuration Files

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations  
- `configs/training/`: Training configurations

### Interactive Demo

Launch the Streamlit demo for interactive experimentation:

```bash
streamlit run demo/app.py
```

Features:
- **Model Selection**: Choose between different continual learning approaches
- **Parameter Tuning**: Adjust hyperparameters in real-time
- **Visualization**: Interactive plots of learning curves and metrics
- **Comparison**: Side-by-side performance comparisons

## Evaluation Metrics

### Continual Learning Metrics

- **Average Accuracy**: Overall performance across all tasks
- **Backward Transfer**: How learning new tasks affects old tasks
- **Forward Transfer**: How learning old tasks helps new tasks
- **Forgetting**: How much performance degrades on old tasks
- **Learning Efficiency**: Sample efficiency across tasks

### Visualization

- **Learning Curves**: Training loss and accuracy over time
- **Task Performance**: Accuracy comparison across tasks
- **Transfer Analysis**: Backward and forward transfer visualization
- **Confusion Matrices**: Detailed classification performance

## Project Structure

```
progressive-neural-networks/
├── src/progressive_nn/           # Main source code
│   ├── models/                   # Model implementations
│   │   └── core.py              # Core continual learning models
│   ├── data/                    # Data loading utilities
│   │   └── __init__.py          # Dataset classes
│   ├── train/                   # Training utilities
│   │   └── __init__.py          # Training loops and trainers
│   ├── metrics/                 # Evaluation metrics
│   │   └── __init__.py          # Continual learning metrics
│   └── utils.py                 # Utility functions
├── configs/                     # Configuration files
│   ├── model/                   # Model configurations
│   ├── data/                    # Dataset configurations
│   └── training/                # Training configurations
├── scripts/                     # Training scripts
│   └── train.py                 # Main training script
├── demo/                        # Interactive demo
│   └── app.py                   # Streamlit application
├── tests/                       # Unit tests
├── assets/                      # Generated assets
├── data/                        # Data storage
└── logs/                        # Training logs
```

## Research Applications

### Academic Research

- **Continual Learning**: Study of methods to prevent catastrophic forgetting
- **Meta-Learning**: Investigation of learning-to-learn approaches
- **Transfer Learning**: Analysis of knowledge transfer between tasks
- **Neural Architecture Search**: Exploration of dynamic architectures

### Educational Use

- **Machine Learning Courses**: Hands-on continual learning demonstrations
- **Research Methodology**: Example of proper experimental design
- **Code Quality**: Demonstration of clean, documented research code
- **Visualization**: Interactive learning tools for complex concepts

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ tests/
ruff check src/ tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Quality

- **Type Hints**: All functions include type annotations
- **Documentation**: Google-style docstrings throughout
- **Testing**: Comprehensive test coverage
- **Linting**: Black formatting, Ruff linting
- **MyPy**: Static type checking

## References

### Key Papers

1. **Progressive Neural Networks**
   - Rusu, A. A., et al. "Progressive neural networks." arXiv preprint arXiv:1606.04671 (2016).

2. **Elastic Weight Consolidation**
   - Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." PNAS 114.13 (2017): 3521-3526.

3. **Learning Without Forgetting**
   - Li, Z., & Hoiem, D. "Learning without forgetting." ECCV 2016.

4. **PackNet**
   - Mallya, A., & Lazebnik, S. "Packnet: Adding multiple tasks to a single network by iterative pruning." CVPR 2018.

### Additional Resources

- [Continual Learning Survey](https://arxiv.org/abs/1810.13166)
- [Avalanche Framework](https://avalanche.continualai.org/)
- [Continual Learning Papers](https://github.com/optimass/continual_learning_papers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**kryptologyst**

- GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)
- This project is part of the "1000 AI Projects" series

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Continual learning research community for foundational work
- Streamlit team for the interactive demo framework
- Hydra team for configuration management

## Support

For questions, issues, or contributions:

1. **Issues**: Use GitHub Issues for bug reports and feature requests
2. **Discussions**: Use GitHub Discussions for questions and general discussion
3. **Pull Requests**: Welcome contributions following the contribution guidelines

---

**Remember**: This is a research and educational tool. Always validate results and consider safety implications before any real-world application.
# Progressive-Neural-Networks
