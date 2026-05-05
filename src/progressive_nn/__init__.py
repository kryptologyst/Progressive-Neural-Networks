"""Progressive Neural Networks package."""

__version__ = "1.0.0"
__author__ = "kryptologyst"
__email__ = "kryptologyst@example.com"

from .models.core import (
    ProgressiveNeuralNetwork,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    PackNet,
    SimpleMLP
)

from .data import (
    DigitsContinualLearning,
    SyntheticContinualLearning,
    create_permuted_mnist_tasks,
    get_device,
    set_seed
)

from .metrics import (
    ContinualLearningMetrics,
    AccuracyTracker,
    LearningCurveTracker,
    compute_task_accuracy,
    compute_forgetting_measure,
    compute_backward_transfer
)

from .train import (
    ContinualLearningTrainer,
    ProgressiveNeuralNetworkTrainer,
    EWCTrainer,
    LearningWithoutForgettingTrainer,
    create_trainer,
    run_continual_learning_experiment
)

__all__ = [
    # Models
    "ProgressiveNeuralNetwork",
    "ElasticWeightConsolidation", 
    "LearningWithoutForgetting",
    "PackNet",
    "SimpleMLP",
    
    # Data
    "DigitsContinualLearning",
    "SyntheticContinualLearning",
    "create_permuted_mnist_tasks",
    "get_device",
    "set_seed",
    
    # Metrics
    "ContinualLearningMetrics",
    "AccuracyTracker",
    "LearningCurveTracker",
    "compute_task_accuracy",
    "compute_forgetting_measure",
    "compute_backward_transfer",
    
    # Training
    "ContinualLearningTrainer",
    "ProgressiveNeuralNetworkTrainer",
    "EWCTrainer", 
    "LearningWithoutForgettingTrainer",
    "create_trainer",
    "run_continual_learning_experiment"
]
