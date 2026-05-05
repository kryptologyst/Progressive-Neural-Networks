"""Streamlit demo for Progressive Neural Networks."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from progressive_nn.data import DigitsContinualLearning, SyntheticContinualLearning
from progressive_nn.models.core import (
    ProgressiveNeuralNetwork,
    ElasticWeightConsolidation,
    LearningWithoutForgetting,
    PackNet,
    SimpleMLP
)
from progressive_nn.train import run_continual_learning_experiment
from progressive_nn.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Progressive Neural Networks Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main demo application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Progressive Neural Networks Demo</h1>', 
                unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <ul>
        <li>Not intended for production use or real-world decision making</li>
        <li>Results are for illustrative purposes and may not reflect real-world performance</li>
        <li>Continual learning models require careful validation before deployment</li>
        <li>Author: <a href="https://github.com/kryptologyst">kryptologyst</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Progressive Neural Network", "Elastic Weight Consolidation (EWC)", 
             "Learning Without Forgetting (LwF)", "PackNet", "Simple MLP (Baseline)"],
            help="Choose the continual learning approach to demonstrate"
        )
        
        # Dataset selection
        dataset_type = st.selectbox(
            "Select Dataset",
            ["Digits Dataset", "Synthetic Dataset"],
            help="Choose the dataset for the experiment"
        )
        
        # Experiment parameters
        st.subheader("📊 Experiment Parameters")
        num_tasks = st.slider("Number of Tasks", 2, 5, 3, 
                            help="Number of sequential tasks to learn")
        epochs_per_task = st.slider("Epochs per Task", 5, 20, 10,
                                  help="Number of training epochs per task")
        batch_size = st.slider("Batch Size", 16, 64, 32,
                             help="Training batch size")
        
        # Model parameters
        st.subheader("🏗️ Model Parameters")
        hidden_size = st.slider("Hidden Size", 64, 256, 128,
                              help="Size of hidden layers")
        
        # Advanced parameters
        with st.expander("🔬 Advanced Parameters"):
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            
            if "EWC" in model_type:
                ewc_lambda = st.number_input("EWC Lambda", 100.0, 5000.0, 1000.0, 100.0)
            elif "LwF" in model_type:
                temperature = st.number_input("Distillation Temperature", 1.0, 10.0, 3.0, 0.1)
            elif "PackNet" in model_type:
                prune_ratio = st.number_input("Prune Ratio", 0.1, 0.9, 0.5, 0.1)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Run Experiment")
        
        if st.button("▶️ Start Training", type="primary"):
            run_experiment(model_type, dataset_type, num_tasks, epochs_per_task, 
                         batch_size, hidden_size, learning_rate)
    
    with col2:
        st.header("📈 Quick Stats")
        st.info("""
        **Progressive Neural Networks** are designed to learn new tasks without forgetting previous ones by adding new "columns" while keeping old ones frozen.
        
        **Key Metrics:**
        - **Backward Transfer**: How learning new tasks affects old tasks
        - **Forward Transfer**: How learning old tasks helps new tasks  
        - **Forgetting**: How much performance degrades on old tasks
        """)


def run_experiment(model_type: str, dataset_type: str, num_tasks: int, 
                  epochs_per_task: int, batch_size: int, hidden_size: int, 
                  learning_rate: float):
    """Run the continual learning experiment."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create dataset
        status_text.text("Creating dataset...")
        progress_bar.progress(10)
        
        if dataset_type == "Digits Dataset":
            data = DigitsContinualLearning(num_tasks=num_tasks, random_state=42)
            input_size = 64
            output_size = 10 // num_tasks
        else:
            data = SyntheticContinualLearning(
                num_tasks=num_tasks, 
                input_dim=20, 
                num_classes_per_task=2,
                random_state=42
            )
            input_size = 20
            output_size = 2
        
        # Create task loaders
        task_loaders = {}
        for task_id in range(num_tasks):
            train_loader, test_loader = data.get_task_dataloaders(task_id, batch_size)
            task_loaders[task_id] = (train_loader, test_loader)
        
        progress_bar.progress(20)
        
        # Create model
        status_text.text("Creating model...")
        progress_bar.progress(30)
        
        if model_type == "Progressive Neural Network":
            model = ProgressiveNeuralNetwork(input_size, hidden_size, output_size)
            model_type_key = "progressive"
        elif model_type == "Elastic Weight Consolidation (EWC)":
            model = ElasticWeightConsolidation(input_size, hidden_size, output_size, 
                                            ewc_lambda=st.session_state.get('ewc_lambda', 1000.0))
            model_type_key = "ewc"
        elif model_type == "Learning Without Forgetting (LwF)":
            model = LearningWithoutForgetting(input_size, hidden_size, output_size,
                                           temperature=st.session_state.get('temperature', 3.0))
            model_type_key = "lwf"
        elif model_type == "PackNet":
            model = PackNet(input_size, hidden_size, output_size,
                          prune_ratio=st.session_state.get('prune_ratio', 0.5))
            model_type_key = "packnet"
        else:  # Simple MLP
            model = SimpleMLP(input_size, hidden_size, output_size)
            model_type_key = "default"
        
        progress_bar.progress(40)
        
        # Run experiment
        status_text.text("Training model...")
        progress_bar.progress(50)
        
        results = run_continual_learning_experiment(
            model=model,
            task_loaders=task_loaders,
            model_type=model_type_key,
            epochs_per_task=epochs_per_task,
            verbose=False
        )
        
        progress_bar.progress(90)
        
        # Display results
        status_text.text("Generating visualizations...")
        display_results(results, model_type)
        
        progress_bar.progress(100)
        status_text.text("✅ Experiment completed!")
        
    except Exception as e:
        st.error(f"❌ Error running experiment: {str(e)}")
        st.exception(e)


def display_results(results: dict, model_type: str):
    """Display experiment results."""
    
    st.header("📊 Results")
    
    # Extract metrics
    cl_metrics = results.get('continual_learning_metrics', {})
    task_results = results.get('task_results', {})
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Learning Curves", "🎯 Task Performance", 
                                      "🔄 Transfer Metrics", "📋 Summary"])
    
    with tab1:
        display_learning_curves(task_results)
    
    with tab2:
        display_task_performance(task_results)
    
    with tab3:
        display_transfer_metrics(cl_metrics)
    
    with tab4:
        display_summary(cl_metrics, model_type)


def display_learning_curves(task_results: dict):
    """Display learning curves for each task."""
    
    fig = make_subplots(
        rows=len(task_results), cols=1,
        subplot_titles=[f"Task {task_id} Learning Curve" for task_id in task_results.keys()],
        vertical_spacing=0.1
    )
    
    for i, (task_id, metrics) in enumerate(task_results.items(), 1):
        epochs = list(range(1, len(metrics['train_losses']) + 1))
        
        # Add loss curve
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train_losses'], 
                      mode='lines', name=f'Task {task_id} Loss',
                      line=dict(color=f'hsl({i*60}, 70%, 50%)')),
            row=i, col=1
        )
        
        # Add accuracy curve
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['train_accuracies'], 
                      mode='lines', name=f'Task {task_id} Accuracy',
                      line=dict(color=f'hsl({i*60}, 70%, 50%)', dash='dash')),
            row=i, col=1
        )
    
    fig.update_layout(
        title="Learning Curves Across Tasks",
        height=300 * len(task_results),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss / Accuracy")
    
    st.plotly_chart(fig, use_container_width=True)


def display_task_performance(task_results: dict):
    """Display task performance comparison."""
    
    # Create performance dataframe
    performance_data = []
    for task_id, metrics in task_results.items():
        performance_data.append({
            'Task': f'Task {task_id}',
            'Final Train Accuracy': f"{metrics['final_train_accuracy']:.3f}",
            'Test Accuracy': f"{metrics['test_accuracy']:.3f}",
            'Final Loss': f"{metrics['final_train_loss']:.3f}"
        })
    
    df = pd.DataFrame(performance_data)
    
    # Display table
    st.subheader("Task Performance Summary")
    st.dataframe(df, use_container_width=True)
    
    # Create accuracy comparison chart
    fig = px.bar(
        df, x='Task', y='Test Accuracy',
        title="Test Accuracy by Task",
        color='Test Accuracy',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def display_transfer_metrics(cl_metrics: dict):
    """Display continual learning transfer metrics."""
    
    if not cl_metrics:
        st.warning("No transfer metrics available.")
        return
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_acc = cl_metrics.get('average_accuracy', 0)
        st.metric("Average Accuracy", f"{avg_acc:.3f}")
    
    with col2:
        avg_bt = cl_metrics.get('average_backward_transfer', 0)
        st.metric("Avg Backward Transfer", f"{avg_bt:.3f}")
    
    with col3:
        avg_ft = cl_metrics.get('average_forward_transfer', 0)
        st.metric("Avg Forward Transfer", f"{avg_ft:.3f}")
    
    with col4:
        avg_forget = cl_metrics.get('average_forgetting', 0)
        st.metric("Avg Forgetting", f"{avg_forget:.3f}")
    
    # Detailed metrics table
    st.subheader("Detailed Transfer Metrics")
    
    detailed_metrics = []
    for key, value in cl_metrics.items():
        if isinstance(value, (int, float)):
            detailed_metrics.append({
                'Metric': key.replace('_', ' ').title(),
                'Value': f"{value:.4f}"
            })
    
    if detailed_metrics:
        df_metrics = pd.DataFrame(detailed_metrics)
        st.dataframe(df_metrics, use_container_width=True)


def display_summary(cl_metrics: dict, model_type: str):
    """Display experiment summary."""
    
    st.subheader("🎯 Experiment Summary")
    
    # Model info
    st.markdown(f"**Model Type:** {model_type}")
    st.markdown(f"**Total Tasks:** {len(cl_metrics.get('task_results', {}))}")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    avg_acc = cl_metrics.get('average_accuracy', 0)
    avg_forget = cl_metrics.get('average_forgetting', 0)
    
    if avg_acc > 0.8:
        st.success("✅ **High Performance**: Model achieved good overall accuracy")
    elif avg_acc > 0.6:
        st.info("ℹ️ **Moderate Performance**: Model shows reasonable learning capability")
    else:
        st.warning("⚠️ **Low Performance**: Model may need tuning or more training")
    
    if avg_forget < 0.1:
        st.success("✅ **Low Forgetting**: Model retains knowledge well across tasks")
    elif avg_forget < 0.3:
        st.info("ℹ️ **Moderate Forgetting**: Some knowledge loss observed")
    else:
        st.warning("⚠️ **High Forgetting**: Significant knowledge loss between tasks")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if avg_acc < 0.7:
        recommendations.append("• Increase training epochs or learning rate")
        recommendations.append("• Try different model architectures")
    
    if avg_forget > 0.2:
        recommendations.append("• Consider stronger regularization techniques")
        recommendations.append("• Implement replay mechanisms")
    
    if not recommendations:
        recommendations.append("• Model performance looks good!")
        recommendations.append("• Try experimenting with different datasets")
    
    for rec in recommendations:
        st.markdown(rec)


if __name__ == "__main__":
    main()
