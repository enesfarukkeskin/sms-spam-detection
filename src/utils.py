"""
Utility functions for the SMS Spam Detection project.
"""

import os
import pickle
import logging
import pandas as pd
from typing import Any, Dict, Tuple, Union
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split the dataset into train and test sets.

    Args:
        file_path (str): Path to the CSV file
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['Message', 'Label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
        
        # Split the data
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['Label']
        )
        
        logger.info(f"Data split completed. Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_model(model: Any, file_path: str) -> None:
    """
    Save model to disk.

    Args:
        model: The model object to save
        file_path (str): Path where to save the model
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {file_path}")
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(file_path: str) -> Any:
    """
    Load model from disk.

    Args:
        file_path (str): Path to the saved model

    Returns:
        The loaded model object
    """
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {file_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_metrics(y_true: list, y_pred: list) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        logger.info("Evaluation metrics calculated successfully")
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise

def plot_confusion_matrix(y_true: list, y_pred: list, labels: list = None) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        labels (list, optional): List of labels. Defaults to None.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if labels:
            plt.xticks(range(len(labels)), labels)
            plt.yticks(range(len(labels)), labels)
        plt.tight_layout()
        plt.show()
        logger.info("Confusion matrix plotted successfully")
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def get_class_distribution(df: pd.DataFrame, label_column: str = 'Label') -> Dict[str, int]:
    """
    Get class distribution in the dataset.

    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of the label column

    Returns:
        Dict[str, int]: Dictionary containing class distribution
    """
    try:
        distribution = df[label_column].value_counts().to_dict()
        logger.info("Class distribution calculated successfully")
        return distribution
    
    except Exception as e:
        logger.error(f"Error getting class distribution: {str(e)}")
        raise

def setup_experiment_tracking(experiment_name: str) -> None:
    """
    Setup experiment tracking (placeholder for MLflow or other tracking tools).

    Args:
        experiment_name (str): Name of the experiment
    """
    # This is a placeholder for implementing experiment tracking
    # You can implement MLflow, Weights & Biases, or other tracking tools here
    logger.info(f"Experiment tracking setup for: {experiment_name}")
    pass