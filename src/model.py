import sys
sys.path.append('/Users/enes/Downloads/teknasyon-case-ai-datascientist/challenge/SMS Spam Detection')
"""
SMS Spam Detection model implementation.
"""

import logging
import pickle
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from .preprocessing import TextPreprocessor
from .features import FeatureExtractor

class SMSSpamDetector:
    """SMS Spam Detection model class."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the SMS Spam Detector.
        
        Args:
            model_path: Optional path to load a pre-trained model
        """
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model = LinearSVC(class_weight='balanced', random_state=42)
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_data(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Prepare text data for model training or prediction.
        
        Args:
            texts: List of SMS messages
            fit: Whether to fit the feature extractor (True for training)
            
        Returns:
            Processed features ready for the model
        """
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_texts(texts)
        
        # Extract features
        if fit:
            features = self.feature_extractor.fit_transform(processed_texts)
        else:
            features = self.feature_extractor.transform(processed_texts)
        
        return features
    
    def train(self, texts: List[str], labels: List[str]) -> None:
        """
        Train the spam detection model.
        
        Args:
            texts: List of SMS messages
            labels: List of corresponding labels
        """
        # Prepare data with fit=True for training
        X = self.prepare_data(texts, fit=True)
        
        # Train model
        self.model.fit(X, labels)
        
        logging.info("Model training completed")
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict spam probability for given texts.
        
        Args:
            texts: List of SMS messages
            
        Returns:
            List of predictions
        """
        # Prepare data
        X = self.prepare_data(texts, fit=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            texts: List of SMS messages
            true_labels: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(texts)
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return report
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_extractor': self.feature_extractor
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data['preprocessor']
        self.feature_extractor = model_data['feature_extractor']
        
        logging.info(f"Model loaded from {path}")