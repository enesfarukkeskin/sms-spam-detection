"""
Feature extraction module for SMS Spam Detection.
"""

import re
import logging
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Feature extractor for SMS messages.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_custom_features: bool = True
    ):
        """
        Initialize the FeatureExtractor.

        Args:
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): The lower and upper boundary of the range of n-values for different n-grams
            min_df (int): Minimum document frequency for TF-IDF
            max_df (float): Maximum document frequency for TF-IDF
            use_custom_features (bool): Whether to use custom features in addition to TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_custom_features = use_custom_features
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\b\d{10,11}\b')
        self.currency_pattern = re.compile(r'[$€₺]|\d+\s*tl|\d+\s*lira')
        self.number_pattern = re.compile(r'\d+')
        
    def get_custom_features(self, text: str) -> Dict[str, Union[int, float]]:
        """
        Extract custom features from text.

        Args:
            text (str): Input text

        Returns:
            Dict[str, Union[int, float]]: Dictionary of custom features
        """
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'contains_url': 1 if self.url_pattern.search(text.lower()) else 0,
            'contains_email': 1 if self.email_pattern.search(text.lower()) else 0,
            'contains_phone': 1 if self.phone_pattern.search(text) else 0,
            'contains_currency': 1 if self.currency_pattern.search(text.lower()) else 0,
            'number_count': len(self.number_pattern.findall(text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
            'unique_words_ratio': len(set(text.split())) / (len(text.split()) + 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'spam_trigger_count': self._count_spam_triggers(text.lower())
        }
        return features
    
    def _count_spam_triggers(self, text: str) -> int:
        """
        Count occurrence of common spam trigger words.

        Args:
            text (str): Input text

        Returns:
            int: Count of spam trigger words
        """
        spam_triggers = {
            'free', 'win', 'winner', 'prize', 'offer', 'urgent', 'limited',
            'click', 'subscribe', 'discount', 'cash', 'money', 'guaranteed',
            'congratulations', 'selected', 'exclusive', 'reward', 'instant',
            'bedava', 'kazandınız', 'tebrikler', 'özel', 'indirim', 'fırsat',
            'acil', 'hediye', 'kampanya', 'çekiliş'
        }
        return sum(1 for word in text.split() if word in spam_triggers)
    
    def fit(self, X: List[str], y: Optional[List[str]] = None):
        """
        Fit the feature extractor.

        Args:
            X (List[str]): List of text samples
            y (Optional[List[str]]): Target labels (not used)

        Returns:
            self: The fitted feature extractor
        """
        try:
            # Fit TF-IDF vectorizer
            self.tfidf.fit(X)
            return self
        
        except Exception as e:
            logging.error(f"Error fitting feature extractor: {str(e)}")
            raise
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        Transform text samples into feature matrix.

        Args:
            X (List[str]): List of text samples

        Returns:
            np.ndarray: Feature matrix
        """
        try:
            # Transform text using TF-IDF
            tfidf_features = self.tfidf.transform(X)
            
            if not self.use_custom_features:
                return tfidf_features
            
            # Extract custom features
            custom_features = []
            for text in X:
                features = self.get_custom_features(text)
                custom_features.append(list(features.values()))
            
            # Combine TF-IDF and custom features
            custom_features_array = np.array(custom_features)
            combined_features = np.hstack((
                tfidf_features.toarray(),
                custom_features_array
            ))
            
            return combined_features
        
        except Exception as e:
            logging.error(f"Error transforming features: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names.

        Returns:
            List[str]: List of feature names
        """
        tfidf_features = self.tfidf.get_feature_names_out()
        
        if not self.use_custom_features:
            return list(tfidf_features)
        
        # Add custom feature names
        custom_features = [
            'text_length', 'word_count', 'avg_word_length', 'contains_url',
            'contains_email', 'contains_phone', 'contains_currency',
            'number_count', 'uppercase_ratio', 'unique_words_ratio',
            'exclamation_count', 'question_count', 'spam_trigger_count'
        ]
        
        return list(tfidf_features) + custom_features
    
    def get_feature_importance(self, model_coef: np.ndarray) -> pd.DataFrame:
        """
        Get feature importance based on model coefficients.

        Args:
            model_coef (np.ndarray): Model coefficients

        Returns:
            pd.DataFrame: DataFrame with feature importance scores
        """
        feature_names = self.get_feature_names()
        
        if len(model_coef) != len(feature_names):
            raise ValueError("Length of model coefficients does not match number of features")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model_coef)
        })
        
        return importance_df.sort_values('importance', ascending=False)