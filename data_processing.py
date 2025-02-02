"""
Data processing script for SMS Spam Detection.
"""

import pandas as pd
import numpy as np
from src.preprocessing import TextPreprocessor
from src.features import FeatureExtractor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_check_data():
    """Load and check the data for any issues."""
    logger.info("Loading data...")
    df = pd.read_csv('data/raw/sms_spam_train.csv')
    
    # Display basic information
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nSample Data:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Check class distribution
    print("\nClass Distribution:")
    print(df['Label'].value_counts())
    print("\nClass Distribution (%):")
    print(df['Label'].value_counts(normalize=True) * 100)
    
    return df

import os

def preprocess_data(df):
    """Preprocess the data using our preprocessing module."""
    logger.info("Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Process messages
    logger.info("Processing text messages...")
    processed_texts = preprocessor.preprocess_texts(df['Message'].tolist())
    
    # Add processed texts to dataframe
    df['processed_text'] = processed_texts
    
    # Ensure the 'data/processed' directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    logger.info("Saving processed data...")
    df.to_csv('data/processed/processed_spam.csv', index=False)
    
    return df


def extract_features(df):
    """Extract features using our feature extraction module."""
    logger.info("Extracting features...")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Fit and transform the data
    features = feature_extractor.fit_transform(df['processed_text'])
    
    # Convert to dataframe
    feature_names = feature_extractor.get_feature_names()
    feature_df = pd.DataFrame(features, columns=feature_names)
    
    # Save features
    logger.info("Saving features...")
    feature_df.to_csv('data/processed/features.csv', index=False)
    
    return feature_df

def main():
    try:
        # 1. Load and check data
        df = load_and_check_data()
        
        # 2. Preprocess data
        processed_df = preprocess_data(df)
        
        # 3. Extract features
        feature_df = extract_features(processed_df)
        
        logger.info("Data processing completed successfully!")
        
        # Print some statistics about the processed data
        print("\nProcessed Data Statistics:")
        print("-" * 30)
        print(f"Original messages: {len(df)}")
        print(f"Processed messages: {len(processed_df)}")
        print(f"Number of features: {feature_df.shape[1]}")
        
    except Exception as e:
        logger.error(f"An error occurred during data processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()