"""
Main execution script for SMS Spam Detection.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import SMSSpamDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(df):
    """Prepare and validate data."""
    # Convert to string and handle any null values
    df['processed_text'] = df['processed_text'].astype(str)
    # Convert labels to consistent format
    df['Label'] = df['Label'].map({'spam': 'Spam', 'ham': 'Not Spam'})
    return df

def main():
    try:
        # 1. Load preprocessed data
        logger.info("Loading preprocessed data...")
        processed_df = pd.read_csv('data/processed/processed_spam.csv')
        
        # 2. Prepare data
        logger.info("Preparing data...")
        processed_df = prepare_data(processed_df)
        
        # 3. Split data
        logger.info("Splitting data into train and test sets...")
        train_df, test_df = train_test_split(
            processed_df,
            test_size=0.2,
            random_state=42,
            stratify=processed_df['Label']
        )
        
        # Log data split information
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        # 4. Initialize model
        logger.info("Initializing model...")
        detector = SMSSpamDetector()
        
        # 5. Train model using processed text
        logger.info("Training model...")
        detector.train(
            texts=train_df['processed_text'].tolist(),
            labels=train_df['Label'].tolist()
        )
        
        # 6. Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = detector.evaluate(
            texts=test_df['processed_text'].tolist(),
            true_labels=test_df['Label'].tolist()
        )
        
        # Print evaluation results
        print("\nModel Evaluation Results:")
        print("-" * 30)
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Precision (Spam): {evaluation_results['Spam']['precision']:.4f}")
        print(f"Recall (Spam): {evaluation_results['Spam']['recall']:.4f}")
        print(f"F1-score (Spam): {evaluation_results['Spam']['f1-score']:.4f}")
        
        # 7. Save model
        logger.info("Saving model...")
        detector.save_model('models/trained_model.pkl')
        
        # 8. Test with sample messages
        logger.info("Testing with sample messages...")
        sample_messages = [
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize club",
            "Hey, what time are you coming home for dinner?",
            "Congratulations! You've been selected for a free iPhone! Click here to claim",
            "Kampanya! Tüm ürünlerde %50 indirim fırsatı, hemen tıkla!",
            "Akşam yemeğinde ne yiyelim?"
        ]
        
        predictions = detector.predict(sample_messages)
        
        print("\nSample Predictions:")
        print("-" * 30)
        for message, prediction in zip(sample_messages, predictions):
            print(f"Message: {message[:50]}...")
            print(f"Prediction: {prediction}\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()