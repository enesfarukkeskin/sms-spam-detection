"""
Unit tests for text preprocessing module.
"""

import pytest
from src.preprocessing import TextPreprocessor

def test_basic_preprocessing():
    """Test basic text preprocessing functionality."""
    preprocessor = TextPreprocessor()
    text = "Hello World! This is a test message."
    processed = preprocessor.preprocess_text(text)
    assert isinstance(processed, str)
    assert processed.islower()
    assert "!" not in processed

def test_url_removal():
    """Test URL removal functionality."""
    preprocessor = TextPreprocessor(remove_urls=True)
    text = "Check this link https://example.com and www.test.com"
    processed = preprocessor.preprocess_text(text)
    assert "https://" not in processed
    assert "www." not in processed

def test_number_removal():
    """Test number removal functionality."""
    preprocessor = TextPreprocessor(remove_numbers=True)
    text = "Call me at 12345 or send 100 dollars"
    processed = preprocessor.preprocess_text(text)
    assert not any(char.isdigit() for char in processed)

def test_stopword_removal():
    """Test stopword removal functionality."""
    preprocessor = TextPreprocessor(remove_stops=True)
    text = "this is a test message"
    processed = preprocessor.preprocess_text(text)
    assert "is" not in processed.split()
    assert "a" not in processed.split()

def test_custom_stopwords():
    """Test custom stopwords functionality."""
    custom_stops = ["test", "custom"]
    preprocessor = TextPreprocessor(custom_stops=custom_stops)
    text = "this is a test custom message"
    processed = preprocessor.preprocess_text(text)
    assert "test" not in processed.split()
    assert "custom" not in processed.split()

def test_minimum_word_length():
    """Test minimum word length functionality."""
    preprocessor = TextPreprocessor(min_word_length=3)
    text = "a ab abc abcd"
    processed = preprocessor.preprocess_text(text)
    words = processed.split()
    assert all(len(word) >= 3 for word in words)

def test_turkish_character_handling():
    """Test handling of Turkish characters."""
    preprocessor = TextPreprocessor()
    text = "Merhaba dünya çğıöşü"
    processed = preprocessor.preprocess_text(text)
    assert "ç" in processed
    assert "ğ" in processed
    assert "ı" in processed
    assert "ö" in processed
    assert "ş" in processed
    assert "ü" in processed

def test_special_token_extraction():
    """Test special token extraction functionality."""
    preprocessor = TextPreprocessor()
    text = "Call 12345 or visit https://example.com"
    tokens = preprocessor.get_special_tokens(text)
    assert "12345" in tokens
    assert "https://example.com" in tokens

def test_batch_processing():
    """Test batch text processing functionality."""
    preprocessor = TextPreprocessor()
    texts = ["Hello World!", "Test Message 123", "https://example.com"]
    processed = preprocessor.preprocess_texts(texts)
    assert len(processed) == len(texts)
    assert all(isinstance(text, str) for text in processed)

def test_stopword_management():
    """Test stopword management functionality."""
    preprocessor = TextPreprocessor()
    initial_stopwords = preprocessor.get_stopwords()
    
    # Add custom stopwords
    new_stopwords = ["custom1", "custom2"]
    preprocessor.add_custom_stopwords(new_stopwords)
    assert all(word in preprocessor.get_stopwords() for word in new_stopwords)
    
    # Remove custom stopwords
    preprocessor.remove_custom_stopwords(new_stopwords)
    assert all(word not in preprocessor.get_stopwords() for word in new_stopwords)

def test_error_handling():
    """Test error handling in preprocessing."""
    preprocessor = TextPreprocessor()
    
    # Test with None input
    with pytest.raises(AttributeError):
        preprocessor.preprocess_text(None)
    
    # Test with non-string input
    with pytest.raises(AttributeError):
        preprocessor.preprocess_text(123)
    
    # Test with empty string
    assert preprocessor.preprocess_text("") == ""

def test_lemmatization():
    """Test lemmatization functionality."""
    preprocessor = TextPreprocessor(lemmatize=True)
    text = "running cars cities"
    processed = preprocessor.preprocess_text(text)
    assert "running" not in processed
    assert "run" in processed
    assert "cars" not in processed
    assert "car" in processed
    assert "cities" not in processed
    assert "city" in processed

def test_preprocessor_initialization():
    """Test preprocessor initialization with different parameters."""
    # Test with all features disabled
    preprocessor = TextPreprocessor(
        remove_stops=False,
        remove_urls=False,
        remove_numbers=False,
        lemmatize=False
    )
    text = "Hello123 http://test.com"
    processed = preprocessor.preprocess_text(text)
    assert "123" in processed
    assert "http" in processed
    
    # Test with all features enabled
    preprocessor = TextPreprocessor(
        remove_stops=True,
        remove_urls=True,
        remove_numbers=True,
        lemmatize=True
    )
    processed = preprocessor.preprocess_text(text)
    assert "123" not in processed
    assert "http" not in processed

if __name__ == "__main__":
    pytest.main([__file__])