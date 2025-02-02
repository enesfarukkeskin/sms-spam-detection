"""
Text preprocessing module for SMS Spam Detection.
"""

import re
import logging
from typing import List, Optional, Union

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Download required NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except Exception as e:
    logging.warning(f"Error downloading NLTK resources: {str(e)}")

class TextPreprocessor:
    """
    A class for text preprocessing operations.
    """
    
    def __init__(
        self,
        language: str = 'english',
        remove_stops: bool = True,
        remove_urls: bool = True,
        remove_numbers: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        min_word_length: int = 2,
        custom_stops: Optional[List[str]] = None
    ):
        """
        Initialize the TextPreprocessor.

        Args:
            language (str): Language for stopwords ('english' or 'turkish')
            remove_stops (bool): Whether to remove stopwords
            remove_urls (bool): Whether to remove URLs
            remove_numbers (bool): Whether to remove numbers
            lemmatize (bool): Whether to apply lemmatization
            stem (bool): Whether to apply stemming
            min_word_length (int): Minimum word length to keep
            custom_stops (List[str], optional): Custom stopwords to add
        """
        self.language = language
        self.remove_stops = remove_stops
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.lemmatize = lemmatize
        self.stem = stem
        self.min_word_length = min_word_length
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = SnowballStemmer(language) if stem else None
        
        # Setup stopwords
        self.stop_words = set()
        if remove_stops:
            self.stop_words.update(stopwords.words(language))
            if language == 'english':
                # Add Turkish stopwords for multilingual support
                self.stop_words.update(stopwords.words('turkish'))
        if custom_stops:
            self.stop_words.update(custom_stops)
            
        # Compile regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^a-zçğıöşüA-ZÇĞİÖŞÜ\s]')
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text input.

        Args:
            text (str): Input text

        Returns:
            str: Preprocessed text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs if specified
            if self.remove_urls:
                text = self.url_pattern.sub(' ', text)
            
            # Remove numbers if specified
            if self.remove_numbers:
                text = self.number_pattern.sub(' ', text)
            
            # Remove special characters
            text = self.special_char_pattern.sub(' ', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Process tokens
            processed_tokens = []
            for token in tokens:
                # Skip short words
                if len(token) < self.min_word_length:
                    continue
                    
                # Skip stopwords
                if self.remove_stops and token in self.stop_words:
                    continue
                
                # Apply lemmatization
                if self.lemmatize:
                    token = self.lemmatizer.lemmatize(token)
                
                # Apply stemming
                if self.stem:
                    token = self.stemmer.stem(token)
                
                processed_tokens.append(token)
            
            # Join tokens back into text
            return ' '.join(processed_tokens)
        
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            raise
            
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.

        Args:
            texts (List[str]): List of input texts

        Returns:
            List[str]: List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]
    
    def get_special_tokens(self, text: str) -> List[str]:
        """
        Extract special tokens (URLs, numbers, etc.) from text.

        Args:
            text (str): Input text

        Returns:
            List[str]: List of special tokens
        """
        special_tokens = []
        
        # Extract URLs
        urls = self.url_pattern.findall(text)
        if urls:
            special_tokens.extend(urls)
            
        # Extract numbers
        numbers = self.number_pattern.findall(text)
        if numbers:
            special_tokens.extend(numbers)
            
        return special_tokens
    
    def add_custom_stopwords(self, stopwords: List[str]) -> None:
        """
        Add custom stopwords to the existing set.

        Args:
            stopwords (List[str]): List of stopwords to add
        """
        self.stop_words.update(stopwords)
        
    def remove_custom_stopwords(self, stopwords: List[str]) -> None:
        """
        Remove custom stopwords from the existing set.

        Args:
            stopwords (List[str]): List of stopwords to remove
        """
        self.stop_words.difference_update(stopwords)
        
    def get_stopwords(self) -> List[str]:
        """
        Get current list of stopwords.

        Returns:
            List[str]: List of stopwords
        """
        return list(self.stop_words)
    