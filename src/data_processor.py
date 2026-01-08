
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

class DataProcessor:
    def __init__(self):
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Cleans the input text by:
        1. Removing non-alphabetic characters
        2. Lowercasing
        3. Tokenizing
        4. Removing stopwords
        5. Lemmatizing
        """
        if not isinstance(text, str):
            return ""
        
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        return ' '.join(cleaned_tokens)

    def load_data(self, filepath):
        """Loads data from a CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)

    def prepare_data(self, df, text_column='Statement', label_column='Label'):
        """Prepares features (X) and labels (y) from dataframe."""
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Columns {text_column} and {label_column} must exist in dataframe")
        
        # Apply cleaning
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        X = df['cleaned_text']
        y = df[label_column]
        return X, y
