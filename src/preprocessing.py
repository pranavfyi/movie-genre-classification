"""
Text preprocessing module with NLTK lemmatization.
"""

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data (silent)
for resource in ["punkt", "punkt_tab", "wordnet", "stopwords", "omw-1.4"]:
    nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline with lemmatization and stop word removal."""
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    
    def clean_text(self, text):
        """Full preprocessing pipeline."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Remove punctuation and special characters
        text = re.sub(r"[^a-z\s]", "", text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Lemmatize
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return " ".join(tokens)
    
    def preprocess_series(self, series):
        """Apply preprocessing to a pandas Series."""
        return series.apply(self.clean_text)
    
    def get_text_stats(self, series):
        """Return text statistics for a pandas Series."""
        word_counts = series.apply(lambda x: len(str(x).split()))
        return {
            "avg_words": word_counts.mean(),
            "min_words": word_counts.min(),
            "max_words": word_counts.max(),
            "total_vocab": len(set(" ".join(series.astype(str)).split())),
        }
