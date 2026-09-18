"""
Feature engineering module with TF-IDF and n-gram support.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureEngineer:
    """TF-IDF feature extraction with n-gram support and analysis."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2), stop_words="english"):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            sublinear_tf=True,  # Apply log normalization
        )
        self.count_vec = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            stop_words=stop_words,
        )
        self._is_fitted = False
    
    def fit_transform(self, X_train):
        """Fit TF-IDF on training data and return transformed features."""
        X_transformed = self.tfidf.fit_transform(X_train)
        self._is_fitted = True
        return X_transformed
    
    def transform(self, X):
        """Transform new data using the fitted vectorizer."""
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fit before transform.")
        return self.tfidf.transform(X)
    
    def get_feature_names(self):
        """Return feature names from the fitted vectorizer."""
        return self.tfidf.get_feature_names_out()
    
    def get_top_features_per_class(self, X_train, y_train, top_n=15):
        """Get the top TF-IDF features for each class."""
        if not self._is_fitted:
            raise ValueError("Must fit first.")
        
        feature_names = self.get_feature_names()
        top_features = {}
        
        for label in sorted(y_train.unique()):
            mask = (y_train == label).values
            indices = np.where(mask)[0]
            class_tfidf = X_train[indices].mean(axis=0)
            class_tfidf = np.asarray(class_tfidf).flatten()
            top_indices = class_tfidf.argsort()[-top_n:][::-1]
            top_features[label] = [
                (feature_names[i], class_tfidf[i]) for i in top_indices
            ]
        
        return top_features
    
    def compare_vectorizers(self, X_train, X_test, y_train, y_test):
        """Compare CountVectorizer vs TF-IDF accuracy using Naive Bayes."""
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score
        
        # CountVectorizer (unigrams only)
        cv = CountVectorizer(max_features=self.max_features, stop_words=self.stop_words)
        X_train_cv = cv.fit_transform(X_train)
        X_test_cv = cv.transform(X_test)
        nb_cv = MultinomialNB()
        nb_cv.fit(X_train_cv, y_train)
        acc_cv = accuracy_score(y_test, nb_cv.predict(X_test_cv))
        
        # TF-IDF (unigrams + bigrams)
        X_train_tfidf = self.fit_transform(X_train)
        X_test_tfidf = self.transform(X_test)
        nb_tfidf = MultinomialNB()
        nb_tfidf.fit(X_train_tfidf, y_train)
        acc_tfidf = accuracy_score(y_test, nb_tfidf.predict(X_test_tfidf))
        
        return {
            "count_vectorizer": {"accuracy": acc_cv, "features": X_train_cv.shape[1]},
            "tfidf": {"accuracy": acc_tfidf, "features": X_train_tfidf.shape[1]},
        }
    
    def get_vocab_size(self):
        """Return the vocabulary size of the fitted vectorizer."""
        if self._is_fitted:
            return len(self.tfidf.vocabulary_)
        return 0
