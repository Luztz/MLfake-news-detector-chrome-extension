import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TextSelector(BaseEstimator, TransformerMixin):
    """Select text column from dataframe"""
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

class SentimentSelector(BaseEstimator, TransformerMixin):
    """Select sentiment column from dataframe"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[['sentiment']]

class ReadabilitySelector(BaseEstimator, TransformerMixin):
    """Select readability column from dataframe"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[['readability']]

class WordCountSelector(BaseEstimator, TransformerMixin):
    """Select word_count column from dataframe"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[['word_count']]

class VaderSentimentSelector(BaseEstimator, TransformerMixin):
    """Select vader_sentiment column from dataframe"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[['vader_sentiment']] 