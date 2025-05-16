import re
import string
from typing import List, Optional

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text (str): Raw text content
        
    Returns:
        str: Cleaned and normalized text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_features(text: str) -> List[str]:
    """
    Extract relevant features from text for classification.
    
    Args:
        text (str): Cleaned text content
        
    Returns:
        List[str]: List of extracted features
    """
    features = []
    
    # Add word-based features
    words = text.split()
    features.extend(words)
    
    # Add bigram features
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    features.extend(bigrams)
    
    return features

def truncate_text(text: str, max_words: int = 1000) -> str:
    """
    Truncate text to a maximum number of words.
    
    Args:
        text (str): Input text
        max_words (int): Maximum number of words to keep
        
    Returns:
        str: Truncated text
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

def get_key_phrases(text: str, n: int = 5) -> List[str]:
    """
    Extract key phrases from text that might indicate fake news.
    
    Args:
        text (str): Input text
        n (int): Number of key phrases to return
        
    Returns:
        List[str]: List of key phrases
    """
    # Common phrases that might indicate fake news
    suspicious_patterns = [
        r'you won\'t believe',
        r'shocking',
        r'mind blowing',
        r'conspiracy',
        r'secret',
        r'they don\'t want you to know',
        r'miracle',
        r'cure',
        r'click here',
        r'share now'
    ]
    
    matches = []
    for pattern in suspicious_patterns:
        if re.search(pattern, text.lower()):
            matches.append(pattern)
    
    return matches[:n]

def analyze_sentiment(text: str) -> Optional[float]:
    """
    Simple sentiment analysis of text.
    Returns a score between -1 (negative) and 1 (positive).
    
    Args:
        text (str): Input text
        
    Returns:
        Optional[float]: Sentiment score or None if analysis fails
    """
    try:
        # This is a very basic implementation
        # In a real application, you would use a proper sentiment analysis library
        positive_words = {'good', 'great', 'awesome', 'excellent', 'happy', 'positive', 'true', 'fact'}
        negative_words = {'bad', 'terrible', 'awful', 'poor', 'negative', 'fake', 'false', 'hoax'}
        
        words = set(text.lower().split())
        
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        return (positive_count - negative_count) / total
        
    except Exception:
        return None 