import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import shap
from datetime import datetime
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.pipeline import Pipeline
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
import requests
import math
from urllib.parse import urlparse
import logging

# Add the server directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define TextSelector class
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key=None):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # Use a default key if not set
        key = getattr(self, 'key', None)
        if key is None:
            # Try to infer the key if possible
            if isinstance(data_dict, pd.DataFrame):
                key = data_dict.columns[0]
            elif hasattr(data_dict, 'keys'):
                key = list(data_dict.keys())[0]
            else:
                raise AttributeError("No key set for TextSelector and could not infer from input.")
            self.key = key
        if isinstance(data_dict, pd.DataFrame):
            return data_dict[key].values
        return data_dict[key]

    def __setstate__(self, state):
        # Called during unpickling
        self.__dict__.update(state)
        if 'key' not in self.__dict__:
            self.key = None

# Define MetaSelector class
class MetaSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]

# Define select_sentiment class
class select_sentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return np.array([self.sia.polarity_scores(text)['compound'] for text in texts])

# Define select_readability class
class select_readability(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return np.array([textstat.flesch_reading_ease(text) for text in texts])

# Define select_word_count class
class select_word_count(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return np.array([len(text.split()) for text in texts])

# Define select_vader_sentiment class
class select_vader_sentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return np.array([self.sia.polarity_scores(text)['compound'] for text in texts])

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

from server.utils.text_processor import clean_text, get_key_phrases
from server.utils.database import init_db, save_prediction, save_feedback
from server.config import Config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB limit
CORS(app)

# Initialize database
init_db()

# Global variable for model
model = None

def get_readability(text):
    try:
        # Pre-clean text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        cleaned_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_text)

        # Try direct calculation first
        raw_score = textstat.flesch_reading_ease(cleaned_text)
        
        # If score is extremely negative or invalid, use a fallback approach
        if raw_score < 0 or not isinstance(raw_score, (int, float)) or math.isnan(raw_score):
            print(f"Invalid direct score: {raw_score}, using fallback method")
            
            # Fallback: use a conservative difficult score based on text complexity
            word_count = len(re.findall(r'\b\w+\b', cleaned_text))
            complex_words = len([w for w in cleaned_text.split() if textstat.syllable_count(w) >= 3])
            complex_ratio = complex_words / max(1, word_count)
            
            # Higher ratio of complex words = lower readability score
            fallback_score = 50 - (complex_ratio * 100)
            raw_score = max(1, min(100, fallback_score))
            print(f"Fallback score based on complex word ratio: {raw_score:.2f}")
        
        # Ensure minimum score of 1 (extremely difficult but not zero)
        final_score = max(1, min(100, raw_score))
        
        # Get interpretation based on score ranges
        if final_score >= 90:
            interpretation = "Very easy (5th grade)"
            difficulty_level = "Very Easy"
        elif final_score >= 60:
            interpretation = "Standard (8-9th grade)"
            difficulty_level = "Standard"
        elif final_score >= 30:
            interpretation = "Difficult (college)"
            difficulty_level = "Difficult"
        else:
            interpretation = "Very difficult (academic or obfuscated)"
            difficulty_level = "Very Difficult"
        
        print(f"\nReadability Analysis:")
        print(f"Raw score: {raw_score:.2f}")
        print(f"Final score: {final_score:.2f}")
        print(f"Interpretation: {interpretation}")
        
        # Store interpretation for explanation generation
        if not hasattr(get_readability, 'last_interpretation'):
            get_readability.last_interpretation = {}
        get_readability.last_interpretation = {
            'score': final_score,
            'level': difficulty_level,
            'interpretation': interpretation
        }
        
        return final_score
        
    except Exception as e:
        print(f"Readability error: {str(e)}")
        traceback.print_exc()
        # Default to a difficult score
        return 25.0

def get_vader_sentiment(text):
    try:
        # Take first 5000 characters if text is too long
        sample_text = text[:5000]
        scores = sia.polarity_scores(sample_text)
        compound_score = scores['compound']
        
        # Add interpretation for debugging
        if compound_score >= 0.8:
            interpretation = "Extremely positive"
        elif compound_score >= 0.3:
            interpretation = "Positive"
        elif compound_score <= -0.8:
            interpretation = "Extremely negative"
        elif compound_score <= -0.3:
            interpretation = "Negative"
        else:
            interpretation = "Neutral"
            
        print(f"VADER scores: {scores}")
        print(f"Compound score: {compound_score} - {interpretation}")
        return compound_score
    except Exception as e:
        print(f"Error calculating VADER sentiment: {str(e)}")
        return 0.0  # Return neutral score on error

def get_lexical_diversity(text):
    try:
        # Tokenize and clean words
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]  # Keep only alphanumeric words
        
        if not words:
            return 0.5  # Return neutral score for empty text
            
        # Calculate diversity
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words
        
        # Add interpretation for debugging
        if diversity >= 0.6:
            interpretation = "High (well-written)"
        elif diversity >= 0.4:
            interpretation = "Medium (standard)"
        else:
            interpretation = "Low (repetitive)"
            
        print(f"Lexical diversity metrics:")
        print(f"Unique words: {unique_words}")
        print(f"Total words: {total_words}")
        print(f"Diversity score: {diversity:.3f} - {interpretation}")
        
        return diversity
    except Exception as e:
        print(f"Error calculating lexical diversity: {str(e)}")
        return 0.5  # Return neutral score on error

def extract_features(text):
    """Extract all features used in training."""
    try:
        # Calculate features
        readability = get_readability(text)
        vader = get_vader_sentiment(text)
        lexical = get_lexical_diversity(text)
        
        print(f"Debug - Feature extraction results:")
        print(f"Readability score: {readability}")
        print(f"VADER sentiment: {vader}")
        print(f"Lexical diversity: {lexical}")
        
        # Create input DataFrame with features - ensure readability is not zero
        readability = max(1.0, readability)  # Force minimum of 1.0
        
        features = pd.DataFrame([{
            'text': text,  # For TF-IDF
            'readability': float(readability),  # Ensure it's a float and not zero
            'vader_sentiment': vader,
            'lexical_diversity': lexical
        }])
        
        # Additional debug to verify DataFrame contents
        print("\nDebug - Features DataFrame:")
        print(features[['readability', 'vader_sentiment', 'lexical_diversity']])
        
        return features
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        traceback.print_exc()
        return None

def load_model():
    """Load the trained model or download it if not present."""
    global model
    try:
        model_path = Config.MODEL_PATH
        model_url = "https://drive.google.com/uc?export=download&id=1kWMOE9yOPQlY_uoXhv8jvf4K1iifhViN"

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Downloading from Google Drive...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Download the model
            response = requests.get(model_url)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    f.write(response.content)
                print("Model downloaded successfully.")
            else:
                print("Failed to download model:", response.status_code)
                return False

        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def extract_article_content(url):
    """Extract content from article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Get both title and text
        content = []
        if article.title:
            content.append(article.title)
        if article.text:
            content.append(article.text)
            
        full_text = ' '.join(content)
        
        # Basic cleaning
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        # Ensure proper sentence spacing
        cleaned_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned_text)
        # Remove any non-printable characters
        cleaned_text = ''.join(char for char in cleaned_text if char.isprintable())
        
        print(f"Extracted text length: {len(cleaned_text)} characters")
        
        if not cleaned_text:
            print("Warning: No text content extracted from URL")
            return None
            
        return cleaned_text
    except Exception as e:
        print(f"Error extracting article content: {str(e)}")
        traceback.print_exc()
        return None

def generate_explanation(features, confidence):
    """Generate explanation based on features and confidence score."""
    explanations = []
    
    # Get readability interpretation if available
    readability_info = getattr(get_readability, 'last_interpretation', None)
    
    # Readability analysis
    if readability_info:
        if readability_info['score'] < 30:
            explanations.append(f"The article has poor readability ({readability_info['level']}), which can be a sign of misleading content.")
        elif readability_info['score'] > 70:
            explanations.append(f"The article has good readability ({readability_info['level']}), typical of professional journalism.")
    else:
        if features['readability'] < 30:
            explanations.append("The article has poor readability, which can be a sign of misleading content.")
        elif features['readability'] > 70:
            explanations.append("The article has good readability, typical of professional journalism.")
    
    # Sentiment analysis
    if abs(features['vader_sentiment']) > 0.5:
        if features['vader_sentiment'] > 0.5:
            explanations.append("The article shows strong positive sentiment, which may indicate bias.")
        else:
            explanations.append("The article shows strong negative sentiment, which may indicate emotional manipulation.")
    elif abs(features['vader_sentiment']) < 0.2:
        explanations.append("The article shows neutral sentiment, typical of objective reporting.")
    
    # Lexical diversity analysis
    if features['lexical_diversity'] < 0.4:
        explanations.append("Low lexical diversity might indicate repetitive or low-quality content.")
    elif features['lexical_diversity'] > 0.6:
        explanations.append("High lexical diversity suggests well-written, varied content.")
    
    # If no specific explanations, use confidence-based explanation
    if not explanations:
        if confidence > 0.7:
            explanations.append("The model detected strong patterns associated with legitimate news articles.")
        elif confidence < 0.3:
            explanations.append("The model detected patterns commonly found in misleading content.")
        else:
            explanations.append("The model found mixed patterns, making classification uncertain.")
    
    return explanations

def is_valid_url(url):
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except:
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type != 'application/json':
            return jsonify({'error': 'Invalid content type'}), 400
        # Check if model is loaded
        if model is None:
            if not load_model():
                return jsonify({
                    'error': 'Model not available. Please ensure the model file exists in the model directory.'
                }), 503
        data = request.get_json()
        url = data.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        if not is_valid_url(url):
            return jsonify({'error': 'Invalid URL. Please enter a valid http or https URL.'}), 400
        content = extract_article_content(url)
        if not content:
            return jsonify({'error': 'Could not extract article content'}), 400
        processed_content = clean_text(content)
        features = extract_features(processed_content)
        if features is None:
            return jsonify({'error': 'Feature extraction failed'}), 500
        try:
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0].tolist() if hasattr(model, 'predict_proba') else None
            print(f"Debug - Prediction results:")
            print(f"Raw prediction: {prediction}")
            print(f"Probabilities: {proba}")
            feature_dict = features.to_dict(orient='records')[0]
            confidence = proba[1] if proba else 0.5
            readability_info = getattr(get_readability, 'last_interpretation', None)
            readability_score = feature_dict.get('readability', 0)
            explanations = generate_explanation(feature_dict, confidence)

            # Save prediction to DB and get prediction_id
            from utils.database import save_prediction
            prediction_id = save_prediction(
                url,
                int(prediction),
                confidence,
                {
                    'readability': readability_score,
                    'vader_sentiment': feature_dict['vader_sentiment'],
                    'lexical_diversity': feature_dict['lexical_diversity']
                }
            )

            return jsonify({
                'prediction': int(prediction),
                'confidence': confidence,
                'features': {
                    'readability': readability_score,
                    'vader_sentiment': feature_dict['vader_sentiment'],
                    'lexical_diversity': feature_dict['lexical_diversity']
                },
                'readability_info': readability_info,
                'explanations': explanations,
                'prediction_id': prediction_id
            })
        except Exception as e:
            logging.error("Internal server error during prediction", exc_info=True)
            return jsonify({'error': 'Internal server error. Please try again later.'}), 500
    except Exception as e:
        logging.error("Internal server error in predict route", exc_info=True)
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get statistics about predictions and feedback."""
    try:
        stats = get_statistics()
        if stats is None:
            return jsonify({'error': 'Failed to retrieve statistics'}), 500
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recent', methods=['GET'])
def get_recent():
    """Get recent analyses with feedback."""
    try:
        limit = request.args.get('limit', default=10, type=int)
        analyses = get_recent_analyses(limit)
        return jsonify(analyses)
    except Exception as e:
        print(f"Error getting recent analyses: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Save user feedback for a prediction."""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        agreed = data.get('agreed')
        if prediction_id is None or agreed is None:
            return jsonify({'error': 'prediction_id and agreed are required'}), 400
        save_feedback(prediction_id, agreed)
        return jsonify({'status': 'success', 'message': 'Feedback recorded'})
    except Exception as e:
        logging.error("Internal server error in feedback route", exc_info=True)
        return jsonify({'error': 'Internal server error. Please try again later.'}), 500

@app.route('/info')
def info():
    """Render the info page with model explanation."""
    # Get detailed analysis params if present
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    readability = request.args.get('readability')
    sentiment = request.args.get('sentiment')
    diversity = request.args.get('diversity')
    analyzed_url = request.args.get('analyzed_url')
    
    # Format parameters for display
    analysis_data = None
    if prediction is not None:
        is_real = int(prediction) == 1
        status = "Likely Real" if is_real else "Likely Fake"
        
        # Get feature interpretations
        readability_level = "Unknown"
        if readability:
            read_val = float(readability)
            if read_val >= 90:
                readability_level = "Very easy (5th grade)"
            elif read_val >= 60:
                readability_level = "Standard (8-9th grade)"
            elif read_val >= 30:
                readability_level = "Difficult (college)"
            else:
                readability_level = "Very difficult (academic)"
        
        sentiment_level = "Unknown"
        if sentiment:
            sent_val = float(sentiment)
            if sent_val >= 0.8:
                sentiment_level = "Extremely positive"
            elif sent_val >= 0.3:
                sentiment_level = "Positive"
            elif sent_val <= -0.8:
                sentiment_level = "Extremely negative"
            elif sent_val <= -0.3:
                sentiment_level = "Negative"
            else:
                sentiment_level = "Neutral"
        
        diversity_level = "Unknown"
        if diversity:
            div_val = float(diversity)
            if div_val >= 0.6:
                diversity_level = "High (well-written)"
            elif div_val >= 0.4:
                diversity_level = "Medium (standard)"
            else:
                diversity_level = "Low (repetitive)"
        
        # Create analysis data dictionary
        analysis_data = {
            'prediction': status,
            'confidence': confidence,
            'readability': {
                'value': readability,
                'interpretation': readability_level
            },
            'sentiment': {
                'value': sentiment,
                'interpretation': sentiment_level
            },
            'diversity': {
                'value': diversity,
                'interpretation': diversity_level
            },
            'url': analyzed_url
        }
    
    return render_template('info.html', analysis=analysis_data)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status."""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/extension/images/<path:filename>')
def extension_images(filename):
    return send_from_directory(os.path.join(app.root_path, '..', 'extension', 'images'), filename)

if __name__ == '__main__':
    # Try to load the model on startup
    load_model()
    app.run(debug=False) 