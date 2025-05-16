from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import os
import shap
from datetime import datetime
import sys
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
from sklearn.pipeline import Pipeline
from model import (
    TextSelector,
    SentimentSelector,
    ReadabilitySelector,
    WordCountSelector,
    VaderSentimentSelector
)

# Add the server directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the model directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from server.utils.text_processor import clean_text, extract_features, get_key_phrases
from server.utils.database import init_db, save_prediction, save_feedback
from server.config import Config

app = Flask(__name__)
CORS(app)

# Initialize database
init_db()

# Global variable for model
model = None

def load_model():
    """Load the trained model or download it if not present."""
    global model
    try:
        model_path = Config.MODEL_PATH
        model_url = "https://drive.google.com/uc?export=download&id=1b1BCx3rFpC3OFZ69uuk2o-Ubdq8VMFDr"

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
        return article.text
    except Exception as e:
        print(f"Error extracting article content: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        
        # Extract article content
        content = extract_article_content(url)
        if not content:
            return jsonify({'error': 'Could not extract article content'}), 400
        
        # Preprocess the content
        processed_content = clean_text(content)
        
        try:
            # Create a DataFrame with the required structure for the pipeline
            input_data = pd.DataFrame({
                'text': [processed_content],
                'sentiment': [0],  # Placeholder values
                'readability': [0],
                'word_count': [0],
                'vader_sentiment': [0]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = max(probabilities)
            
            # Extract features
            features = extract_features(content)
            
            # Generate reasons for the prediction
            reasons = []
            if prediction == 0:  # Fake news
                if features['sentiment'] < -0.3:
                    reasons.append("The article has a strongly negative sentiment, which is often used in misleading content.")
                if features['readability'] < 30:
                    reasons.append("The article is hard to read, which is sometimes used to obscure misleading information.")
                if features['vader_sentiment'] < -0.5:
                    reasons.append("The VADER sentiment is negative, indicating a misleading tone.")
                if features['word_count'] > 2000:
                    reasons.append("The article is unusually long, which can be a tactic to overwhelm readers with information.")
            else:  # Real news
                if -0.1 <= features['sentiment'] <= 0.1:
                    reasons.append("The article maintains a neutral tone, typical of factual reporting.")
                if features['readability'] > 50:
                    reasons.append("The article is easy to read, suggesting clear and straightforward information.")
                if -0.2 <= features['vader_sentiment'] <= 0.2:
                    reasons.append("The VADER sentiment is balanced, indicating objective reporting.")
                if 500 <= features['word_count'] <= 1500:
                    reasons.append("The article length is appropriate for comprehensive coverage.")
            
            # Store prediction in database
            prediction_id = save_prediction(
                url=url,
                prediction='Real' if prediction == 1 else 'Fake',
                confidence=float(confidence),
                features=features
            )
            
            return jsonify({
                'prediction': 'fake' if prediction == 0 else 'real',
                'confidence': float(confidence),
                'features': features,
                'reasons': reasons,
                'prediction_id': prediction_id
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Error making prediction: {str(e)}'
            }), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        url = data.get('url')
        feedback_type = data.get('feedback')
        comment = data.get('comment', '')
        
        if not url or not feedback_type:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Save feedback
        if save_feedback(url, feedback_type, comment):
            return jsonify({'message': 'Feedback recorded successfully'})
        else:
            return jsonify({'error': 'Failed to save feedback'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info')
def info():
    """Render the info page with model explanation."""
    return render_template('info.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status."""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

if __name__ == '__main__':
    # Try to load the model on startup
    load_model()
    app.run(debug=True) 