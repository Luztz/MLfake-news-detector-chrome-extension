import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import os
import sys

# Add the server directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.config import Config

def get_db_connection() -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize the database with required tables."""
    conn = get_db_connection()
    c = conn.cursor()

    # Create predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            features TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            agreed BOOLEAN NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')

    conn.commit()
    conn.close()

def save_prediction(url, prediction, confidence, features):
    """Save a prediction to the database."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        c.execute('''
            INSERT INTO predictions (url, prediction, confidence, features)
            VALUES (?, ?, ?, ?)
        ''', (url, prediction, confidence, json.dumps(features)))
        
        prediction_id = c.lastrowid
        conn.commit()
        return prediction_id
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        return None
    finally:
        conn.close()

def save_feedback(prediction_id, agreed):
    """Save user feedback to the database."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        c.execute('''
            INSERT INTO feedback (prediction_id, agreed)
            VALUES (?, ?)
        ''', (prediction_id, agreed))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False
    finally:
        conn.close()

def get_statistics():
    """Get statistics about predictions."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        # Get total analyses
        c.execute('SELECT COUNT(*) as total FROM predictions')
        total_analyses = c.fetchone()['total']
        
        # Get feature averages
        c.execute('''
            SELECT 
                AVG(json_extract(features, '$.readability')) as avg_readability,
                AVG(json_extract(features, '$.vader_sentiment')) as avg_sentiment,
                AVG(json_extract(features, '$.lexical_diversity')) as avg_lexical
            FROM predictions
        ''')
        feature_averages = c.fetchone()
        
        # Get confidence distribution
        c.execute('''
            SELECT 
                SUM(CASE WHEN confidence <= 25 THEN 1 ELSE 0 END) as range_0_25,
                SUM(CASE WHEN confidence > 25 AND confidence <= 50 THEN 1 ELSE 0 END) as range_26_50,
                SUM(CASE WHEN confidence > 50 AND confidence <= 75 THEN 1 ELSE 0 END) as range_51_75,
                SUM(CASE WHEN confidence > 75 THEN 1 ELSE 0 END) as range_76_100
            FROM predictions
        ''')
        confidence_distribution = c.fetchone()
        
        return {
            'total_analyses': total_analyses,
            'feature_averages': {
                'readability': round(feature_averages['avg_readability'], 2),
                'vader_sentiment': round(feature_averages['avg_sentiment'], 2),
                'lexical_diversity': round(feature_averages['avg_lexical'], 2)
            },
            'confidence_distribution': {
                '0-25': confidence_distribution['range_0_25'],
                '26-50': confidence_distribution['range_26_50'],
                '51-75': confidence_distribution['range_51_75'],
                '76-100': confidence_distribution['range_76_100']
            }
        }
    except Exception as e:
        print(f"Error getting prediction stats: {str(e)}")
        return None
    finally:
        conn.close()

def get_recent_analyses(limit=10):
    """Get recent analyses."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        c.execute('''
            SELECT p.*, f.agreed
            FROM predictions p
            LEFT JOIN feedback f ON p.id = f.prediction_id
            ORDER BY p.created_at DESC
            LIMIT ?
        ''', (limit,))
        
        analyses = []
        for row in c.fetchall():
            analyses.append({
                'url': row['url'],
                'prediction': row['prediction'],
                'confidence': row['confidence'],
                'features': json.loads(row['features']),
                'created_at': row['created_at'],
                'feedback': row['agreed']
            })
        
        return analyses
    finally:
        conn.close()

def get_prediction_history(
    limit: int = 10,
    offset: int = 0
) -> List[Dict[str, Union[str, float, datetime]]]:
    """
    Get recent prediction history.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        List of prediction records
    """
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute('''
            SELECT p.*, 
                   COUNT(f.id) as feedback_count,
                   SUM(CASE WHEN f.helpful THEN 1 ELSE 0 END) as helpful_count
            FROM predictions p
            LEFT JOIN feedback f ON p.id = f.prediction_id
            GROUP BY p.id
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        return [dict(row) for row in c.fetchall()]
    finally:
        conn.close()

def get_prediction_stats():
    """Get statistics about predictions."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        # Get total predictions
        c.execute('SELECT COUNT(*) as count FROM predictions')
        total_predictions = c.fetchone()['count']

        # Get prediction distribution
        c.execute('''
            SELECT prediction, COUNT(*) as count
            FROM predictions
            GROUP BY prediction
        ''')
        prediction_distribution = dict(c.fetchall())

        # Get average confidence
        c.execute('SELECT AVG(confidence) as avg_confidence FROM predictions')
        avg_confidence = c.fetchone()['avg_confidence']

        return {
            'total_predictions': total_predictions,
            'prediction_distribution': prediction_distribution,
            'average_confidence': avg_confidence
        }
    except Exception as e:
        print(f"Error getting prediction stats: {str(e)}")
        return None
    finally:
        conn.close()

def get_feedback_stats():
    """Get statistics about user feedback."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        # Get total feedback
        c.execute('SELECT COUNT(*) as count FROM feedback')
        total_feedback = c.fetchone()['count']

        # Get feedback distribution
        c.execute('''
            SELECT feedback_type, COUNT(*) as count
            FROM feedback
            GROUP BY feedback_type
        ''')
        feedback_distribution = dict(c.fetchall())

        return {
            'total_feedback': total_feedback,
            'feedback_distribution': feedback_distribution
        }
    except Exception as e:
        print(f"Error getting feedback stats: {str(e)}")
        return None
    finally:
        conn.close() 