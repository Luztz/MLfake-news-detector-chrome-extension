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
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            features TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create feedback table
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
        
        conn.commit()
        return c.lastrowid
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        return None
    finally:
        conn.close()

def save_feedback(url, feedback_type, comment=''):
    """Save user feedback to the database."""
    conn = get_db_connection()
    c = conn.cursor()

    try:
        c.execute('''
            INSERT INTO feedback (url, feedback_type, comment)
            VALUES (?, ?, ?)
        ''', (url, feedback_type, comment))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False
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
            ORDER BY p.timestamp DESC
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