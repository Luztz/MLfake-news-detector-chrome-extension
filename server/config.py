import os

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change-in-production'
    DEBUG = False

    # Database settings
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server/database/feedback.db')
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model/fake_news_rf_pipeline.pkl')
    
    # API settings
    CORS_ORIGINS = [
        'chrome-extension://*',  # Allow all Chrome extensions
        'http://localhost:*',    # Allow local development
    ]
    
    # Content settings
    MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'html'}
    
    # Security settings
    CSRF_ENABLED = True
    SSL_ENABLED = False  # Set to True in production 