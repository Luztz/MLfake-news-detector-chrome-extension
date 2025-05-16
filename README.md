# Fake News Detector Chrome Extension

A Chrome extension that uses machine learning to detect potentially misleading news articles.

## Features

- Real-time analysis of news articles
- Confidence score with detailed explanation
- Feature breakdown (sentiment, readability, word count, VADER sentiment)
- User feedback system
- Detailed model information page

## Installation

### Chrome Extension
1. Clone this repository
2. Open Chrome and go to `chrome://extensions/`
3. Enable "Developer mode"
4. Click "Load unpacked" and select the `extension` folder

### Server Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python server/app.py
```

## Development

The project consists of two main components:

1. **Chrome Extension** (`extension/`)
   - Popup interface for URL input
   - Result display with confidence score
   - Feature breakdown visualization
   - Feedback system

2. **Backend Server** (`server/`)
   - Flask API endpoints
   - Machine learning model integration
   - Database for predictions and feedback
   - Information page with model details

## API Endpoints

- `POST /predict`: Analyze a news article URL
- `POST /feedback`: Submit user feedback
- `GET /info`: Get model information page
- `GET /health`: Server health check

## Model Features

- Sentiment Analysis
- Readability Score
- Word Count
- VADER Sentiment Analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TextBlob for sentiment analysis
- VADER for social media sentiment analysis
- Flask for the backend framework
- Chrome Extension API
