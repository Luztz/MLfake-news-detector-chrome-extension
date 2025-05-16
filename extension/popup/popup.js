// Remove unused changeColor logic
// Remove all old/duplicate rendering logic for results, explanation, and features
// Only use the new modern card UI rendering
// Ensure results are cleared before rendering
// Ensure backBtn is always defined in the correct scope

// Constants
const API_URL = 'https://fake-news-api.onrender.com';
const DOM_ELEMENTS = {
    urlInput: document.getElementById('url-input'),
    analyzeBtn: document.getElementById('analyze-btn'),
    resultsSection: document.getElementById('results'),
    defaultMessage: document.getElementById('default-message'),
    expandBtn: document.getElementById('expand-btn'),
    thumbsUpBtn: document.getElementById('thumbs-up'),
    thumbsDownBtn: document.getElementById('thumbs-down'),
    feedbackModal: document.getElementById('feedback-modal'),
    closeModal: document.querySelector('.close'),
    feedbackText: document.getElementById('feedback-text'),
    submitFeedback: document.getElementById('submit-feedback'),
    resultIcon: document.getElementById('result-icon'),
    resultTitle: document.getElementById('result-title'),
    confidenceValue: document.getElementById('confidence-value'),
    confidenceBarFill: document.getElementById('confidence-bar-fill'),
    reasonsList: document.getElementById('result-reasons'),
    sentimentValue: document.getElementById('sentiment-value'),
    readabilityValue: document.getElementById('readability-value'),
    wordcountValue: document.getElementById('wordcount-value'),
    vaderValue: document.getElementById('vader-value')
};

// State management
let selectedFeedback = null;
let isAnalyzing = false;

// Utility functions
const showError = (message) => {
    alert(message);
    console.error(message);
};

const resetFeedbackState = () => {
    selectedFeedback = null;
    DOM_ELEMENTS.feedbackText.value = '';
    DOM_ELEMENTS.thumbsUpBtn.classList.remove('active');
    DOM_ELEMENTS.thumbsDownBtn.classList.remove('active');
    DOM_ELEMENTS.feedbackModal.style.display = 'none';
};

const updateButtonState = (isLoading) => {
    isAnalyzing = isLoading;
    DOM_ELEMENTS.analyzeBtn.disabled = isLoading;
    DOM_ELEMENTS.analyzeBtn.innerHTML = isLoading 
        ? '<i class="fas fa-spinner fa-spin"></i> Analyzing...'
        : '<i class="fas fa-search"></i> Analyze';
};

const sanitizeText = (text) => {
    return text.replace(/[\uFFFD\u2028\u2029\u00A0\u200B\u200C\u200D\uFEFF]/g, '')
               .replace(/[\x00-\x1F\x7F-\x9F]/g, '');
};

// API functions
const analyzeUrl = async (url) => {
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: sanitizeText(url) })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        throw new Error(`Analysis failed: ${error.message}`);
    }
};

const submitFeedback = async (url, feedback, comment) => {
    try {
        const response = await fetch(`${API_URL}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                url: sanitizeText(url),
                feedback,
                comment: sanitizeText(comment)
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        throw new Error(`Feedback submission failed: ${error.message}`);
    }
};

// UI update functions
const updateResultDisplay = (data) => {
    // Update result icon and title
    const isFake = data.prediction === 'fake';
    DOM_ELEMENTS.resultIcon.className = `result-icon ${isFake ? 'fake' : 'real'} fas fa-${isFake ? 'exclamation-triangle' : 'check-circle'}`;
    DOM_ELEMENTS.resultTitle.textContent = isFake ? '🚫 Likely Fake News' : '✅ Likely Real News';

    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    DOM_ELEMENTS.confidenceValue.textContent = `${confidence}%`;
    DOM_ELEMENTS.confidenceBarFill.style.width = `${confidence}%`;

    // Update reasons
    DOM_ELEMENTS.reasonsList.innerHTML = data.reasons
        .map(reason => `<li>${sanitizeText(reason)}</li>`)
        .join('');

    // Update features
    DOM_ELEMENTS.sentimentValue.textContent = data.features.sentiment.toFixed(2);
    DOM_ELEMENTS.readabilityValue.textContent = data.features.readability.toFixed(2);
    DOM_ELEMENTS.wordcountValue.textContent = data.features.word_count;
    DOM_ELEMENTS.vaderValue.textContent = data.features.vader_sentiment.toFixed(2);
};

// Event handlers
const handleAnalyze = async () => {
    const url = DOM_ELEMENTS.urlInput.value.trim();
    if (!url) {
        showError('Please enter a URL to analyze');
        return;
    }

    if (isAnalyzing) return;

    try {
        updateButtonState(true);
        const data = await analyzeUrl(url);
        
        DOM_ELEMENTS.defaultMessage.style.display = 'none';
        DOM_ELEMENTS.resultsSection.style.display = 'block';
        updateResultDisplay(data);
    } catch (error) {
        showError(error.message);
    } finally {
        updateButtonState(false);
    }
};

const handleFeedback = async () => {
    if (!selectedFeedback) return;

    try {
        await submitFeedback(
            DOM_ELEMENTS.urlInput.value,
            selectedFeedback,
            DOM_ELEMENTS.feedbackText.value
        );
        
        alert('Thank you for your feedback!');
        resetFeedbackState();
    } catch (error) {
        showError(error.message);
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Analyze button click
    DOM_ELEMENTS.analyzeBtn.addEventListener('click', handleAnalyze);

    // URL input enter key
    DOM_ELEMENTS.urlInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isAnalyzing) {
            handleAnalyze();
        }
    });

    // Expand explanation button
    DOM_ELEMENTS.expandBtn.addEventListener('click', () => {
        window.open(`${API_URL}/info`, '_blank');
    });

    // Feedback buttons
    DOM_ELEMENTS.thumbsUpBtn.addEventListener('click', () => {
        selectedFeedback = 'positive';
        DOM_ELEMENTS.thumbsUpBtn.classList.add('active');
        DOM_ELEMENTS.thumbsDownBtn.classList.remove('active');
        DOM_ELEMENTS.feedbackModal.style.display = 'block';
    });

    DOM_ELEMENTS.thumbsDownBtn.addEventListener('click', () => {
        selectedFeedback = 'negative';
        DOM_ELEMENTS.thumbsDownBtn.classList.add('active');
        DOM_ELEMENTS.thumbsUpBtn.classList.remove('active');
        DOM_ELEMENTS.feedbackModal.style.display = 'block';
    });

    // Modal close button
    DOM_ELEMENTS.closeModal.addEventListener('click', resetFeedbackState);

    // Modal outside click
    window.addEventListener('click', (event) => {
        if (event.target === DOM_ELEMENTS.feedbackModal) {
            resetFeedbackState();
        }
    });

    // Submit feedback
    DOM_ELEMENTS.submitFeedback.addEventListener('click', handleFeedback);
});

// When back is pressed, remove the back button
backBtn && backBtn.remove();