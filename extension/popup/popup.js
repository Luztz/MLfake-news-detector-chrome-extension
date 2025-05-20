// Remove unused changeColor logic
// Remove all old/duplicate rendering logic for results, explanation, and features
// Only use the new modern card UI rendering
// Ensure results are cleared before rendering
// Ensure backBtn is always defined in the correct scope

// Constants
const API_URL = 'http://127.0.0.1:5000';
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
    readabilityValue: document.getElementById('readability-value'),
    vaderValue: document.getElementById('vader-value'),
    lexicalValue: document.getElementById('lexical-value')
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
    const isFake = data.prediction === 'fake';
    const confidence = Math.round(data.confidence * 100);
    const level = getResultLevel(data.confidence, isFake);
    DOM_ELEMENTS.resultIcon.className = `result-icon ${isFake ? 'fake' : 'real'} fas ${level.icon}`;
    DOM_ELEMENTS.resultTitle.textContent = level.text;
    DOM_ELEMENTS.confidenceValue.textContent = `${confidence}%`;
    DOM_ELEMENTS.confidenceBarFill.style.width = `${confidence}%`;
    
    // Update features display
    DOM_ELEMENTS.readabilityValue.textContent = formatReadabilityScore(data.features.readability);
    DOM_ELEMENTS.vaderValue.textContent = formatVaderSentiment(data.features.vader_sentiment);
    DOM_ELEMENTS.lexicalValue.textContent = formatLexicalDiversity(data.features.lexical_diversity);
    
    // Update reasons
    DOM_ELEMENTS.reasonsList.innerHTML = data.reasons.map(reason => `<li>${sanitizeText(reason)}</li>`).join('');
};

// Helper functions for formatting feature values
const formatReadabilityScore = (score) => {
    if (score === undefined) return '-';
    const rounded = Math.round(score);
    return `${rounded} (${getReadabilityLevel(rounded)})`;
};

const formatVaderSentiment = (score) => {
    if (score === undefined) return '-';
    const rounded = score.toFixed(2);
    return `${rounded} (${getSentimentLevel(score)})`;
};

const formatLexicalDiversity = (score) => {
    if (score === undefined) return '-';
    return (score * 100).toFixed(1) + '%';
};

const getReadabilityLevel = (score) => {
    if (score >= 90) return 'Very Easy';
    if (score >= 80) return 'Easy';
    if (score >= 70) return 'Fairly Easy';
    if (score >= 60) return 'Standard';
    if (score >= 50) return 'Fairly Difficult';
    if (score >= 30) return 'Difficult';
    return 'Very Difficult';
};

const getSentimentLevel = (score) => {
    if (score >= 0.5) return 'Very Positive';
    if (score >= 0.1) return 'Positive';
    if (score > -0.1) return 'Neutral';
    if (score > -0.5) return 'Negative';
    return 'Very Negative';
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
    // Autofill current tab's URL
    if (chrome.tabs) {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs && tabs[0] && tabs[0].url) {
                DOM_ELEMENTS.urlInput.value = tabs[0].url;
            }
        });
    }
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

    // Learn More button
    const learnMoreBtn = document.getElementById('learn-more-btn');
    if (learnMoreBtn) {
        learnMoreBtn.addEventListener('click', () => {
            window.open('https://your-learn-more-page.com', '_blank');
        });
    }
});

// When back is pressed, remove the back button
backBtn && backBtn.remove();

// Add light/dark mode toggle logic
const modeToggle = document.getElementById('mode-toggle');
const body = document.body;

function setMode(mode) {
    if (mode === 'dark') {
        body.classList.add('dark-mode');
        modeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        body.classList.remove('dark-mode');
        modeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    }
    localStorage.setItem('theme', mode);
}

modeToggle.addEventListener('click', () => {
    const isDark = body.classList.contains('dark-mode');
    setMode(isDark ? 'light' : 'dark');
});

// On load, set theme from localStorage
const savedTheme = localStorage.getItem('theme') || 'light';
setMode(savedTheme);

// Map confidence to 4-level result header
function getResultLevel(confidence, isFake) {
    // confidence: 0-1, isFake: boolean
    const percent = Math.round(confidence * 100);
    if (isFake) {
        if (percent < 25) return {icon: 'fa-skull-crossbones', text: '🚫 Very Likely Fake News', color: 'danger'};
        if (percent < 50) return {icon: 'fa-exclamation-triangle', text: '⚠️ Likely Fake News', color: 'warning'};
        return {icon: 'fa-question-circle', text: '⚠️ Uncertain', color: 'warning'};
    } else {
        if (percent < 75) return {icon: 'fa-info-circle', text: '⚠️ Likely Real News', color: 'info'};
        return {icon: 'fa-check-circle', text: '✅ Very Likely Real News', color: 'success'};
    }
}

// Improved explanations based on features
function generateExplanations(prediction, features) {
    const reasons = [];
    if (prediction === 'fake') {
        if (features.readability < 30) reasons.push('The article is hard to read, which is sometimes used to obscure misleading information.');
        if (features.vader_sentiment < -0.5) reasons.push('The VADER sentiment is strongly negative, indicating a misleading or emotional tone.');
        if (features.sentiment < -0.3) reasons.push('The overall sentiment is negative, which can be a sign of bias or manipulation.');
        if (features.word_count > 2000) reasons.push('The article is unusually long, which can be a tactic to overwhelm readers.');
        if (features.lexical_diversity < 0.3) reasons.push('Low lexical diversity may indicate repetitive or low-quality content.');
        if (reasons.length === 0) reasons.push('The model detected patterns commonly found in misleading articles.');
    } else {
        if (features.readability > 50) reasons.push('The article is easy to read, suggesting clear and straightforward information.');
        if (features.vader_sentiment > 0.2) reasons.push('The VADER sentiment is positive, indicating a balanced or neutral tone.');
        if (features.sentiment > 0.1) reasons.push('The overall sentiment is positive, typical of factual reporting.');
        if (features.word_count >= 500 && features.word_count <= 1500) reasons.push('The article length is appropriate for comprehensive coverage.');
        if (features.lexical_diversity > 0.5) reasons.push('High lexical diversity suggests a well-written and informative article.');
        if (reasons.length === 0) reasons.push('The model detected patterns commonly found in factual articles.');
    }
    return reasons;
}