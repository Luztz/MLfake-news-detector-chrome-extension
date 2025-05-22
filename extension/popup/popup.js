// Remove unused changeColor logic
// Remove all old/duplicate rendering logic for results, explanation, and features
// Only use the new modern card UI rendering
// Ensure results are cleared before rendering
// Ensure backBtn is always defined in the correct scope

// Constants
const API_URL = 'http://127.0.0.1:5000';
const DOM_ELEMENTS = {
    urlInput: document.getElementById('urlInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingState: document.getElementById('loadingState'),
    confidenceLabel: document.getElementById('confidenceLabel'),
    confidenceIcon: document.getElementById('confidenceIcon'),
    confidenceText: document.getElementById('confidenceText'),
    confidencePercent: document.getElementById('confidencePercent'),
    confidenceBarFill: document.getElementById('confidenceBarFill'),
    confidenceMarker: document.getElementById('confidenceMarker'),
    explanationsList: document.getElementById('explanationsList'),
    readabilityScore: document.getElementById('readabilityScore'),
    vaderSentiment: document.getElementById('vaderSentiment'),
    lexicalDiversity: document.getElementById('lexicalDiversity'),
    learnMoreBtn: document.getElementById('learnMoreBtn'),
    agreeBtn: document.getElementById('agreeBtn'),
    disagreeBtn: document.getElementById('disagreeBtn'),
    feedbackMessage: document.getElementById('feedbackMessage'),
    currentPredictionId: document.getElementById('currentPredictionId'),
    themeToggle: document.getElementById('themeToggle'),
    themeIcon: document.getElementById('themeIcon'),
    submitFeedback: document.getElementById('submitFeedback')
};

// State management
let selectedFeedback = null;
let isAnalyzing = false;
let lastAnalysisData = null; // Store the latest analysis data

// Utility functions
const showError = (message) => {
    if (message) {
        console.error('Error:', message);
        alert(message);
    }
};

const resetFeedbackState = () => {
    selectedFeedback = null;
    if (DOM_ELEMENTS.confidenceText) {
        DOM_ELEMENTS.confidenceText.textContent = '';
        DOM_ELEMENTS.confidenceIcon.textContent = '';
        DOM_ELEMENTS.confidenceLabel.className = '';
        DOM_ELEMENTS.confidencePercent.textContent = '';
        DOM_ELEMENTS.confidenceBarFill.style.width = '0%';
        DOM_ELEMENTS.confidenceMarker.style.left = '0%';
        DOM_ELEMENTS.explanationsList.innerHTML = '';
        DOM_ELEMENTS.readabilityScore.textContent = '';
        DOM_ELEMENTS.vaderSentiment.textContent = '';
        DOM_ELEMENTS.lexicalDiversity.textContent = '';
        DOM_ELEMENTS.resultsSection.style.display = 'none';
        DOM_ELEMENTS.loadingState.style.display = 'none';
    }
};

const updateButtonState = (isLoading) => {
    console.log('updateButtonState called, isLoading:', isLoading);
    isAnalyzing = isLoading;
    if (DOM_ELEMENTS.analyzeBtn) {
        DOM_ELEMENTS.analyzeBtn.disabled = isLoading;
        DOM_ELEMENTS.loadingState.style.display = isLoading ? 'block' : 'none';
        DOM_ELEMENTS.analyzeBtn.innerHTML = isLoading 
            ? '<i class="fas fa-spinner fa-spin"></i> Analyzing...'
            : '<i class="fas fa-search"></i> Analyze';
    }
};

const sanitizeText = (text) => {
    if (!text) return '';
    return text.toString().replace(/[\uFFFD\u2028\u2029\u00A0\u200B\u200C\u200D\uFEFF]/g, '')
               .replace(/[\x00-\x1F\x7F-\x9F]/g, '');
};

// API functions
const analyzeUrl = async (url) => {
    console.log('analyzeUrl called with URL:', url);
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

// UI update functions
const updateResultDisplay = (data) => {
    console.log('Updating display with data:', data);
    if (!data || !DOM_ELEMENTS.confidencePercent) return;
    
    // Store the latest analysis data for Learn More
    lastAnalysisData = data;
    
    try {
        // Determine if it's fake or real (binary classification)
        const prediction = data.prediction || 0;
        const isReal = prediction === 1;
        
        // Update confidence display
        // For Fake News: confidence = 100 - (confidence * 100)
        // For Real News: confidence = confidence * 100
        let confidence = 0;
        if (isReal) {
            confidence = (data.confidence || 0) * 100; // Real: higher confidence = more real
        } else {
            confidence = 100 - ((data.confidence || 0) * 100); // Fake: lower original confidence = more fake
        }
        
        DOM_ELEMENTS.confidencePercent.textContent = confidence.toFixed(1);
        
        // Update confidence bar
        DOM_ELEMENTS.confidenceBarFill.style.width = `${confidence}%`;
        DOM_ELEMENTS.confidenceMarker.style.left = `${confidence}%`;
        
        // Update bar color only for fake news (red)
        if (isReal) {
            DOM_ELEMENTS.confidenceBarFill.style.backgroundColor = ''; // No background color for real news
        } else {
            DOM_ELEMENTS.confidenceBarFill.style.backgroundColor = '#F44336'; // Red for fake news
        }
        
        // Update classification text and icon based on binary classification
        if (isReal) {
            DOM_ELEMENTS.confidenceIcon.textContent = '✅';
            DOM_ELEMENTS.confidenceText.textContent = 'Likely Real News';
            DOM_ELEMENTS.confidenceLabel.className = 'confidence-label real';
        } else {
            DOM_ELEMENTS.confidenceIcon.textContent = '❌';
            DOM_ELEMENTS.confidenceText.textContent = 'Likely Fake News';
            DOM_ELEMENTS.confidenceLabel.className = 'confidence-label fake';
        }
        
        // Update explanations
        DOM_ELEMENTS.explanationsList.innerHTML = '';
        if (data.explanations && Array.isArray(data.explanations)) {
            data.explanations.forEach(explanation => {
                if (explanation) {
                    const li = document.createElement('li');
                    li.textContent = explanation;
                    DOM_ELEMENTS.explanationsList.appendChild(li);
                }
            });
        }
        
        // Update feature values with proper formatting
        const features = data.features || {};
        
        // Format readability score (0-100)
        const readability = features.readability;
        if (readability !== undefined && readability !== null) {
            // Format based on value - even show zero values
            if (readability === 0) {
                DOM_ELEMENTS.readabilityScore.textContent = '0';
            } else {
                DOM_ELEMENTS.readabilityScore.textContent = readability.toFixed(2);
            }
        } else {
            DOM_ELEMENTS.readabilityScore.textContent = '-';
        }
        
        // Format VADER sentiment (-1 to 1)
        const sentiment = features.vader_sentiment;
        if (sentiment !== undefined && sentiment !== null) {
            // Round to 2 decimal places, but show as whole number if close to -1, 0, or 1
            const roundedSentiment = Math.abs(sentiment - Math.round(sentiment)) < 0.01 
                ? Math.round(sentiment).toString()
                : sentiment.toFixed(2);
            DOM_ELEMENTS.vaderSentiment.textContent = roundedSentiment;
        } else {
            DOM_ELEMENTS.vaderSentiment.textContent = '-';
        }
        
        // Format lexical diversity (0-1)
        const diversity = features.lexical_diversity;
        if (diversity !== undefined && diversity !== null) {
            DOM_ELEMENTS.lexicalDiversity.textContent = diversity.toFixed(2);
        } else {
            DOM_ELEMENTS.lexicalDiversity.textContent = '-';
        }
        
        // Store prediction ID if available
        if (data.prediction_id) {
            DOM_ELEMENTS.currentPredictionId.value = data.prediction_id;
        }
        
        // Show results section
        DOM_ELEMENTS.resultsSection.style.display = 'block';
    } catch (error) {
        console.error('Error updating result display:', error);
        showError('Error displaying results');
    }
};

// Event handlers
const handleAnalyze = async () => {
    console.log('handleAnalyze called');
    if (!DOM_ELEMENTS.urlInput) return;
    
    const url = DOM_ELEMENTS.urlInput.value.trim();
    console.log('URL to analyze:', url);
    
    if (!url) {
        showError('Please enter a URL to analyze');
        return;
    }

    if (isAnalyzing) {
        console.log('Already analyzing, returning');
        return;
    }

    try {
        updateButtonState(true);
        const data = await analyzeUrl(url);
        console.log('Analysis data received:', data);
        
        updateResultDisplay(data);
    } catch (error) {
        console.error('Error during analysis:', error);
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
            DOM_ELEMENTS.confidenceText.value
        );
        
        alert('Thank you for your feedback!');
        resetFeedbackState();
    } catch (error) {
        showError(error.message);
    }
};

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded');
    console.log('DOM Elements:', DOM_ELEMENTS);
    
    // Theme toggle logic
    if (DOM_ELEMENTS.themeToggle && DOM_ELEMENTS.themeIcon) {
        const setTheme = (mode) => {
            if (mode === 'dark') {
                document.body.classList.add('dark-theme');
                DOM_ELEMENTS.themeIcon.textContent = '🌙';
            } else {
                document.body.classList.remove('dark-theme');
                DOM_ELEMENTS.themeIcon.textContent = '☀️';
            }
            localStorage.setItem('theme', mode);
        };

        // Set theme from localStorage or default to light
        const savedTheme = localStorage.getItem('theme') || 'light';
        setTheme(savedTheme);

        DOM_ELEMENTS.themeToggle.addEventListener('click', () => {
            const isDark = document.body.classList.contains('dark-theme');
            setTheme(isDark ? 'light' : 'dark');
        });
    }

    // Autofill current tab's URL
    if (chrome && chrome.tabs) {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs && tabs[0] && tabs[0].url && DOM_ELEMENTS.urlInput) {
                DOM_ELEMENTS.urlInput.value = tabs[0].url;
            }
        });
    }

    // Analyze button click
    if (DOM_ELEMENTS.analyzeBtn) {
        console.log('Adding click listener to analyzeBtn');
        DOM_ELEMENTS.analyzeBtn.addEventListener('click', handleAnalyze);
    }

    // URL input enter key
    if (DOM_ELEMENTS.urlInput) {
        DOM_ELEMENTS.urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !isAnalyzing) {
                handleAnalyze();
            }
        });
    }

    // Learn More button
    if (DOM_ELEMENTS.learnMoreBtn) {
        DOM_ELEMENTS.learnMoreBtn.addEventListener('click', () => {
            // Prepare detailed analysis data to pass to the info page
            let detailsUrl = `${API_URL}/info`;
            
            // If we have analysis data, add it as query parameters
            if (lastAnalysisData) {
                const params = new URLSearchParams();
                const confidence = (lastAnalysisData.confidence || 0) * 100;
                
                // Add basic analysis results
                params.append('confidence', confidence.toFixed(1));
                params.append('prediction', lastAnalysisData.prediction);
                
                // Add feature values if available
                if (lastAnalysisData.features) {
                    const features = lastAnalysisData.features;
                    if (features.readability !== undefined) params.append('readability', features.readability);
                    if (features.vader_sentiment !== undefined) params.append('sentiment', features.vader_sentiment);
                    if (features.lexical_diversity !== undefined) params.append('diversity', features.lexical_diversity);
                }
                
                // Add the URL being analyzed
                if (DOM_ELEMENTS.urlInput && DOM_ELEMENTS.urlInput.value) {
                    params.append('analyzed_url', encodeURIComponent(DOM_ELEMENTS.urlInput.value));
                }
                
                // Add theme preference
                const isDarkTheme = document.body.classList.contains('dark-theme');
                params.append('theme', isDarkTheme ? 'dark' : 'light');
                
                detailsUrl += `?${params.toString()}`;
            } else {
                // If no analysis data, at least pass the theme
                const isDarkTheme = document.body.classList.contains('dark-theme');
                detailsUrl += `?theme=${isDarkTheme ? 'dark' : 'light'}`;
            }
            
            window.open(detailsUrl, '_blank');
        });
    }

    // Feedback buttons
    if (DOM_ELEMENTS.agreeBtn) {
        DOM_ELEMENTS.agreeBtn.addEventListener('click', () => {
            selectedFeedback = 'positive';
            DOM_ELEMENTS.agreeBtn.classList.add('active');
            if (DOM_ELEMENTS.disagreeBtn) {
                DOM_ELEMENTS.disagreeBtn.classList.remove('active');
            }
            // No longer change the UI display when user provides feedback
            if (DOM_ELEMENTS.feedbackMessage) {
                DOM_ELEMENTS.feedbackMessage.textContent = 'Thank you! Your feedback helps improve our model.';
                DOM_ELEMENTS.feedbackMessage.className = 'feedback-message success';
                setTimeout(() => {
                    DOM_ELEMENTS.feedbackMessage.textContent = '';
                    DOM_ELEMENTS.feedbackMessage.className = 'feedback-message';
                }, 3000);
            }
        });
    }

    if (DOM_ELEMENTS.disagreeBtn) {
        DOM_ELEMENTS.disagreeBtn.addEventListener('click', () => {
            selectedFeedback = 'negative';
            DOM_ELEMENTS.disagreeBtn.classList.add('active');
            if (DOM_ELEMENTS.agreeBtn) {
                DOM_ELEMENTS.agreeBtn.classList.remove('active');
            }
            // No longer change the UI display when user provides feedback
            if (DOM_ELEMENTS.feedbackMessage) {
                DOM_ELEMENTS.feedbackMessage.textContent = 'Thank you! Your feedback helps improve our model.';
                DOM_ELEMENTS.feedbackMessage.className = 'feedback-message success';
                setTimeout(() => {
                    DOM_ELEMENTS.feedbackMessage.textContent = '';
                    DOM_ELEMENTS.feedbackMessage.className = 'feedback-message';
                }, 3000);
            }
        });
    }

    // Submit feedback
    DOM_ELEMENTS.submitFeedback.addEventListener('click', handleFeedback);

    // Initialize statistics
    initializeStatistics();
});

// When back is pressed, remove the back button
backBtn && backBtn.remove();

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
function generateExplanations(isFake, features) {
    const reasons = [];
    if (!isFake) {  // For real news (high confidence)
        if (features.readability > 50) reasons.push('The article has good readability, which is common in legitimate news sources.');
        if (features.sentiment > -0.2 && features.sentiment < 0.2) reasons.push('The article shows balanced sentiment, typical of objective reporting.');
        if (features.lexical_diversity > 0.5) reasons.push('High lexical diversity suggests professional and varied writing.');
        if (reasons.length === 0) reasons.push('The model detected patterns commonly found in factual articles.');
    } else {  // For fake news (low confidence)
        if (features.readability < 30) reasons.push('The article has poor readability, which can be a sign of misleading content.');
        if (Math.abs(features.sentiment) > 0.5) reasons.push('The article shows extreme sentiment, which may indicate bias or emotional manipulation.');
        if (features.lexical_diversity < 0.3) reasons.push('Low lexical diversity might indicate repetitive or low-quality content.');
        if (reasons.length === 0) reasons.push('The model detected patterns commonly found in misleading articles.');
    }
    return reasons;
}

// Statistics handling
async function fetchStatistics() {
    try {
        const response = await fetch(`${API_URL}/statistics`);
        if (!response.ok) throw new Error('Failed to fetch statistics');
        return await response.json();
    } catch (error) {
        console.error('Error fetching statistics:', error);
        return null;
    }
}

function updateStatisticsDisplay(stats) {
    const statsContainer = document.getElementById('statisticsContainer');
    if (!statsContainer) return;

    // Update total analyses
    document.getElementById('totalAnalyses').textContent = stats.total_analyses;

    // Update feature averages
    const features = stats.feature_averages;
    document.getElementById('avgReadability').textContent = features.readability.toFixed(2);
    document.getElementById('avgSentiment').textContent = features.sentiment.toFixed(2);
    document.getElementById('avgLexical').textContent = features.lexical_diversity.toFixed(2);

    // Update confidence distribution
    const distribution = stats.confidence_distribution;
    document.getElementById('conf0_25').textContent = distribution['0-25'];
    document.getElementById('conf26_50').textContent = distribution['26-50'];
    document.getElementById('conf51_75').textContent = distribution['51-75'];
    document.getElementById('conf76_100').textContent = distribution['76-100'];
}

// Enhanced feedback handling
async function submitFeedback(predictionId, agreed) {
    try {
        const response = await fetch(`${API_URL}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prediction_id: predictionId,
                agreed: agreed
            })
        });

        if (!response.ok) throw new Error('Failed to submit feedback');
        
        const result = await response.json();
        if (result.status === 'success') {
            showFeedbackSuccess();
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        showFeedbackError();
    }
}

function showFeedbackSuccess() {
    const feedbackMessage = document.getElementById('feedbackMessage');
    if (feedbackMessage) {
        feedbackMessage.textContent = 'Thank you for your feedback!';
        feedbackMessage.className = 'feedback-message success';
        setTimeout(() => {
            feedbackMessage.textContent = '';
            feedbackMessage.className = 'feedback-message';
        }, 3000);
    }
}

function showFeedbackError() {
    const feedbackMessage = document.getElementById('feedbackMessage');
    feedbackMessage.textContent = 'Failed to submit feedback. Please try again.';
    feedbackMessage.className = 'feedback-message error';
    setTimeout(() => {
        feedbackMessage.textContent = '';
        feedbackMessage.className = 'feedback-message';
    }, 3000);
}

// Initialize statistics display
async function initializeStatistics() {
    const stats = await fetchStatistics();
    if (stats) {
        updateStatisticsDisplay(stats);
    }
}