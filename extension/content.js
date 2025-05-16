// Debug log to verify content script is loaded
console.log("Content script loaded");

// Function to extract main article content
function extractArticleContent() {
    // Common article content selectors
    const selectors = [
        'article',
        '[role="article"]',
        '.article-content',
        '.post-content',
        '.entry-content',
        '.story-content',
        'main',
        '#main-content',
        '.main-content'
    ];

    // Try each selector
    for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element) {
            // Get all text content, preserving structure
            const content = element.innerText;
            if (content && content.length > 100) { // Basic validation
                return {
                    title: document.title,
                    content: content,
                    url: window.location.href
                };
            }
        }
    }

    // Fallback: get all text from body
    const fallbackContent = document.body ? document.body.innerText : '';
    if (fallbackContent && fallbackContent.length > 100) {
        return {
            title: document.title,
            content: fallbackContent,
            url: window.location.href
        };
    }
    return { title: document.title, content: '', url: window.location.href };
}

// Listen for messages from the extension
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request && request.action === "getContent") {
        const articleData = extractArticleContent();
        sendResponse(articleData);
    }
    return true; // Keep the message channel open for async response
}); 