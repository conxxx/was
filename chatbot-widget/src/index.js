import './widget.css'; // Import the widget styles
// Entry point for the chatbot widget
import { marked } from 'marked'; // Import marked for configuration
import * as state from './state.js';
import * as config from './config.js';
import * as utils from './utils.js';
import * as api from './api.js';
import * as ui from './ui.js';
import * as handlers from './handlers.js';
import * as voice from './voice.js';
// import * as draggable from './draggable.js'; // Draggable removed

// --- Global Namespace for Control ---
window.chatbotWidget = window.chatbotWidget || {};

// --- Find the script tag and get config ---
const scriptTag = document.currentScript;
const chatbotId = scriptTag?.getAttribute('data-chatbot-id');
const apiKey = scriptTag?.getAttribute('data-api-key');

if (!chatbotId || chatbotId === 'YOUR_CHATBOT_ID_HERE') {
    console.error('Chatbot Widget Error: data-chatbot-id attribute is missing or not set on the script tag.');
    // Stop execution if no Chatbot ID
    // Consider throwing an error or returning to prevent further execution
} else if (!apiKey || apiKey === 'YOUR_API_KEY_HERE') {
    console.error('Chatbot Widget Error: data-api-key attribute is missing or not set on the script tag.');
    // Allow loading but config fetch will use defaults/fail gracefully in api.js
} else {
    console.log(`Chatbot Widget Initializing for ID: ${chatbotId}`);
    state.setChatbotId(chatbotId); // Store chatbotId in state
    // Set credentials for modules that need them directly (optional, could pass as args)
    handlers.setCredentials(chatbotId, apiKey);
    voice.setCredentials(chatbotId, apiKey);
    // draggable.setChatbotId(chatbotId); // Draggable removed
}

// --- Main Initialization Function ---
async function init() {
    // --- END IMMEDIATE LIBRARY CHECK --- // This block is removed by the replace below

    if (!chatbotId || !apiKey) {
        console.error("Widget initialization aborted due to missing chatbotId or apiKey.");
        return; // Don't proceed if critical info is missing
    }

    // Generate/Load Session ID
    const sessionKey = state.getSessionStorageKey(chatbotId);
    let currentSessionId = sessionStorage.getItem(sessionKey);
    if (!currentSessionId) {
        currentSessionId = utils.generateUUID();
        sessionStorage.setItem(sessionKey, currentSessionId);
        console.log("New session ID generated:", currentSessionId);
    } else {
        console.log("Existing session ID loaded:", currentSessionId);
    }
    state.setSessionId(currentSessionId);

    // Check initial consent status
    const consentKey = state.getConsentStorageKey(chatbotId);
    state.setUserConsented(sessionStorage.getItem(consentKey) === 'true');
    console.log("Initial user consent status:", state.userConsented);

    // Configure marked once
    marked.setOptions({ breaks: true, gfm: true });
    console.log("Marked configured.");

    // Fetch config (Libraries are bundled now)
    const fetchedConfig = await api.fetchChatbotConfig(chatbotId, apiKey);

    // Create UI (pass fetched config)
    // ui.addStyles(); // Removed - Styles are imported via widget.css
    ui.createUI(fetchedConfig, chatbotId); // Pass config and chatbotId

    // Create UI (pass fetched config) - This second call seems redundant, removing it and the style call
    // ui.addStyles(); // Removed
    // ui.createUI(fetchedConfig, chatbotId); // Removed redundant call

    // --- Position loading removed as draggable feature is disabled ---
    const widgetContainer = state.uiElements.widgetContainer; // Keep for resize listener

    // Apply final config settings (including default position class)
    // Pass null for savedPosition as it's no longer loaded
    ui.applyConfigSettings(fetchedConfig, chatbotId, null);

    // Add a short delay before making the widget visible
    // This allows the browser time to apply the CSS position class styles
    setTimeout(() => {
        if (state.uiElements.widgetContainer) {
            state.uiElements.widgetContainer.style.opacity = '1';
            state.uiElements.widgetContainer.style.visibility = 'visible';
            console.log("Widget container made visible after delay.");
        }
        // --- Add event listener for file input HERE ---
        if (state.uiElements.imageInput) {
            state.uiElements.imageInput.addEventListener('change', handlers.handleFileSelection);
            console.log("Event listener for file input ('change') added AFTER widget visible.");
        } else {
            console.error("Widget Error: Could not find imageInput element to attach listener (after visible).");
        }
        // --- End add event listener ---
    }, 50); // 50ms delay

    // --- Initialize Dark Mode State ---
    const darkModeKey = state.getDarkModeStorageKey(chatbotId);
    const savedDarkMode = localStorage.getItem(darkModeKey);
    const initialDarkMode = savedDarkMode === 'true';
    state.setIsDarkMode(initialDarkMode);
    console.log(`Widget: Initial dark mode state loaded: ${initialDarkMode}`);
    if (initialDarkMode && state.uiElements.widgetContainer) {
        state.uiElements.widgetContainer.classList.add('widget-dark-mode');
    }
    // Update the button icon based on the loaded state
    if (typeof ui.updateDarkModeButtonIcon === 'function') {
        ui.updateDarkModeButtonIcon();
    } else {
        console.warn("Widget: ui.updateDarkModeButtonIcon function not found during init.");
    }
    // --- End Initialize Dark Mode State ---


    // Add resize listener AFTER initial position logic
    if (widgetContainer) {
         // Add resize listener AFTER initial position logic
          window.addEventListener('resize', () => {
              // if (widgetContainer) draggable.ensureWidgetOnScreen(widgetContainer); // Draggable removed
              // Basic resize handling might be needed later if window shrinks drastically
          });
     }


    // Load initial messages (handles welcome message, history, consent check)
    handlers.loadMessages(); // Uses state internally now


    // Open window if configured to start open
    if (fetchedConfig.start_open && !state.isChatOpen) {
        // Use a small delay to ensure everything is rendered
        setTimeout(handlers.toggleChatWindow, 100);
    }

    console.log("Widget initialized successfully.");
      // Log the chatbot config after the widget is initialized
      handlers.logChatbotConfig();
}


// --- Start Initialization ---
// Use DOMContentLoaded to ensure the DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // DOMContentLoaded has already fired
    init();
}
