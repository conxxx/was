// Functions for creating and manipulating the widget's UI
import { DEFAULT_PRIMARY_COLOR, DEFAULT_TEXT_COLOR, supportedLanguages } from './config.js';
import * as state from './state.js';
import { marked } from 'marked'; // Import marked
import DOMPurify from 'dompurify'; // Import DOMPurify
// Library loaders from utils.js are no longer needed here
// import { fetchAndDisplayLogo as apiFetchAndDisplayLogo } from './api.js'; // Assuming logo update is handled differently
import {
    toggleChatWindow,
    handleSendMessage,
    handleMicClick,
    handleLanguageChange,
    handleFileUploadClick,
    handleFileSelection, // Needed for image input listener
    handleCancelWidgetImage,
    handleSummarizeButtonClick,
    handleClearHistoryClick,
    handleConsentAccept,
    handleFeedbackEventListener, // Centralized listener
    showDetailedFeedbackForm, // Needed for feedback link
    handleToggleDarkMode // Add the missing handler import
} from './handlers.js'; // Import handlers needed for createUI listeners
// import { makeDraggable, ensureWidgetOnScreen } from './draggable.js'; // Draggable removed
// Import playback state for UI updates
// import { activeAudioMessageId, playbackState, currentAudioTime, currentAudioDuration } from './state.js'; // Keep for reference if needed, now accessed via state object
// Import handlers for controls (will be created later)
import { handlePlayPauseClick, handleStopClick, handleSeek } from './handlers.js';

// --- Time Formatting Helper ---
function formatTime(timeInSeconds) {
    if (isNaN(timeInSeconds) || timeInSeconds === Infinity) {
        return '0:00';
    }
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}
// --- End Time Formatting Helper ---


// --- Create UI ---
// Note: Styles are now handled by importing widget.css in index.js
export function createUI(config, chatbotId) { // Accept config and chatbotId
    const widgetContainer = document.createElement('div');
    widgetContainer.id = 'chatbot-widget-container';
    state.setUIElement('widgetContainer', widgetContainer); // Store reference

    const chatToggleButton = document.createElement('button');
    chatToggleButton.id = 'chatbot-toggle-button';
    state.setUIElement('chatToggleButton', chatToggleButton); // Store reference
    let initialButtonContent = '';
    if (config.launcher_icon_url) {
        initialButtonContent = `<img src="${config.launcher_icon_url}" alt="Launcher Icon">`;
    } else if (config.launcher_text) {
        initialButtonContent = `<span>${config.launcher_text}</span>`;
    } else {
        initialButtonContent = '<span>&#128172;</span>';
    }
    chatToggleButton.innerHTML = initialButtonContent;
    chatToggleButton.addEventListener('click', toggleChatWindow);

    const chatWindow = document.createElement('div');
    chatWindow.id = 'chatbot-window';
    state.setUIElement('chatWindow', chatWindow); // Store reference

    let headerHandle = null; // Define headerHandle here

    if (config.show_widget_header !== false) {
        const header = document.createElement('div');
        header.className = 'chatbot-header';
        headerHandle = header; // Assign header as a handle

        const logoImg = document.createElement('img');
        logoImg.id = 'chatbot-logo';
        logoImg.alt = 'Chatbot Logo';
        state.setUIElement('logoImg', logoImg); // Store reference

        const headerTitleElement = document.createElement('span');
        headerTitleElement.textContent = 'Loading...';
        state.setUIElement('headerTitleElement', headerTitleElement); // Store reference

        const closeButton = document.createElement('button');
        closeButton.className = 'chatbot-close-button';
        closeButton.innerHTML = '&times;';
        closeButton.title = 'Close Chat';
        closeButton.addEventListener('click', toggleChatWindow);
        state.setUIElement('closeButton', closeButton); // Store reference

        const clearHistoryButton = document.createElement('button');
        clearHistoryButton.className = 'chatbot-clear-button';
        clearHistoryButton.innerHTML = '🧹';
        clearHistoryButton.title = 'Clear Chat History';
        clearHistoryButton.addEventListener('click', handleClearHistoryClick);
        state.setUIElement('clearHistoryButton', clearHistoryButton); // Store reference

        const darkModeToggleButton = document.createElement('button');
        darkModeToggleButton.id = 'chatbot-dark-mode-toggle';
        darkModeToggleButton.innerHTML = '<i class="bi bi-moon-stars-fill"></i>'; // Default to moon (light mode)
        darkModeToggleButton.title = 'Toggle Dark Mode';
        darkModeToggleButton.addEventListener('click', handleToggleDarkMode); // Call the directly imported handler
        state.setUIElement('darkModeToggleButton', darkModeToggleButton); // Store reference

        header.appendChild(logoImg);
        header.appendChild(headerTitleElement);
        header.appendChild(darkModeToggleButton); // Add before clear/close
        header.appendChild(clearHistoryButton);
        header.appendChild(closeButton);
        chatWindow.appendChild(header);
    }

    const messageContainer = document.createElement('div');
    messageContainer.id = 'chatbot-messages';
    state.setUIElement('messageContainer', messageContainer); // Store reference
    chatWindow.appendChild(messageContainer);

    const consentMessageContainer = document.createElement('div');
    consentMessageContainer.id = 'chatbot-consent-area';
    state.setUIElement('consentMessageContainer', consentMessageContainer); // Store reference
    chatWindow.appendChild(consentMessageContainer);

    const imagePreviewArea = document.createElement('div');
    imagePreviewArea.id = 'widget-image-preview';
    const previewImg = document.createElement('img');
    previewImg.id = 'widget-preview-img';
    previewImg.alt = 'Preview';
    const previewName = document.createElement('span');
    previewName.id = 'widget-preview-name';
    const cancelBtn = document.createElement('button');
    cancelBtn.innerHTML = '<i class="bi bi-x-circle-fill"></i>'; // Use icon
    cancelBtn.title = 'Cancel Image Upload';
    cancelBtn.id = 'widget-cancel-image-btn';
    cancelBtn.addEventListener('click', handleCancelWidgetImage);
    imagePreviewArea.appendChild(previewImg);
    imagePreviewArea.appendChild(previewName);
    imagePreviewArea.appendChild(cancelBtn);
    state.setUIElement('imagePreviewArea', imagePreviewArea); // Store references
    state.setUIElement('previewImg', previewImg);
    state.setUIElement('previewName', previewName);
    state.setUIElement('cancelImageBtn', cancelBtn);
    chatWindow.appendChild(imagePreviewArea);

    const inputArea = document.createElement('div');
    inputArea.id = 'chatbot-input-area';
    state.setUIElement('inputArea', inputArea); // Store reference

    const imageInput = document.createElement('input');
    imageInput.type = 'file';
    imageInput.id = 'widget-image-input';
    imageInput.accept = 'image/*';
    imageInput.style.display = 'none';
    // Listener added in handlers.js or index.js where handleFileSelection is defined
    state.setUIElement('imageInput', imageInput); // Store reference
    inputArea.appendChild(imageInput);

    const textInput = document.createElement('input');
    textInput.type = 'text';
    textInput.placeholder = 'Ask something...';
    // textInput.style.display = 'none'; // Visibility controlled by updateInputDisabledState
    // Keypress listener for Enter key
    textInput.addEventListener('keypress', (e) => {
        // Only send if Enter is pressed AND a request is not currently loading
        if (e.key === 'Enter' && !state.isLoading) {
            handleSendMessage();
        }
        // If loading, Enter key does nothing in the input field
    });
    // Input listener to update Send button state as user types
    textInput.addEventListener('input', () => {
        updateInputDisabledState(); // Re-evaluate button state on input change
    });
    state.setUIElement('textInput', textInput); // Store reference

    const sendButton = document.createElement('button');
    sendButton.id = 'chatbot-send-button';
    sendButton.classList.add('chatbot-action-button'); // Add common class
    sendButton.innerHTML = '<i class="bi bi-send-fill"></i>';
    sendButton.title = 'Send Message';
    sendButton.addEventListener('click', handleSendMessage);
    state.setUIElement('sendButton', sendButton); // Store reference

    inputArea.appendChild(textInput); // Input first

    const micButton = document.createElement('button');
    micButton.innerHTML = '<i class="bi bi-mic-fill"></i>';
    micButton.type = 'button';
    micButton.title = 'Record Voice Input';
    micButton.classList.add('chatbot-action-button'); // Add common class
    micButton.addEventListener('click', handleMicClick);
    state.setUIElement('micButton', micButton); // Store reference
    inputArea.appendChild(micButton); // Append action buttons

    const languageSelector = document.createElement('select');
    languageSelector.title = 'Select language';
    // languageSelector.style.display = 'none'; // Visibility controlled by updateInputDisabledState
    languageSelector.innerHTML = '';
    supportedLanguages.forEach(lang => {
        const voiceOption = document.createElement('option');
        voiceOption.value = lang.code;
        voiceOption.textContent = lang.name;
        languageSelector.appendChild(voiceOption);
    });
    languageSelector.addEventListener('change', handleLanguageChange);
    state.setUIElement('languageSelector', languageSelector); // Store reference
    inputArea.appendChild(languageSelector); // Append action buttons

    const fileUploadButton = document.createElement('button');
    fileUploadButton.innerHTML = '<i class="bi bi-paperclip"></i>';
    fileUploadButton.type = 'button';
    fileUploadButton.title = 'Upload Image for Analysis';
    fileUploadButton.classList.add('chatbot-action-button'); // Add common class
    fileUploadButton.addEventListener('click', handleFileUploadClick);
    state.setUIElement('fileUploadButton', fileUploadButton); // Store reference
    inputArea.appendChild(fileUploadButton); // Append action buttons

    const summarizeButton = document.createElement('button');
    summarizeButton.innerHTML = '<i class="bi bi-card-text"></i>';
    summarizeButton.type = 'button';
    summarizeButton.title = 'Summarize Content (URL or Text)';
    summarizeButton.classList.add('chatbot-action-button'); // Add common class
    summarizeButton.addEventListener('click', handleSummarizeButtonClick);
    state.setUIElement('summarizeButton', summarizeButton); // Store reference
    inputArea.appendChild(summarizeButton); // Append action buttons

    inputArea.appendChild(sendButton); // Send button last

    chatWindow.appendChild(inputArea);

    const voiceStatusElement = document.createElement('div');
    voiceStatusElement.id = 'chatbot-voice-status';
    state.setUIElement('voiceStatusElement', voiceStatusElement); // Store reference
    chatWindow.appendChild(voiceStatusElement);

    widgetContainer.appendChild(chatWindow);
    widgetContainer.appendChild(chatToggleButton);

    // Positioning styles are now part of widget.css

    document.body.appendChild(widgetContainer);

    // Draggable functionality removed
    // makeDraggable(widgetContainer, headerHandle, chatToggleButton);

    // Add centralized feedback listener AFTER messageContainer is created
    if (state.uiElements.messageContainer) { // Check if messageContainer exists
        state.uiElements.messageContainer.addEventListener('click', handleFeedbackEventListener);
    }
}


// --- Add Message to UI ---
// No longer async as library readiness is handled by imports
// --- START REPLACEMENT: addMessage ---
export function addMessage(content, role, isError = false, skipSave = false, messageId = null, audioUrl = null) { // Add audioUrl parameter
    // Libraries are imported, no need for ensureLibsReady check here

    const { messageContainer } = state.uiElements; // Get elements from state
    if (!messageContainer) return; // Safety check

    const messageElement = document.createElement('div');
    messageElement.classList.add('chatbot-message');
    messageElement.classList.add(role);
    if (isError) {
        messageElement.classList.add('error');
    }

    const labelElement = document.createElement('span');
    labelElement.className = 'msg-label';
    labelElement.textContent = role === 'user' ? 'ME' : 'OUR AI';
    messageElement.appendChild(labelElement);

    const currentMessageId = messageId || `local-${Date.now()}-${Math.random().toString(16).substring(2, 8)}`;
    messageElement.setAttribute('data-message-id', currentMessageId);

    let messageHTML = '';
    const config = state.chatbotConfig; // Use config from state
    console.log(`[ui:addMessage] Adding message. Role: ${role}, isError: ${isError}, messageId: ${currentMessageId}, config.voice_enabled: ${config.voice_enabled}`); // LOG: Entry & Config Check

    if (role === 'assistant' && !isError) {
        if (config.avatar_url) {
            const avatarImg = document.createElement('img');
            avatarImg.src = config.avatar_url;
            avatarImg.alt = 'Avatar';
            avatarImg.className = 'chatbot-avatar';
            messageElement.appendChild(avatarImg);
        }
            if (typeof content === 'string') {
                // Use imported libraries directly
                try {
                    // marked is configured once in index.js
                    const rawHTML = marked(content); // Use imported marked
                    messageHTML = DOMPurify.sanitize(rawHTML); // Use imported DOMPurify
                } catch (e) {
                    console.error("Error processing Markdown/Sanitizing:", e);
                    // Fallback to text content if processing fails
                    messageElement.textContent = content;
                    messageHTML = null; // Prevent setting innerHTML later
            }
        } else {
            console.error("Invalid content received in addMessage:", content);
            messageElement.textContent = '';
            messageHTML = null;
        }
    } else {
        // For user messages or errors, just display the text content safely
        messageHTML = document.createTextNode(content).textContent; // Safely encode content
    }

    const contentContainer = document.createElement('div');
    contentContainer.style.flexGrow = '1';
    if (messageHTML !== null) {
        contentContainer.innerHTML = messageHTML;
    } else {
         // If messageHTML is null (meaning we used textContent above),
         // we need to re-assign the text content here.
         // This handles the case where Markdown/DOMPurify failed or wasn't available.
         contentContainer.textContent = content;
    }


    if (config.show_message_timestamps !== false) {
        const timeElement = document.createElement('span');
        timeElement.className = 'msg-time';
        timeElement.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        contentContainer.appendChild(timeElement);
    }

    messageElement.appendChild(contentContainer); // Add content div first

    // --- Add Audio Controls (if assistant message and voice enabled) ---
    if (role === 'assistant' && !isError && config.voice_enabled === true) {
        console.log(`[ui:addMessage] Conditions met for adding audio controls for messageId: ${currentMessageId}`); // LOG: Condition Met
        const audioControlContainer = document.createElement('div');
        audioControlContainer.className = 'audio-controls'; // Initially hidden via CSS
        audioControlContainer.setAttribute('data-audio-controls-for', currentMessageId);

        const playPauseButton = document.createElement('button');
        playPauseButton.innerHTML = '▶️'; // Default: Play icon - Will be updated by updateMessageAudioControlsUI
        playPauseButton.title = 'Play';
        playPauseButton.setAttribute('data-audio-action', 'play-pause');
        playPauseButton.addEventListener('click', () => handlePlayPauseClick(currentMessageId));

        const stopButton = document.createElement('button');
        stopButton.innerHTML = '⏹️';
        stopButton.title = 'Stop';
        stopButton.style.display = 'none'; // Hide stop initially
        stopButton.setAttribute('data-audio-action', 'stop');
        stopButton.addEventListener('click', () => handleStopClick(currentMessageId));

        const currentTimeDisplay = document.createElement('span');
        currentTimeDisplay.className = 'time-display';
        currentTimeDisplay.textContent = '0:00';
        currentTimeDisplay.setAttribute('data-audio-time', 'current');

        const seekBar = document.createElement('input');
        seekBar.type = 'range';
        seekBar.min = '0';
        seekBar.max = '0'; // Will be set on loadedmetadata
        seekBar.value = '0';
        seekBar.disabled = true;
        seekBar.title = 'Seek';
        seekBar.setAttribute('data-audio-seek', 'seek');
        // Use 'input' for smoother seeking while dragging
        seekBar.addEventListener('input', (e) => handleSeek(currentMessageId, e.target.value));
        // 'change' event can also be used if needed for final value after release

        const durationDisplay = document.createElement('span');
        durationDisplay.className = 'time-display';
        durationDisplay.textContent = '0:00';
        durationDisplay.setAttribute('data-audio-time', 'duration');

        audioControlContainer.appendChild(playPauseButton);
        audioControlContainer.appendChild(stopButton);
        audioControlContainer.appendChild(currentTimeDisplay);
        audioControlContainer.appendChild(seekBar);
        audioControlContainer.appendChild(durationDisplay);

        // Append audio controls container AFTER the content container within the message element
        messageElement.appendChild(audioControlContainer);
    } else {
         console.log(`[ui:addMessage] Skipping audio controls for messageId: ${currentMessageId}. Role: ${role}, isError: ${isError}, voice_enabled: ${config.voice_enabled}`); // LOG: Condition Not Met
    }
    // --- End Add Audio Controls ---


    if (role === 'assistant' && !isError && config.feedback_thumbs_enabled === true) {
        const feedbackContainer = document.createElement('div');
        feedbackContainer.style.marginTop = '5px'; // Keep feedback below audio controls if both exist
        feedbackContainer.style.textAlign = (config.avatar_url) ? 'left' : 'right';

        const thumbUp = document.createElement('button');
        thumbUp.innerHTML = '👍';
        thumbUp.classList.add('feedback-thumb');
        thumbUp.setAttribute('data-feedback', 'up'); // Use 'up'/'down' consistently
        thumbUp.setAttribute('data-message-id', currentMessageId);
        // thumbUp.style.marginLeft = '5px'; // Use CSS class
        // thumbUp.style.cursor = 'pointer';
        // thumbUp.style.border = 'none';
        // thumbUp.style.background = 'none';

        const thumbDown = document.createElement('button');
        thumbDown.innerHTML = '👎';
        thumbDown.classList.add('feedback-thumb');
        thumbDown.setAttribute('data-feedback', 'down'); // Use 'up'/'down' consistently
        thumbDown.setAttribute('data-message-id', currentMessageId);
        // thumbDown.style.marginLeft = '5px'; // Use CSS class
        // thumbDown.style.cursor = 'pointer';
        // thumbDown.style.border = 'none';
        // thumbDown.style.background = 'none';

        feedbackContainer.appendChild(thumbUp);
        feedbackContainer.appendChild(thumbDown);

        if (config.detailed_feedback_enabled === true) {
            const detailedLink = document.createElement('a');
            detailedLink.href = '#';
            detailedLink.textContent = 'Provide details';
            detailedLink.classList.add('detailed-feedback-link');
            detailedLink.setAttribute('data-message-id', currentMessageId);
            // detailedLink.style.fontSize = '0.8em'; // Use CSS class
            // detailedLink.style.marginLeft = '10px';
            // detailedLink.style.color = 'var(--chatbot-primary-color)';
            // detailedLink.style.textDecoration = 'underline';
            // Click listener for this link is handled by the central listener on messageContainer
            feedbackContainer.appendChild(detailedLink);
        }

        contentContainer.appendChild(feedbackContainer);
    }

    messageContainer.appendChild(messageElement);
    messageContainer.scrollTop = messageContainer.scrollHeight;

    // Save message to state and potentially localStorage
    if (!skipSave && config.save_history_enabled === true) {
        const chatbotId = state.getChatbotId(); // Get chatbotId from state
        if (chatbotId) {
            const storageKey = state.getStorageKey(chatbotId);
            // Include audioUrl in the saved message object if provided
            state.messages.push({ role: role, content: content, message_id: currentMessageId, audio_url: audioUrl });
            try {
                localStorage.setItem(storageKey, JSON.stringify(state.messages));
            } catch (e) {
                console.warn("Could not save chat history to localStorage. Might be full or disabled.", e);
            }
        } else {
            console.warn("Could not save message: chatbotId not found.");
        }
    }
    return currentMessageId; // Return the actual ID used for the message element
}
// --- END REPLACEMENT: addMessage ---

// --- Show/Hide Loading Indicator and Cancel Button ---
export function showLoading(loading) {
    const { messageContainer, cancelRequestButton } = state.uiElements; // Add cancelRequestButton
    if (!messageContainer) return;

    state.setIsLoading(loading); // Update main loading state
    let loadingElement = messageContainer.querySelector('.chatbot-loading');
    if (loading) {
        if (!loadingElement) {
            loadingElement = document.createElement('div');
            loadingElement.className = 'chatbot-loading';
            loadingElement.textContent = 'Thinking...';
            messageContainer.appendChild(loadingElement);
            messageContainer.scrollTop = messageContainer.scrollHeight;
        }
    } else {
        if (loadingElement) {
            loadingElement.remove();
        }
    }

    // Show/hide cancel button based on loading state
    if (cancelRequestButton) {
        cancelRequestButton.style.display = loading ? 'inline-block' : 'none';
    }

    updateInputDisabledState(); // Update general input disabled state
}

// --- Display Consent UI ---
export function displayConsentUI() {
    const { consentMessageContainer, inputArea } = state.uiElements;
    const config = state.chatbotConfig;

    if (consentMessageContainer && config.consent_required && !state.userConsented) {
        consentMessageContainer.innerHTML = '';
        const consentText = document.createElement('p');
        consentText.textContent = config.consent_message || 'Please consent to continue.';

        const consentAcceptButton = document.createElement('button');
        consentAcceptButton.id = 'chatbot-consent-accept-button';
        consentAcceptButton.textContent = 'Accept';
        consentAcceptButton.addEventListener('click', handleConsentAccept);
        state.setUIElement('consentAcceptButton', consentAcceptButton); // Store reference

        consentMessageContainer.appendChild(consentText);
        consentMessageContainer.appendChild(consentAcceptButton);
        consentMessageContainer.style.display = 'block';

        if (inputArea) inputArea.style.display = 'none';
    } else if (consentMessageContainer) {
        consentMessageContainer.style.display = 'none';
        updateInputDisabledState(); // Re-evaluate input visibility/state
    }
}

// --- Update Voice Status Display ---
export function updateVoiceStatus(message = '', isError = false) {
    const { voiceStatusElement } = state.uiElements;
    if (!voiceStatusElement) return;
    voiceStatusElement.textContent = message;
    voiceStatusElement.style.color = isError ? 'red' : '#666';
}

// --- Update Input Disabled State ---
export function updateInputDisabledState() {
    const {
        textInput, sendButton, micButton, languageSelector,
        fileUploadButton, summarizeButton, inputArea
    } = state.uiElements;
    const config = state.chatbotConfig;

    const disableInputs = state.isLoading || state.isRecording || state.sttLoading || state.ttsLoading || state.widgetIsUploadingImage;

    // Determine visibility based on config and consent
    const showTextInput = config.text_chat_enabled !== false && (!config.consent_required || state.userConsented);
    const showVoiceInput = config.voice_enabled === true && (!config.consent_required || state.userConsented);
    const showFileUpload = config.image_analysis_enabled === true && (!config.consent_required || state.userConsented);
    const showSummarize = config.summarization_enabled === true && (!config.consent_required || state.userConsented);
    const showInputArea = showTextInput || showVoiceInput || showFileUpload || showSummarize; // Show area if any input is enabled

    if (textInput) textInput.style.display = showTextInput ? 'flex' : 'none';
    if (sendButton) sendButton.style.display = showTextInput ? 'flex' : 'none';
    if (micButton) micButton.style.display = showVoiceInput ? 'flex' : 'none';
    // Show language selector if EITHER voice OR summarization is enabled AND consent given
    if (languageSelector) languageSelector.style.display = (showVoiceInput || showSummarize) ? 'flex' : 'none';
    if (fileUploadButton) fileUploadButton.style.display = showFileUpload ? 'flex' : 'none';
    if (summarizeButton) summarizeButton.style.display = showSummarize ? 'flex' : 'none';
    if (inputArea) inputArea.style.display = showInputArea ? 'flex' : 'none';

    // Enable/disable based on loading states and content
    if (textInput) textInput.disabled = disableInputs;
    // Send button specific logic
    if (sendButton) {
        // Determine if sending is possible *when not loading*
        const hasText = textInput && textInput.value.trim() !== '';
        const hasImage = !!state.widgetSelectedImageFile; // Convert to boolean
        // Recalculate disableInputs *just for the send button* to be extra sure
        const sendShouldBeDisabled = state.isLoading || state.isRecording || state.sttLoading || state.ttsLoading || state.widgetIsUploadingImage;
        const canSendNormal = !sendShouldBeDisabled && (hasText || hasImage); // Can send if text OR image is present and not busy

        // --- MORE DEBUG LOGGING ---
        console.log(`[UI Update State Check] isLoading=${state.isLoading}, isRecording=${state.isRecording}, sttLoading=${state.sttLoading}, ttsLoading=${state.ttsLoading}, widgetIsUploadingImage=${state.widgetIsUploadingImage}`);
        console.log(`[UI Update Send Logic] disableInputs(overall)=${disableInputs}, sendShouldBeDisabled=${sendShouldBeDisabled}, hasText=${hasText}, hasImage=${hasImage}, canSendNormal=${canSendNormal}`);
        // --- END MORE DEBUG LOGGING ---


        if (state.isLoading) { // This condition specifically handles the "Cancel" state appearance
            // Change to Cancel button appearance
            // console.log("[updateInputDisabledState] Setting button to CANCEL state (enabled).");
            sendButton.innerHTML = '<i class="bi bi-stop-circle-fill"></i>';
            sendButton.title = 'Cancel Request';
            sendButton.style.background = '#dc3545'; // Red background for cancel
            sendButton.disabled = false; // Cancel button is always enabled
        } else {
            // Revert to Send button appearance
            // console.log(`[updateInputDisabledState] Setting button to SEND state (disabled: ${!canSendNormal}).`);
            sendButton.innerHTML = '<i class="bi bi-send-fill"></i>';
            sendButton.title = 'Send Message';
            // Revert background - the CSS class rule should handle the gradient
            sendButton.style.background = '';
            // Disable based on whether content is available to send
            sendButton.disabled = !canSendNormal;
            console.log(`[UI Update Send Logic] Setting Send button disabled = ${sendButton.disabled}`);
        }
    }
    // End Send button specific logic
    if (micButton) micButton.disabled = disableInputs; // Disable if any main operation is loading
    if (languageSelector) languageSelector.disabled = disableInputs || state.isRecording; // Disable if loading or recording
    if (fileUploadButton) fileUploadButton.disabled = disableInputs; // Disable if any main operation is loading
    if (summarizeButton) {
         // Summarize button is enabled if the feature is on and not otherwise busy.
         // The actual action depends on whether input is present.
         summarizeButton.disabled = disableInputs; // Only disable if other operations are blocking
    }


    // Special handling for mic button appearance during recording or voice API processing
    if (micButton) {
        const isVoiceBusy = state.isRecording || state.sttLoading;
        micButton.innerHTML = isVoiceBusy ? '<i class="bi bi-stop-fill"></i>' : '<i class="bi bi-mic-fill"></i>'; // Stop icon if recording OR processing API
        // Use red background only when actively recording/processing voice, revert otherwise
        // Let the CSS class rule handle the default gradient background
        micButton.style.background = isVoiceBusy ? '#dc3545' : ''; // Red when busy, else default gradient
        // Update title based on the specific busy state
        micButton.title = state.sttLoading ? 'Stop Processing Voice' : (state.isRecording ? 'Stop Recording' : 'Record Voice Input');
        // Disable mic button if *any* blocking operation is happening (consistency)
        micButton.disabled = disableInputs;
        console.log(`[UI Update Mic Logic] isVoiceBusy=${isVoiceBusy}, Setting Mic button disabled = ${micButton.disabled}`);
    }
}

// --- Update Audio Controls UI for a Specific Message ---
// --- START REPLACEMENT: updateMessageAudioControlsUI ---
export function updateMessageAudioControlsUI(messageId) {
    // --- DEBUG LOG: Function Entry & State ---
    console.log(`[DEBUG:ControlsUI] updateMessageAudioControlsUI START for messageId: ${messageId}`);
    console.log(`[DEBUG:ControlsUI]   Current State: activeId=${state.activeAudioMessageId}, playback=${state.playbackState}, time=${state.currentAudioTime}, duration=${state.currentAudioDuration}, ttsLoading=${state.ttsLoading}`);

    const messageElement = state.uiElements.messageContainer?.querySelector(`.chatbot-message[data-message-id="${messageId}"]`);
    if (!messageElement) {
        // --- DEBUG LOG: Message Element Not Found ---
        console.log(`[DEBUG:ControlsUI] Message element not found for messageId: ${messageId}`);
        return;
    }

    const controlsContainer = messageElement.querySelector('.audio-controls');
    // --- DEBUG LOG: Controls Container Found ---
    console.log(`[DEBUG:ControlsUI]   Found message element for ${messageId}: ${!!messageElement}. Controls container found: ${!!controlsContainer}`);

    if (!controlsContainer) {
        console.log(`[DEBUG:ControlsUI]   No controlsContainer found for ${messageId}. Exiting update.`);
        return;
    }

    const playPauseButton = controlsContainer.querySelector('[data-audio-action="play-pause"]');
    const stopButton = controlsContainer.querySelector('[data-audio-action="stop"]');
    const currentTimeDisplay = controlsContainer.querySelector('[data-audio-time="current"]');
    const durationDisplay = controlsContainer.querySelector('[data-audio-time="duration"]');
    const seekBar = controlsContainer.querySelector('[data-audio-seek="seek"]');

    const isActive = state.activeAudioMessageId === messageId;
    const duration = state.currentAudioDuration;
    const currentTime = state.currentAudioTime;
    const currentPlaybackState = state.playbackState;

    // Show controls if there's an active audio message ID for these controls,
    // or if a duration is known (meaning it's ready to be played or has played).
    // This allows controls to appear even if duration is Infinity (live stream) or loading.
    const shouldShowControls = isActive || duration > 0;
    // --- DEBUG LOG: Should Show Controls Check ---
    console.log(`[DEBUG:ControlsUI]   Should show controls for ${messageId}? isActive: ${isActive}, Duration: ${duration}, Result: ${shouldShowControls}`);

    controlsContainer.style.display = shouldShowControls ? 'flex' : 'none';
    console.log(`[DEBUG:ControlsUI]   Set controlsContainer.style.display to: ${controlsContainer.style.display} for ${messageId}`);

    if (shouldShowControls) {
        console.log(`[DEBUG:ControlsUI]   Proceeding to update individual controls for ${messageId}.`);
        if (isActive) {
            // --- DEBUG LOG: Active Path ---
            console.log(`[DEBUG:ControlsUI]   Updating controls for ACTIVE message ${messageId}. Playback state: ${currentPlaybackState}`);
            // Update controls for the currently active audio message
            if (playPauseButton) {
                playPauseButton.innerHTML = currentPlaybackState === 'playing' ? '⏸️' : '▶️';
                playPauseButton.title = currentPlaybackState === 'playing' ? 'Pause' : 'Play';
                playPauseButton.disabled = false; // Should be enabled if active
                console.log(`[DEBUG:ControlsUI]     Play/Pause button for ${messageId}: ${playPauseButton.innerHTML}, disabled: ${playPauseButton.disabled}`);
            }
            if (stopButton) {
                stopButton.style.display = (currentPlaybackState === 'playing' || currentPlaybackState === 'paused') ? 'inline-block' : 'none';
                console.log(`[DEBUG:ControlsUI]     Stop button for ${messageId} display: ${stopButton.style.display}`);
            }
            if (seekBar) {
                seekBar.max = duration || 0;
                seekBar.value = currentTime || 0;
                seekBar.disabled = currentPlaybackState === 'stopped' || !duration || duration === Infinity;
                console.log(`[DEBUG:ControlsUI]     Seek bar for ${messageId}: max=${seekBar.max}, value=${seekBar.value}, disabled=${seekBar.disabled}`);
            }
            if (currentTimeDisplay) {
                currentTimeDisplay.textContent = formatTime(currentTime);
            }
            if (durationDisplay) {
                durationDisplay.textContent = formatTime(duration);
            }
        } else {
            // --- DEBUG LOG: Inactive Path ---
            console.log(`[DEBUG:ControlsUI]   Updating controls for INACTIVE message ${messageId} (but shouldShowControls is true).`);
            // Reset controls for inactive messages (that have duration > 0)
            if (playPauseButton) {
                playPauseButton.innerHTML = '▶️';
                playPauseButton.title = 'Play';
                playPauseButton.disabled = false; // Should be enabled to start playback
                console.log(`[DEBUG:ControlsUI]     Play/Pause button for INACTIVE ${messageId}: ${playPauseButton.innerHTML}, disabled: ${playPauseButton.disabled}`);
            }
            if (stopButton) {
                stopButton.style.display = 'none';
                console.log(`[DEBUG:ControlsUI]     Stop button for INACTIVE ${messageId} display: ${stopButton.style.display}`);
            }
            if (seekBar) {
                seekBar.max = duration || 0; // Show correct max duration
                seekBar.value = 0; // Reset value
                seekBar.disabled = true; // Disable seeking for inactive
                console.log(`[DEBUG:ControlsUI]     Seek bar for INACTIVE ${messageId}: max=${seekBar.max}, value=${seekBar.value}, disabled=${seekBar.disabled}`);
            }
            if (currentTimeDisplay) {
                currentTimeDisplay.textContent = '0:00';
            }
            if (durationDisplay) {
                durationDisplay.textContent = formatTime(duration); // Show correct duration
            }
        }
    } else {
        console.log(`[DEBUG:ControlsUI]   shouldShowControls is false for ${messageId}. Controls remain hidden.`);
    }
    console.log(`[DEBUG:ControlsUI] updateMessageAudioControlsUI END for messageId: ${messageId}`);
}
// --- END REPLACEMENT: updateMessageAudioControlsUI ---

// --- Update Image Preview ---
export function updateImagePreview() {
    console.log("[ui:updateImagePreview] Function called."); // Log function entry
    const { imagePreviewArea, previewImg, previewName } = state.uiElements;
    const selectedFile = state.widgetSelectedImageFile;
    console.log("[ui:updateImagePreview] Selected file from state:", selectedFile); // Log selected file

    if (selectedFile && imagePreviewArea && previewImg && previewName) {
        console.log("[ui:updateImagePreview] Conditions met to display preview.");
        previewName.textContent = selectedFile.name;
        const reader = new FileReader();
        console.log("[ui:updateImagePreview] FileReader created.");

        reader.onloadstart = () => {
            console.log("[ui:updateImagePreview:reader] onloadstart triggered.");
        };
        reader.onprogress = (event) => {
            if (event.lengthComputable) {
                const percentLoaded = Math.round((event.loaded / event.total) * 100);
                console.log(`[ui:updateImagePreview:reader] onprogress: ${percentLoaded}% loaded.`);
            }
        };
        reader.onload = (e) => {
            console.log("[ui:updateImagePreview:reader] onload triggered. Event:", e);
            previewImg.src = e.target.result;
            console.log("[ui:updateImagePreview:reader] previewImg.src set.");
        };
        reader.onerror = (e) => {
            console.error("[ui:updateImagePreview:reader] onerror triggered. Error:", e);
        };
        reader.onloadend = () => {
            console.log("[ui:updateImagePreview:reader] onloadend triggered.");
        };

        reader.readAsDataURL(selectedFile);
        console.log("[ui:updateImagePreview] reader.readAsDataURL() called.");
        imagePreviewArea.style.display = 'flex'; // Or 'block', depending on desired layout
        console.log("[ui:updateImagePreview] Image preview area display set to flex.");
    } else if (imagePreviewArea) {
        console.log("[ui:updateImagePreview] Conditions NOT met to display preview, or clearing preview.");
        imagePreviewArea.style.display = 'none';
        if (previewImg) previewImg.src = '';
        if (previewName) previewName.textContent = '';
        console.log("[ui:updateImagePreview] Image preview cleared.");
    } else {
        console.log("[ui:updateImagePreview] imagePreviewArea not found, cannot update preview.");
    }
    // Also, ensure the main input disabled state is updated as the presence of an image affects send button
    updateInputDisabledState();
    console.log("[ui:updateImagePreview] updateInputDisabledState called.");
}

// --- Fetch and Display Logo ---
// This function now primarily handles the UI update. The actual fetching might be better in api.js or index.js
export function updateLogoDisplay() {
    const { logoImg } = state.uiElements;
    const config = state.chatbotConfig;
    if (logoImg && config.logo_url) {
        logoImg.src = config.logo_url;
        logoImg.style.display = 'inline-block';
        console.log("Logo displayed from:", config.logo_url);
    } else if (logoImg) {
        logoImg.style.display = 'none';
    }
}

// --- Apply Fetched/Default Config Settings ---
// This function coordinates applying settings to the UI and state
export function applyConfigSettings(configToApply, chatbotId, savedPosition = null) { // Accept savedPosition
    console.log("[ui:applyConfigSettings] Applying config settings:", configToApply);
    state.setChatbotConfig(configToApply); // Update the global state config object

    const root = document.documentElement;
    root.style.setProperty('--chatbot-primary-color', configToApply.widget_primary_color || DEFAULT_PRIMARY_COLOR);
    root.style.setProperty('--chatbot-text-color', configToApply.widget_text_color || DEFAULT_TEXT_COLOR);
    root.style.setProperty('--chatbot-widget-background', configToApply.widget_background_color || '#ffffff');
    root.style.setProperty('--chatbot-user-message-background', configToApply.user_message_color || '#dcf8c6');
    root.style.setProperty('--chatbot-bot-message-background', configToApply.bot_message_color || '#eee');
    root.style.setProperty('--chatbot-input-background', configToApply.input_background_color || '#ffffff');

    const { headerTitleElement, chatToggleButton, widgetContainer, languageSelector, clearHistoryButton } = state.uiElements;

    if (headerTitleElement) {
        headerTitleElement.textContent = configToApply.name || 'Chat Assistant';
    }

    if (chatToggleButton) {
        let buttonContent = '';
         if (configToApply.launcher_icon_url) {
             buttonContent = `<img src="${configToApply.launcher_icon_url}" alt="Launcher Icon">`;
         } else if (configToApply.launcher_text) {
             buttonContent = `<span>${configToApply.launcher_text}</span>`;
         } else {
             buttonContent = state.isChatOpen ? '<span>&times;</span>' : '<span>&#128172;</span>';
         }
         chatToggleButton.innerHTML = buttonContent;
    }

    // --- Apply Widget Position ---
    if (widgetContainer) {
        // Remove existing position classes before applying new logic
        widgetContainer.classList.remove(
            'chatbot-position-bottom-right',
            'chatbot-position-bottom-left',
            'chatbot-position-top-right',
            'chatbot-position-top-left'
        );

        // Always apply the default position class based on config
        const positionClass = `chatbot-position-${configToApply.widget_position || 'bottom-right'}`; // Default to bottom-right
        widgetContainer.classList.add(positionClass);
        console.log(`Applied default position class: ${positionClass}`);

        // Visibility will be handled after a short delay in index.js
        // to allow CSS class rules to apply first.
        // widgetContainer.style.opacity = '1'; // Removed
        // widgetContainer.style.visibility = 'visible'; // Removed
    }
    // --- End Apply Widget Position ---
    updateLogoDisplay(); // Update logo based on config

    if (languageSelector) {
        let initialLang = configToApply.text_language || 'en'; // Default to text lang
        if (configToApply.voice_enabled && configToApply.voice_input_language) {
             const simpleInputLang = configToApply.voice_input_language.split('-')[0];
             const supportedLang = supportedLanguages.find(l => l.code === simpleInputLang);
             if (supportedLang) {
                 initialLang = supportedLang.code;
             } else {
                 console.warn(`Widget: Configured voice input language (${configToApply.voice_input_language}) not directly supported. Using default/text language.`);
             }
        }
        state.setSelectedLanguage(initialLang); // Update state
        languageSelector.value = initialLang; // Update UI
    }

    if (clearHistoryButton) {
        clearHistoryButton.style.display = configToApply.allow_user_history_clearing === true ? 'inline-block' : 'none';
    }

    // Load messages (handles welcome message, history, and consent) - Moved to index.js/init
    // loadMessages(configToApply, chatbotId); // Pass chatbotId

    // Update input states based on loaded config
    updateInputDisabledState();

    // Handle start_open configuration - Moved to index.js/init
    // if (configToApply.start_open === true && !state.isChatOpen) {
    //     toggleChatWindow();
    // }
}

// --- Update Dark Mode Button Icon ---
export function updateDarkModeButtonIcon() {
    const { darkModeToggleButton } = state.uiElements;
    if (!darkModeToggleButton) return;

    if (state.isDarkMode) {
        darkModeToggleButton.innerHTML = '<i class="bi bi-sun-fill"></i>'; // Sun icon for dark mode
        darkModeToggleButton.title = 'Switch to Light Mode';
    } else {
        darkModeToggleButton.innerHTML = '<i class="bi bi-moon-stars-fill"></i>'; // Moon icon for light mode
        darkModeToggleButton.title = 'Switch to Dark Mode';
    }
}

// --- Load Messages from Storage ---
// Moved to handlers.js or index.js as it involves state, storage, and UI updates
// export function loadMessages(config, chatbotId) { ... }
