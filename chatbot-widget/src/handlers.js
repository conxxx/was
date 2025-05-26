// Event handler functions for the widget
// import * as state from './state.js'; // REMOVED
// import * as api from './api.js'; // REMOVED
// import * as ui from './ui.js'; // REMOVED
// import * as voice from './voice.js'; // REMOVED
import {
    isChatOpen, setIsChatOpen, uiElements, chatbotConfig, messages, userConsented,
    currentAudioPlayer, setCurrentAudioPlayer, isRecording, isDarkMode, setIsDarkMode,
    getDarkModeStorageKey, getChatbotId, isLoading, widgetSelectedImageFile,
    isSummarizationMode, setIsSummarizationMode, setTtsLoading, activeAudioMessageId,
    playbackState, setPlaybackState, setActiveAudioMessageId, setCurrentAudioTime,
    setUserConsented, getConsentStorageKey, sttLoading, setWidgetSelectedImageFile,
    setSelectedLanguage, setMessages, setCurrentAudioDuration, // Added setCurrentAudioDuration
    lastInputMethod, setLastInputMethod, // Added for TTS auto-play
    setUserCancelledRecording // Import the new setter
} from './state.js';
import {
    summarizeContent, sendQuery, sendFeedback, sendDetailedFeedback,
    clearHistory // Changed: clearChatHistory -> clearHistory, Removed: getChatHistory
} from './api.js';
import {
    updateDarkModeButtonIcon, showLoading, updateVoiceStatus, addMessage,
    updateInputDisabledState, updateMessageAudioControlsUI,
    displayConsentUI, updateImagePreview // Added updateImagePreview
} from './ui.js';
import {
    stopRecording, playAudioFromUrl, cancelVoiceApiCall, startRecording
} from './voice.js'; // Import voice functions (includes playAudioFromUrl now)
// import { ensureWidgetOnScreen } from './draggable.js'; // Draggable removed

let chatbotId = null; // Will be set during initialization
let apiKey = null; // Will be set during initialization
let currentMessageController = null; // Controller for cancelling ongoing requests

export function setCredentials(id, key) {
    chatbotId = id;
    apiKey = key;
}

// --- Toggle Chat Window ---
export function toggleChatWindow() {
    setIsChatOpen(!isChatOpen); // Use direct state variable and setter
    const { chatToggleButton, chatWindow, messageContainer } = uiElements; // Use direct uiElements
    const config = chatbotConfig; // Use direct chatbotConfig

    let buttonContent = '';
    if (config.launcher_icon_url) {
        buttonContent = `<img src="${config.launcher_icon_url}" alt="Launcher Icon">`;
    } else if (config.launcher_text) {
        buttonContent = `<span>${config.launcher_text}</span>`;
    } else {
        buttonContent = isChatOpen ? '<span>&times;</span>' : '<span>&#128172;</span>'; // Use direct isChatOpen
    }
    if (chatToggleButton) chatToggleButton.innerHTML = buttonContent;
    if (chatWindow) chatWindow.classList.toggle('open', isChatOpen); // Use direct isChatOpen

    if (isChatOpen) { // Use direct isChatOpen
        // setTimeout(() => { // ensureWidgetOnScreen removed
        //     const widgetContainer = uiElements.widgetContainer;
        //     if (widgetContainer) ensureWidgetOnScreen(widgetContainer);
        // }, 0);

        if (messages.length === 0 || !userConsented) { // Use direct messages and userConsented
            loadMessages(); // Load messages on open if needed
        }
        setTimeout(() => {
            if (messageContainer) messageContainer.scrollTop = messageContainer.scrollHeight;
        }, 0);
    } else {
        if (currentAudioPlayer) { // Use direct currentAudioPlayer
            console.log("[handlers:toggleChatWindow] Closing chat, pausing active audio player.");
            currentAudioPlayer.pause(); // Use direct currentAudioPlayer
            setCurrentAudioPlayer(null); // Use direct setCurrentAudioPlayer
        }
        if (isRecording) { // Use direct isRecording
            console.log("[handlers:toggleChatWindow] Closing chat, stopping active recording.");
            stopRecording(); // Use function from voice module directly
        }
    }
}

// --- Handle Dark Mode Toggle ---
export function handleToggleDarkMode() {
    const currentMode = isDarkMode; // Use direct isDarkMode
    const newMode = !currentMode;
    setIsDarkMode(newMode); // Use direct setIsDarkMode

    const darkModeKey = getDarkModeStorageKey(getChatbotId()); // Use direct functions
    try {
        localStorage.setItem(darkModeKey, newMode.toString());
        console.log(`Widget: Saved dark mode preference: ${newMode}`);
    } catch (e) {
        console.warn("Widget: Could not save dark mode preference to localStorage.", e);
    }

    uiElements.widgetContainer?.classList.toggle('widget-dark-mode', newMode); // Use direct uiElements
    console.log(`Widget: Toggled dark mode class. New state: ${newMode}`);

    // Update the button icon/title (assuming updateDarkModeButtonIcon exists)
    if (typeof updateDarkModeButtonIcon === 'function') { // Use direct ui function
        updateDarkModeButtonIcon(); // Use direct ui function
    } else {
        console.warn("Widget: updateDarkModeButtonIcon function not found.");
    }
}

// --- Handle Sending Message (Text, Voice, Image, Summarization) ---
export async function handleSendMessage() {
    console.log("[handlers:handleSendMessage] Called. isLoading:", isLoading); // Log entry and state // Use direct isLoading

    // --- Check if currently loading (acting as Cancel button) ---
    if (isLoading) { // If loading, this block runs // Use direct isLoading
        console.log("[handlers:handleSendMessage] isLoading is true, calling handleCancelRequestClick...");
        handleCancelRequestClick(); // Call the cancel handler
        console.log("[handlers:handleSendMessage] Returned after calling cancel.");
        return; // Stop further execution
    }
    // --- End Cancel Check ---

    console.log("[handlers:handleSendMessage] isLoading is false, proceeding to send."); // Log proceeding
    const { textInput } = uiElements; // Use direct uiElements
    let userMessageContent = textInput ? textInput.value.trim() : '';
    const imageFileToSend = widgetSelectedImageFile; // Use direct widgetSelectedImageFile

    if (userMessageContent === '' && !imageFileToSend && !isSummarizationMode) { // Use direct isSummarizationMode
        console.log("[handlers:handleSendMessage] No content, image, or summarization mode. Returning.");
        return; // Don't send empty unless image attached or summarizing
    }

    // --- Handle Summarization Request ---
    if (isSummarizationMode) { // Use direct isSummarizationMode
        console.log("[handlers:handleSendMessage] Handling summarization request. isSummarizationMode:", isSummarizationMode);
        if (userMessageContent === '') {
            addMessage("Please enter a URL or text to summarize.", 'assistant', true); // Use direct ui function
            return;
        }
        addMessage(`Summarizing: ${userMessageContent}`, 'user'); // Use direct ui function
        console.log("[handlers:handleSendMessage] Starting summarization, calling showLoading(true)...");
        showLoading(true); // Use direct ui function
        updateVoiceStatus(''); // Use direct ui function

        // Create AbortController for this request
        currentMessageController = new AbortController();
        const signal = currentMessageController.signal;
        console.log("[handlers:handleSendMessage] Created AbortController for summarization.");

        try {
            const result = await summarizeContent(chatbotId, apiKey, userMessageContent, signal); // Pass signal // Use direct api function
            console.log("[handlers:handleSendMessage] Summarization API result:", result);
            addMessage(result.summary, 'assistant', false, false, result.message_id); // Use direct ui function
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('[handlers:handleSendMessage] Summarization fetch aborted by user.');
                // Don't add a generic error message for aborts
            } else {
                console.error('[handlers:handleSendMessage] Widget Error during summarization:', error);
                addMessage(chatbotConfig.default_error_message || `Error summarizing: ${error.message}`, 'assistant', true); // Use direct ui function and chatbotConfig
            }
        } finally {
            console.log("[handlers:handleSendMessage] Summarization finally block.");
            currentMessageController = null; // Clear controller when request finishes or fails
            showLoading(false); // Use direct ui function
            setIsSummarizationMode(false); // Use direct state setter
            if (textInput) textInput.placeholder = 'Ask something...';
            if (uiElements.summarizeButton) uiElements.summarizeButton.style.backgroundColor = ''; // Use direct uiElements
            updateInputDisabledState(); // Use direct ui function
        }
        if (textInput) textInput.value = '';
        return;
    }
    // --- End Summarization Handling ---

    // --- Handle Regular Text/Image Query ---
    if (userMessageContent !== '' || imageFileToSend) {
        setLastInputMethod('text'); // Set input method for text/image
        console.log("[handlers:handleSendMessage] Handling regular query. lastInputMethod set to 'text'.");
        if (userMessageContent !== '') {
             addMessage(userMessageContent, 'user'); // Use direct ui function
        } else if (imageFileToSend) {
            addMessage(`[Image: ${imageFileToSend.name}]`, 'user'); // Use direct ui function
        }
        console.log("[handlers:handleSendMessage] Starting query, calling showLoading(true)...");
        showLoading(true); // Use direct ui function
        updateVoiceStatus(''); // Use direct ui function
        if (textInput) textInput.value = '';
        handleCancelWidgetImage(); // Clear image preview

        // Create AbortController for this request
        currentMessageController = new AbortController();
        const signal = currentMessageController.signal;
        console.log("[handlers:handleSendMessage] Created AbortController for query.");

        try {
            const result = await sendQuery(chatbotId, apiKey, userMessageContent, imageFileToSend, signal); // Pass signal // Use direct api function
            console.log("[handlers:handleSendMessage] Query API result:", result);

            let messageContent = null;
            if (typeof result.answer === 'string' && result.answer.trim() !== '') {
                messageContent = result.answer;
            } else if (typeof result.response === 'string' && result.response.trim() !== '') {
                messageContent = result.response;
            }

            if (messageContent) {
                // Pass audio_url to addMessage
                addMessage(messageContent, 'assistant', false, false, result.message_id, result.audio_url); // Use direct ui function
            } else {
                console.error("[handlers:handleSendMessage] Received invalid or empty response from API:", result);
                const defaultError = chatbotConfig?.default_error_message || 'Sorry, received an invalid response.'; // Use direct chatbotConfig
                addMessage(defaultError, 'assistant', true, false, null); // Use direct ui function
            }

            // Handle TTS
            if (chatbotConfig.voice_enabled && result.audio_url && !result.audio_url.includes('example.com')) { // Use direct chatbotConfig, check for valid URL
                console.log(`[handlers:handleSendMessage] TTS enabled and valid audio URL found: ${result.audio_url}. Initiating playback for message ${result.message_id}.`);
                setTtsLoading(true); // Use direct setTtsLoading
                updateInputDisabledState(); // Use direct ui function
                updateVoiceStatus('Loading audio...'); // Change status to loading

                // Stop any currently playing audio and reset state
                if (currentAudioPlayer) { // Use direct currentAudioPlayer
                    console.log(`[handlers:handleSendMessage] Stopping previous player (messageId: ${activeAudioMessageId}) before TTS.`);
                    currentAudioPlayer.pause(); // Use direct currentAudioPlayer
                    // Remove previous listeners
                    currentAudioPlayer.onloadedmetadata = null;
                    currentAudioPlayer.ontimeupdate = null;
                    currentAudioPlayer.onended = null;
                    currentAudioPlayer.onerror = null;
                    currentAudioPlayer.onplay = null;
                    currentAudioPlayer.onpause = null;
                    if (currentAudioPlayer.src && currentAudioPlayer.src.startsWith('blob:')) {
                        URL.revokeObjectURL(currentAudioPlayer.src);
                        console.log(`[handlers:handleSendMessage] Revoked Object URL for previous player: ${currentAudioPlayer.src}`);
                    }
                }
                console.log(`[handlers:handleSendMessage] Resetting audio state before TTS playback for ${result.message_id}.`);
                setActiveAudioMessageId(null);
                setPlaybackState('stopped');
                setCurrentAudioTime(0);
                setCurrentAudioDuration(0); // Reset duration for new audio
                setCurrentAudioPlayer(null); // Clear previous player

                const newPlayer = new Audio(result.audio_url);
                const currentMessageId = result.message_id; // Get message ID
                console.log(`[handlers:handleSendMessage] Created new Audio player for TTS message ${currentMessageId}`);
                setCurrentAudioPlayer(newPlayer); // Use direct setCurrentAudioPlayer

                // --- Attach Event Listeners ---
                console.log(`[handlers:handleSendMessage] Attaching TTS event listeners for message ${currentMessageId}`);
                newPlayer.onloadedmetadata = () => {
                    console.log(`[handlers:handleSendMessage:onloadedmetadata] TTS Triggered for ${currentMessageId}. Duration: ${newPlayer.duration}`);
                    // Only update if this is still the intended player
                    if (currentAudioPlayer === newPlayer) {
                        console.log(`[handlers:handleSendMessage:onloadedmetadata] TTS Updating state for ${currentMessageId}. Setting duration: ${newPlayer.duration}`);
                        setCurrentAudioDuration(newPlayer.duration); // Set duration
                        setActiveAudioMessageId(currentMessageId); // Set active ID
                        setCurrentAudioTime(0);
                        setTtsLoading(false); // Loading finished
                        updateInputDisabledState();
                        updateMessageAudioControlsUI(currentMessageId); // Update controls UI
                        updateVoiceStatus(''); // Clear loading status
                    } else {
                        console.log(`[handlers:handleSendMessage:onloadedmetadata] TTS Event for ${currentMessageId}, but player has changed. Ignoring.`);
                    }
                };

                newPlayer.ontimeupdate = () => {
                    if (currentAudioPlayer === newPlayer) {
                        // console.log(`[handlers:handleSendMessage:ontimeupdate] TTS Time update for ${currentMessageId}: ${newPlayer.currentTime}`); // Too noisy
                        setCurrentAudioTime(newPlayer.currentTime);
                        updateMessageAudioControlsUI(currentMessageId);
                    }
                };

                newPlayer.onplay = () => {
                    if (currentAudioPlayer === newPlayer) {
                        console.log(`[handlers:handleSendMessage:onplay] TTS Audio playing for ${currentMessageId}`);
                        setPlaybackState('playing');
                        setActiveAudioMessageId(currentMessageId); // Ensure active ID
                        updateMessageAudioControlsUI(currentMessageId);
                        updateVoiceStatus('Speaking...'); // Update status
                    } else {
                        console.log(`[handlers:handleSendMessage:onplay] TTS Event for ${currentMessageId}, but player has changed. Ignoring.`);
                    }
                };

                newPlayer.onpause = () => {
                    // Only set to 'paused' if it wasn't intentionally stopped
                    if (currentAudioPlayer === newPlayer && playbackState !== 'stopped') {
                        console.log(`[handlers:handleSendMessage:onpause] TTS Audio paused for ${currentMessageId}`);
                        setPlaybackState('paused');
                        updateMessageAudioControlsUI(currentMessageId);
                        updateVoiceStatus('Paused'); // Update status
                    } else {
                        console.log(`[handlers:handleSendMessage:onpause] TTS Event for ${currentMessageId}, but player changed or state is 'stopped'. Ignoring. Current state: ${playbackState}`);
                    }
                };

                newPlayer.onended = () => {
                    console.log(`[handlers:handleSendMessage:onended] TTS Audio ended for ${currentMessageId}`);
                    if (currentAudioPlayer === newPlayer) {
                        console.log(`[handlers:handleSendMessage:onended] TTS Updating state for ended audio ${currentMessageId}`);
                        setPlaybackState('stopped');
                        setActiveAudioMessageId(null);
                        setCurrentAudioTime(0);
                        // Don't reset duration here, keep it for display

                        // Clean up Blob URL if it exists before nullifying player
                        if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                            URL.revokeObjectURL(newPlayer.src);
                            console.log(`[handlers:handleSendMessage:onended] TTS Revoked Object URL on audio end: ${newPlayer.src}`);
                        }

                        setCurrentAudioPlayer(null); // Clear the player
                        updateMessageAudioControlsUI(currentMessageId);
                        updateVoiceStatus(''); // Clear general status
                        setTtsLoading(false); // Explicitly clear TTS loading state
                        updateInputDisabledState(); // Update overall input state
                    } else {
                        console.log(`[handlers:handleSendMessage:onended] TTS Event for ${currentMessageId}, but player has changed. Ignoring.`);
                    }
                };

                newPlayer.onerror = (e) => {
                    console.error(`[handlers:handleSendMessage:onerror] TTS Audio error for ${currentMessageId}:`, e);
                    if (currentAudioPlayer === newPlayer) {
                        console.log(`[handlers:handleSendMessage:onerror] TTS Updating state for errored audio ${currentMessageId}`);
                        updateVoiceStatus('Error playing audio.', true);
                        setPlaybackState('stopped');
                        setActiveAudioMessageId(null);
                        setCurrentAudioTime(0);
                        setCurrentAudioDuration(0); // Reset duration on error

                        // Clean up Blob URL if it exists before nullifying player
                        if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                            URL.revokeObjectURL(newPlayer.src);
                            console.log(`[handlers:handleSendMessage:onerror] TTS Revoked Object URL on audio error: ${newPlayer.src}`);
                        }

                        setCurrentAudioPlayer(null);
                        setTtsLoading(false);
                        updateInputDisabledState();
                        updateMessageAudioControlsUI(currentMessageId);
                    } else {
                        console.log(`[handlers:handleSendMessage:onerror] TTS Event for ${currentMessageId}, but player has changed. Ignoring.`);
                    }
                };
                // --- End Event Listeners ---

                // Conditional Attempt to play
                if (lastInputMethod === 'voice') {
                    console.log(`[handlers:handleSendMessage] Attempting to AUTO-PLAY TTS audio for ${currentMessageId} because lastInputMethod was 'voice'.`);
                    newPlayer.play().catch(e => {
                        console.error("[handlers:handleSendMessage] Error starting AUTO-PLAY TTS audio playback:", e);
                        updateVoiceStatus('Error playing audio.', true);
                        if (currentAudioPlayer === newPlayer) {
                            console.log(`[handlers:handleSendMessage] Updating state after TTS auto-playback start error for ${currentMessageId}`);
                            setPlaybackState('stopped');
                            setActiveAudioMessageId(null);
                            setCurrentAudioPlayer(null);
                            setTtsLoading(false);
                            updateInputDisabledState();
                            updateMessageAudioControlsUI(currentMessageId);
                        } else {
                            console.log(`[handlers:handleSendMessage] TTS Auto-playback start error for ${currentMessageId}, but player has changed. Ignoring.`);
                        }
                    });
                } else {
                    console.log(`[handlers:handleSendMessage] TTS audio ready for ${currentMessageId}, but NOT auto-playing (lastInputMethod: ${lastInputMethod}). Manual play enabled.`);
                    // Player is ready with listeners. UI controls will allow manual play.
                    // onloadedmetadata will still fire, setting duration and clearing ttsLoading.
                    // Ensure UI reflects that audio is ready but not auto-playing.
                    // If ttsLoading was true, onloadedmetadata should handle setting it to false.
                    // If not auto-playing, we might not want to show "Loading audio..." if it loads quickly.
                    // However, the current logic sets ttsLoading true before this block,
                    // and onloadedmetadata sets it false. This should be fine.
                }
            } else {
                 console.warn("[handlers:handleSendMessage] Skipping TTS setup due to missing or placeholder audio URL:", result.audio_url);
                 updateVoiceStatus(''); // Clear status if no audio URL
                 setTtsLoading(false); // Ensure loading state is reset
                 updateInputDisabledState();
            }

        } catch (error) {
             if (error.name === 'AbortError') {
                console.log('[handlers:handleSendMessage] Query fetch aborted by user.');
                // Don't add a generic error message for aborts
            } else {
                console.error('[handlers:handleSendMessage] Widget Error sending message:', error);
                addMessage(chatbotConfig.default_error_message || `Error: ${error.message}`, 'assistant', true); // Use direct ui function and chatbotConfig
            }
        } finally {
            console.log("[handlers:handleSendMessage] Query finally block.");
            currentMessageController = null; // Clear controller when request finishes or fails
            showLoading(false); // Use direct ui function
        }
    }
    // --- End Regular Query Handling ---
}


// --- Audio Playback Handlers ---
// --- START REPLACEMENT: handlePlayPauseClick ---
export function handlePlayPauseClick(messageId) {
    console.log(`[handlers:handlePlayPauseClick] Entered. messageId: ${messageId}`); // LOG: Entry point
    const player = currentAudioPlayer; // Use direct state
    const activeId = activeAudioMessageId; // Use direct state
    const currentPlaybackState = playbackState; // Use direct state
    console.log(`[handlers:handlePlayPauseClick] Current state - activeId: ${activeId}, playbackState: ${currentPlaybackState}, player exists: ${!!player}`); // LOG: State check

    if (player && activeId === messageId) {
        console.log(`[handlers:handlePlayPauseClick] Player exists for this messageId (${messageId}).`); // LOG: Branch taken
        if (currentPlaybackState === 'playing') {
            console.log(`[handlers:handlePlayPauseClick] Pausing playback for ${messageId}`); // LOG: Action
            player.pause();
            // onpause listener in voice.js will set state to 'paused'
        } else if (currentPlaybackState === 'paused' || currentPlaybackState === 'stopped') {
            console.log(`[handlers:handlePlayPauseClick] Attempting to play/resume playback for ${messageId} from state: ${currentPlaybackState}`); // LOG: Action
            player.play().catch(e => {
                console.error(`[handlers:handlePlayPauseClick] Error playing/resuming playback for ${messageId}:`, e); // LOG: Error
                updateVoiceStatus('Error playing audio.', true); // Use direct ui function
                setPlaybackState('stopped'); // Use direct state setter
                setActiveAudioMessageId(null); // Use direct state setter
                setCurrentAudioTime(0); // Use direct state setter
                setCurrentAudioDuration(0); // Reset duration on error
                if (player.src && player.src.startsWith('blob:')) {
                    URL.revokeObjectURL(player.src);
                    console.log(`[handlers:handlePlayPauseClick] Revoked Object URL on playback error: ${player.src}`); // LOG: Cleanup
                }
                setCurrentAudioPlayer(null); // Use direct state setter
                updateMessageAudioControlsUI(messageId); // Use direct ui function
            });
            // onplay listener in voice.js will set state to 'playing'
        } else {
            console.warn(`[handlers:handlePlayPauseClick] Unexpected playback state '${currentPlaybackState}' for active message ${messageId}. Doing nothing.`);
        }
    } else {
        console.log(`[handlers:handlePlayPauseClick] No active player for this messageId (${messageId}) or different message active (${activeId}). Attempting to initiate playback.`); // LOG: Branch taken
        const messageToPlay = messages.find(msg => msg.message_id === messageId && msg.role === 'assistant'); // Use direct state
        console.log(`[handlers:handlePlayPauseClick] Found message in history: ${messageToPlay ? 'Yes' : 'No'}`); // LOG: Message found

        if (messageToPlay && messageToPlay.audio_url) {
            console.log(`[handlers:handlePlayPauseClick] Found message ${messageId} with audio URL: ${messageToPlay.audio_url}. Calling playAudioFromUrl.`); // LOG: Action & URL
            playAudioFromUrl(messageToPlay.audio_url, messageId); // Call function in voice.js
        } else {
            console.error(`[handlers:handlePlayPauseClick] Could not find message ${messageId} or its audio URL in history.`); // LOG: Error
            updateVoiceStatus('Could not find audio for this message.', true); // Use direct ui function
            // Optionally reset state if needed, though playAudioFromUrl should handle this if called previously
            // setPlaybackState('stopped');
            // setActiveAudioMessageId(null);
            // setCurrentAudioTime(0);
            // setCurrentAudioDuration(0);
            updateMessageAudioControlsUI(messageId); // Use direct ui function
        }
    }
    console.log(`[handlers:handlePlayPauseClick] Exiting for messageId: ${messageId}.`); // LOG: Exit point
}
// --- END REPLACEMENT: handlePlayPauseClick ---

// --- START REPLACEMENT: handleStopClick ---
export function handleStopClick(messageId) {
    console.log(`[handlers:handleStopClick] Entered. messageId: ${messageId}`); // LOG: Entry point
    const player = currentAudioPlayer; // Use direct state
    const activeId = activeAudioMessageId; // Use direct state
    console.log(`[handlers:handleStopClick] Current state - activeId: ${activeId}, player exists: ${!!player}`); // LOG: State check

    if (player && activeId === messageId) {
        console.log(`[handlers:handleStopClick] Stopping playback for ${messageId}`); // LOG: Action
        player.pause(); // Pause first
        player.currentTime = 0; // Reset time
        console.log(`[handlers:handleStopClick] Setting playbackState to 'stopped' for ${messageId}`);
        setPlaybackState('stopped'); // Set state BEFORE clearing player
        console.log(`[handlers:handleStopClick] Clearing activeAudioMessageId (was ${activeId})`);
        setActiveAudioMessageId(null); // Clear active ID
        console.log(`[handlers:handleStopClick] Resetting currentAudioTime to 0 for ${messageId}`);
        setCurrentAudioTime(0); // Reset time state
        // Don't reset duration here, keep it for display if needed

        // Clean up Blob URL if it exists before nullifying player
        if (player.src && player.src.startsWith('blob:')) {
            URL.revokeObjectURL(player.src);
            console.log(`[handlers:handleStopClick] Revoked Object URL for stopped audio: ${player.src}`); // LOG: Cleanup
        }

        console.log(`[handlers:handleStopClick] Clearing currentAudioPlayer for ${messageId}`);
        setCurrentAudioPlayer(null); // Clear the player state variable
        updateMessageAudioControlsUI(messageId); // Update UI for the stopped message
        updateVoiceStatus(''); // Clear general status
        updateInputDisabledState(); // Update overall input state
    } else {
         console.warn(`[handlers:handleStopClick] Stop clicked for message ${messageId}, but no active audio player found for this message (activeId: ${activeId}).`); // LOG: Warning
    }
    console.log(`[handlers:handleStopClick] Exiting for messageId: ${messageId}.`); // LOG: Exit point
}
// --- END REPLACEMENT: handleStopClick ---

// --- START REPLACEMENT: handleSeek ---
export function handleSeek(messageId, time) {
    console.log(`[handlers:handleSeek] Entered. messageId: ${messageId}, requested time: ${time}`); // LOG: Entry point
    const player = currentAudioPlayer; // Use direct state
    const activeId = activeAudioMessageId; // Use direct state
    const newTime = parseFloat(time);
    console.log(`[handlers:handleSeek] Current state - activeId: ${activeId}, player exists: ${!!player}, parsed time: ${newTime}`); // LOG: State check

    if (player && activeId === messageId && !isNaN(newTime) && isFinite(newTime)) {
        const duration = player.duration;
        console.log(`[handlers:handleSeek] Player exists for message ${messageId}. Duration: ${duration}`);
        // Check bounds against player duration
        if (newTime >= 0 && newTime <= duration) {
            console.log(`[handlers:handleSeek] Seeking message ${messageId} from ${player.currentTime} to time ${newTime}`); // LOG: Action
            player.currentTime = newTime; // Set player time
            setCurrentAudioTime(newTime); // Update state immediately
            // UI update will happen via ontimeupdate or directly if needed
            updateMessageAudioControlsUI(messageId); // Explicitly update UI after seek
        } else {
             console.warn(`[handlers:handleSeek] Seek time ${newTime} out of bounds [0, ${duration}] for message ${messageId}`); // LOG: Warning
        }
    } else {
         console.warn(`[handlers:handleSeek] Seek attempted for message ${messageId}, but conditions not met: player=${!!player}, activeId=${activeId}, time=${newTime}`); // LOG: Warning
    }
    console.log(`[handlers:handleSeek] Exiting for messageId: ${messageId}.`); // LOG: Exit point
}
// --- END REPLACEMENT: handleSeek ---
// --- End Audio Playback Handlers ---


// --- Handle Feedback Click (Thumbs Up/Down) ---
export async function handleFeedbackClick(event) {
    const button = event.target.closest('.feedback-thumb'); // Ensure we get the button even if icon is clicked
    if (!button) return;

    const feedbackType = button.getAttribute('data-feedback'); // 'up' or 'down'
    const messageId = button.getAttribute('data-message-id');

    if (!messageId || !feedbackType) return;

    const feedbackContainer = button.parentElement;
    const buttons = feedbackContainer.querySelectorAll('.feedback-thumb');
    buttons.forEach(btn => btn.disabled = true); // Disable both buttons

    console.log(`[handlers:handleFeedbackClick] Sending feedback: ${feedbackType} for message ${messageId}`);

    try {
        const result = await sendFeedback(apiKey, messageId, feedbackType); // Use direct api function
        console.log("[handlers:handleFeedbackClick] Feedback result:", result);
        // Leave buttons disabled as visual confirmation
        // Optionally add a class to indicate selection
        button.classList.add('selected');
        // Remove selected class from the other button if present
        const otherButton = feedbackContainer.querySelector(`.feedback-thumb:not([data-feedback="${feedbackType}"])`);
        if (otherButton) otherButton.classList.remove('selected');

    } catch (error) {
        console.error(`[handlers:handleFeedbackClick] Widget Error sending feedback for message ${messageId}:`, error);
        buttons.forEach(btn => {
            btn.disabled = false; // Re-enable on error
            btn.classList.remove('selected'); // Remove selection indication on error
        });
        // Optionally show an error message to the user
        updateVoiceStatus('Failed to send feedback.', true);
    }
}

// --- Show Detailed Feedback Form ---
// This function remains in handlers as it's directly tied to a click event
export function showDetailedFeedbackForm(messageId, messageElement) {
    if (!chatbotConfig || chatbotConfig.detailed_feedback_enabled !== true) { // Use direct chatbotConfig
        console.warn("[handlers:showDetailedFeedbackForm] Blocked: Detailed feedback is disabled.");
        return;
    }
    console.log(`[handlers:showDetailedFeedbackForm] Showing form for message ${messageId}`);

    // Check if a form already exists for this message
    let formContainer = messageElement.querySelector('.detailed-feedback-container');
    if (formContainer) {
        console.log(`[handlers:showDetailedFeedbackForm] Removing existing form for message ${messageId}`);
        formContainer.remove(); // Remove existing form if button is clicked again
        return;
    }

    // Create form elements
    formContainer = document.createElement('div');
    formContainer.className = 'detailed-feedback-container';

    const textarea = document.createElement('textarea');
    textarea.placeholder = 'Provide more details...';
    textarea.rows = 3;

    const submitButton = document.createElement('button');
    submitButton.textContent = 'Submit Feedback';
    submitButton.className = 'detailed-feedback-submit';

    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.className = 'detailed-feedback-cancel';
    cancelButton.onclick = () => {
        console.log(`[handlers:showDetailedFeedbackForm] Cancel button clicked for message ${messageId}`);
        formContainer.remove();
    }

    const statusDiv = document.createElement('div');
    statusDiv.className = 'detailed-feedback-status';

    // Append elements
    formContainer.appendChild(textarea);
    formContainer.appendChild(submitButton);
    formContainer.appendChild(cancelButton);
    formContainer.appendChild(statusDiv);

    // Add event listener for submission
    submitButton.onclick = () => {
        const feedbackText = textarea.value.trim();
        if (feedbackText) {
            console.log(`[handlers:showDetailedFeedbackForm] Submit button clicked for message ${messageId}`);
            handleDetailedFeedbackSubmit(messageId, feedbackText, formContainer, statusDiv);
        } else {
            statusDiv.textContent = 'Please enter feedback.';
            statusDiv.style.color = 'red';
        }
    };

    // Insert the form after the message content
    messageElement.appendChild(formContainer);
    textarea.focus();
}

// --- Handle Detailed Feedback Submission ---
export async function handleDetailedFeedbackSubmit(messageId, feedbackText, formContainer, statusDiv) {
    console.log(`[handlers:handleDetailedFeedbackSubmit] Submitting for message ${messageId}`);
    statusDiv.textContent = 'Submitting...';
    statusDiv.style.color = 'inherit';
    const buttons = formContainer.querySelectorAll('button');
    buttons.forEach(btn => btn.disabled = true);
    const textarea = formContainer.querySelector('textarea');
    if (textarea) textarea.disabled = true;

    try {
        const result = await sendDetailedFeedback(apiKey, messageId, feedbackText); // Use direct api function
        console.log("[handlers:handleDetailedFeedbackSubmit] Detailed feedback result:", result);
        statusDiv.textContent = 'Feedback submitted. Thank you!';
        statusDiv.style.color = 'green';
        setTimeout(() => {
             console.log(`[handlers:handleDetailedFeedbackSubmit] Removing form for message ${messageId} after success.`);
             formContainer.remove();
        }, 2000); // Remove form after success
    } catch (error) {
        console.error(`[handlers:handleDetailedFeedbackSubmit] Widget Error sending detailed feedback for message ${messageId}:`, error);
        statusDiv.textContent = 'Error submitting feedback.';
        statusDiv.style.color = 'red';
        buttons.forEach(btn => btn.disabled = false);
        if (textarea) textarea.disabled = false;
    }
}

// --- Handle Consent Accept ---
export function handleConsentAccept() {
    console.log("[handlers:handleConsentAccept] Consent accepted.");
    setUserConsented(true); // Use direct state setter
    const consentKey = getConsentStorageKey(getChatbotId()); // Use direct functions
    try {
        localStorage.setItem(consentKey, 'true');
        console.log(`[handlers:handleConsentAccept] Saved consent preference to localStorage (key: ${consentKey}).`);
    } catch (e) {
        console.warn("[handlers:handleConsentAccept] Widget: Could not save consent preference to localStorage.", e);
    }
    displayConsentUI(false); // Use direct ui function
    loadMessages(); // Load messages after consent
}

// --- Handle Microphone Click ---
export function handleMicClick() {
    console.log(`[handlers:handleMicClick] Mic button clicked. Current recording state: ${isRecording}`);
    // This function now directly calls startRecording or stopRecording from voice.js
    // The logic to decide which to call is handled by those functions based on the 'isRecording' state.
    if (isRecording) { // Use direct isRecording state
        console.log("[handlers:handleMicClick] Currently recording. User is stopping recording.");
        setUserCancelledRecording(true); // Set the flag to indicate user initiated stop
        console.log("[handlers:handleMicClick] userCancelledRecording flag set to true.");
        stopRecording(); // Use function from voice module directly
        console.log("[handlers:handleMicClick] stopRecording() called.");
    } else {
        console.log("[handlers:handleMicClick] Not currently recording. User is starting recording.");
        startRecording(); // Use function from voice module directly
        console.log("[handlers:handleMicClick] startRecording() called.");
    }
}

// --- Handle File Upload Click ---
export function handleFileUploadClick() {
    console.log(`[handlers:handleFileUploadClick] File upload clicked. image_analysis_enabled: ${chatbotConfig.image_analysis_enabled}`);
    if (!chatbotConfig.image_analysis_enabled) {
      console.log("[handlers:handleFileUploadClick] File upload clicked, but image analysis is disabled.");
      return;
    }
    const { imageInput } = uiElements; // Use direct uiElements
    if (imageInput) {
        console.log("[handlers:handleFileUploadClick] Triggering file input click.");
        imageInput.click();
    } else {
        console.warn("[handlers:handleFileUploadClick] File input element not found.");
    }
}


// Add logging for chatbotConfig
export const logChatbotConfig = () => {
  console.log("[handlers:logChatbotConfig] Chatbot Config:", chatbotConfig);
};

// --- Handle File Selection ---
export function handleFileSelection(event) {
    console.log("[handlers:handleFileSelection] Event listener triggered. Event:", event); // More verbose logging
    const file = event.target.files[0];
    if (file) {
        console.log(`[handlers:handleFileSelection] File selected: ${file.name}, type: ${file.type}, size: ${file.size}`);
        // Basic validation (optional: add more checks like size, type)
        if (file.type.startsWith('image/')) {
            setWidgetSelectedImageFile(file); // Use direct state setter
            console.log("[handlers:handleFileSelection] Image file stored in state.");
            updateImagePreview(); // Call UI function to update preview
        } else {
            alert('Please select an image file.');
            console.warn(`[handlers:handleFileSelection] Invalid file type selected: ${file.type}`);
            setWidgetSelectedImageFile(null); // Use direct state setter
            updateImagePreview(); // Call UI function to clear preview
            event.target.value = null; // Reset file input
        }
    } else {
        console.log("[handlers:handleFileSelection] File selection cancelled by user.");
        setWidgetSelectedImageFile(null); // Clear state if selection is cancelled
        updateImagePreview(); // Call UI function to clear preview
    }
}

// --- Handle Cancel Widget Image ---
export function handleCancelWidgetImage() {
    console.log("[handlers:handleCancelWidgetImage] Cancelling selected image.");
    setWidgetSelectedImageFile(null); // Use direct state setter
    updateImagePreview(); // Call UI function to clear preview
    const { imageInput } = uiElements; // Use direct uiElements, corrected key
    if (imageInput) {
        imageInput.value = null; // Reset the file input element
        console.log("[handlers:handleCancelWidgetImage] File input element reset.");
    }
}

// --- Handle Language Change ---
export function handleLanguageChange(event) {
    const newLang = event.target.value;
    setSelectedLanguage(newLang); // Use direct state setter
    // Potentially trigger actions based on language change, e.g., reload config or messages
    console.log(`[handlers:handleLanguageChange] Widget: Language changed to ${newLang}`);
    // Example: Reload messages or update UI elements if needed
    // loadMessages(); // If messages should reload based on language
}

// --- Handle Summarize Button Click ---
export function handleSummarizeButtonClick() {
    if (!chatbotConfig.summarization_enabled) {
        console.log("[handlers:handleSummarizeButtonClick] Summarize button clicked, but summarization is disabled.");
        return;
    }

    const { textInput, summarizeButton } = uiElements;
    const currentInputText = textInput ? textInput.value.trim() : '';
    const currentIsSummarizationMode = isSummarizationMode;

    console.log(`[handlers:handleSummarizeButtonClick] Clicked. Current mode: ${currentIsSummarizationMode}, Input: "${currentInputText}"`);

    // Scenario 1: Already in summarization mode AND there's input.
    // Or, user clicked Summarize (first time with input), mode was set, and now this is the effective "send".
    if (currentIsSummarizationMode && currentInputText !== '') {
        console.log("[handlers:handleSummarizeButtonClick] Mode is ON and input exists. Proceeding with summarization via handleSendMessage.");
        handleSendMessage(); // This will handle the API call and clear input after.
        return;
    }

    // Scenario 2: Toggle mode or first click without input ready for immediate processing.
    const newSummarizationMode = !currentIsSummarizationMode;
    setIsSummarizationMode(newSummarizationMode);
    console.log(`[handlers:handleSummarizeButtonClick] Toggled summarization mode to: ${newSummarizationMode}`);

    if (textInput) {
        textInput.placeholder = newSummarizationMode ? 'Enter URL or text to summarize...' : 'Ask something...';
        if (newSummarizationMode) {
            // Mode is being turned ON
            if (currentInputText === '') {
                // Mode ON, and no input yet: Prompt user.
                console.log("[handlers:handleSummarizeButtonClick] Mode ON, no input. Prompting user.");
                addMessage("Please enter a URL or text to summarize, then click Summarize or Send.", 'assistant', false);
                textInput.focus();
            } else {
                // Mode ON, and there IS input (e.g., user typed, then clicked Summarize for the first time)
                // The input should NOT be cleared here. It's preserved.
                console.log("[handlers:handleSummarizeButtonClick] Mode ON, input exists. Input preserved. Placeholder updated.");
                textInput.focus(); // Keep focus on the input field
            }
        } else {
            // Mode is being turned OFF: Clear input.
            console.log("[handlers:handleSummarizeButtonClick] Mode OFF. Clearing input.");
            textInput.value = '';
        }
    }

    if (summarizeButton) {
        summarizeButton.style.backgroundColor = newSummarizationMode ? '#4CAF50' : ''; // Example highlight
    }
    updateInputDisabledState();
}

// --- Handle Clear History Click ---
export async function handleClearHistoryClick() {
    if (!chatbotConfig.allow_clear_history) {
        console.log("[handlers:handleClearHistoryClick] Clear history clicked, but it's disabled.");
        return;
    }

    if (confirm("Are you sure you want to clear the chat history?")) {
        console.log("[handlers:handleClearHistoryClick] User confirmed clearing history.");
        try {
            await clearHistory(chatbotId, apiKey); // Use direct api function
            console.log("[handlers:handleClearHistoryClick] History cleared on backend.");
            setMessages([]); // Clear local messages // Use direct state setter
            // renderMessages(); // Removed, handled by state change in ui.js
            addMessage("Chat history cleared.", 'system', true); // Use direct ui function
        } catch (error) {
            console.error("[handlers:handleClearHistoryClick] Widget Error clearing history:", error);
            addMessage("Failed to clear chat history.", 'system', true); // Use direct ui function
        }
    } else {
        console.log("[handlers:handleClearHistoryClick] User cancelled clearing history.");
    }
}

// --- Handle Cancel Request Click (used by Send button when loading) ---
export function handleCancelRequestClick() {
    console.log("[handlers:handleCancelRequestClick] Called.");
    if (currentMessageController) {
        console.log("[handlers:handleCancelRequestClick] Aborting current fetch request...");
        currentMessageController.abort(); // Abort the ongoing fetch request
        currentMessageController = null; // Clear the controller
        showLoading(false); // Hide loading indicator immediately // Use direct ui function
        addMessage("Request cancelled.", 'system', true); // Inform the user // Use direct ui function
        // Reset any relevant states if needed (e.g., summarization mode)
        if (isSummarizationMode) { // Use direct isSummarizationMode
            console.log("[handlers:handleCancelRequestClick] Resetting summarization mode.");
            setIsSummarizationMode(false); // Use direct state setter
            const { textInput, summarizeButton } = uiElements; // Use direct uiElements
            if (textInput) textInput.placeholder = 'Ask something...';
            if (summarizeButton) summarizeButton.style.backgroundColor = '';
        }
        updateInputDisabledState(); // Re-enable input // Use direct ui function
    } else if (sttLoading) { // Check if STT is loading (voice processing) // Use direct sttLoading
         console.log("[handlers:handleCancelRequestClick] Cancelling voice API call (STT was loading)...");
         cancelVoiceApiCall(); // Call the specific cancel function for voice // Use direct voice function
         // The voice.js module should handle UI updates like status messages in its catch/finally block
         // Ensure loading indicators are off and input is enabled
         showLoading(false);
         updateInputDisabledState();
    } else {
        console.log("[handlers:handleCancelRequestClick] No active request or STT loading to cancel.");
    }
}


// --- Load Initial Messages or Welcome Message ---
export function loadMessages() {
    const config = chatbotConfig; // Use direct chatbotConfig
    const consentGiven = userConsented; // Use direct userConsented
    const consentRequired = config.consent_required;

    console.log(`[handlers:loadMessages] Called. Consent Required: ${consentRequired}, Consent Given: ${consentGiven}`);

    if (consentRequired && !consentGiven) {
        console.log("[handlers:loadMessages] Consent required and not given. Displaying consent UI.");
        displayConsentUI(true); // Use direct ui function
        return; // Don't load messages yet
    }

    // If consent is given or not required, proceed
    displayConsentUI(false); // Ensure consent UI is hidden // Use direct ui function

    if (messages.length === 0) { // Use direct messages
        console.log("[handlers:loadMessages] No messages found, attempting to add initial/welcome message.");
        let welcomeMessageText = null;
        let welcomeMessageId = null;

        // Prefer initial_message if available from API/config
        if (config.initial_message) {
            welcomeMessageText = config.initial_message;
            welcomeMessageId = `initial-${Date.now()}`;
        } 
        // Else, check for welcome_message from API/config (API currently provides this)
        else if (config.welcome_message) { 
            welcomeMessageText = config.welcome_message;
            welcomeMessageId = `welcome-${Date.now()}`;
        } 
        // Else, as a final fallback, check for widget_welcome_message (usually from DEFAULT_CONFIG)
        else if (config.widget_welcome_message) { 
            welcomeMessageText = config.widget_welcome_message;
            welcomeMessageId = `widget-welcome-${Date.now()}`;
        }

        if (welcomeMessageText) {
            console.log(`[handlers:loadMessages] Adding welcome message to UI: "${welcomeMessageText}" with ID: ${welcomeMessageId}`);
            // addMessage(text, role, isError, skipSave, messageId, audioUrl)
            // skipSave is false by default in addMessage if not provided, which is what we want (to save to state.messages)
            addMessage(welcomeMessageText, 'assistant', false, false, welcomeMessageId, null); 
        } else {
             console.log("[handlers:loadMessages] No initial_message, welcome_message, or widget_welcome_message defined in config to display.");
        }
        // renderMessages(); // Removed, handled by state change in ui.js
    } else {
        console.log("[handlers:loadMessages] Messages already exist, skipping initial message load.");
        // Ensure messages are rendered if they already exist (might be redundant if UI updates on state change)
        // renderMessages(); // Removed, handled by state change in ui.js
    }
}

// --- Global Event Listener for Feedback/Audio Clicks ---
// Add a single listener to the message container for delegation
export function handleFeedbackEventListener(event) {
    const target = event.target;
    // console.log("[handlers:handleFeedbackEventListener] Event triggered on message container. Target:", target); // Too noisy

    // Handle Feedback Clicks (Thumbs Up/Down)
    const feedbackButton = target.closest('.feedback-thumb');
    if (feedbackButton) {
        console.log("[handlers:handleFeedbackEventListener] Feedback button clicked.");
        handleFeedbackClick(event); // Pass the original event
        return; // Stop further processing if it was a feedback click
    }

    // Handle Detailed Feedback Button Click
    const detailedFeedbackButton = target.closest('.detailed-feedback-trigger');
    if (detailedFeedbackButton) {
        const messageId = detailedFeedbackButton.getAttribute('data-message-id');
        const messageElement = detailedFeedbackButton.closest('.message');
        console.log(`[handlers:handleFeedbackEventListener] Detailed feedback trigger clicked for message ${messageId}.`);
        if (messageId && messageElement) {
            showDetailedFeedbackForm(messageId, messageElement);
        }
        return; // Stop further processing
    }

    // Handle Audio Control Clicks (Play/Pause, Stop, Seek)
    const audioControl = target.closest('[data-message-id][data-action]');
    if (audioControl) {
        const messageId = audioControl.getAttribute('data-message-id');
        const action = audioControl.getAttribute('data-action');
        console.log(`[handlers:handleFeedbackEventListener] Audio control clicked: messageId=${messageId}, action=${action}`);

        if (messageId && action) {
            switch (action) {
                case 'play-pause':
                    handlePlayPauseClick(messageId);
                    break;
                case 'stop':
                    handleStopClick(messageId);
                    break;
                // Add other actions like seek if needed
                default:
                     console.warn(`[handlers:handleFeedbackEventListener] Unknown audio action: ${action}`);
            }
        }
        return; // Stop further processing
    }

     // Handle Seekbar Interaction (Input Event)
     const seekBar = target.closest('.audio-seekbar[data-message-id]');
     if (seekBar && event.type === 'input') { // Listen for 'input' for real-time seeking
         const messageId = seekBar.getAttribute('data-message-id');
         const time = seekBar.value;
         // console.log(`[handlers:handleFeedbackEventListener] Seekbar input event: messageId=${messageId}, time=${time}`); // Too noisy
         if (messageId && time !== null) {
             handleSeek(messageId, time);
         }
         return;
     }

     // console.log("[handlers:handleFeedbackEventListener] Click did not match known handlers."); // Too noisy
}
