// Shared state variables for the widget

export let isChatOpen = false;
export let messages = []; // Will hold { role: 'user'/'assistant', content: '...', message_id: '...' }
export let sessionId = ''; // Will hold the unique session ID
export let chatbotId = ''; // Will hold the chatbot ID for the current widget instance
export let userConsented = false; // Track user consent state for the session
export let isLoading = false; // Main loading state for RAG
export let isRecording = false;
export let sttLoading = false;
export let ttsLoading = false;
export let chatbotConfig = {}; // Store fetched config globally
export let selectedLanguage = 'en'; // Default language (simple code)
export let mediaRecorder = null; // Holds the MediaRecorder instance
export let localAudioChunks = []; // Use local var for chunks
export let currentAudioPlayer = null; // To manage TTS playback
export let widgetSelectedImageFile = null; // Holds the selected File object for the widget
export let widgetImagePreviewUrl = null; // Holds the blob URL for the image preview
export let widgetIsUploadingImage = false; // Loading state for image upload
export let isSummarizationMode = false; // State for summarization UI mode
export let isDarkMode = false; // State for dark mode
export let lastInputMethod = null; // 'voice', 'text', or null
export let userCancelledRecording = false; // Flag for user-cancelled recording

// --- Audio Playback State ---
export let activeAudioMessageId = null; // ID of the message whose audio is playing/paused
export let playbackState = 'stopped'; // 'playing', 'paused', 'stopped'
export let currentAudioTime = 0;
export let currentAudioDuration = 0;
// --- End Audio Playback State ---

// VAD State
export let audioContext = null;
export let analyserNode = null; // Keep for potential future use, but not used by AudioWorklet VAD
export let scriptProcessorNode = null; // Keep for potential future use, but not used by AudioWorklet VAD
export let mediaStreamSource = null;
export let silenceTimer = null; // Keep for potential future use, but not used by AudioWorklet VAD
export let vadActive = false; // Flag to track if VAD logic is running
export let vadNode = null; // Holds the AudioWorkletNode instance

// UI Element References (will be assigned in ui.js or index.js)
export let uiElements = {
    chatToggleButton: null,
    chatWindow: null,
    messageContainer: null,
    inputArea: null,
    textInput: null,
    sendButton: null,
    micButton: null,
    languageSelector: null,
    voiceStatusElement: null,
    closeButton: null,
    headerTitleElement: null,
    fileUploadButton: null,
    clearHistoryButton: null,
    consentMessageContainer: null,
    consentAcceptButton: null,
    summarizeButton: null,
    imagePreviewArea: null,
    previewImg: null,
    previewName: null,
    cancelImageBtn: null,
    imageInput: null,
    widgetContainer: null, // Added container reference
    logoImg: null, // Added logo reference
    darkModeToggleButton: null, // Added dark mode toggle button reference
};

// --- State Update Functions ---
// It's often better to manage state updates via functions
// to potentially add logic or notifications later.

export function setIsChatOpen(value) { isChatOpen = value; }
export function setMessages(value) { messages = value; }
export function setSessionId(value) { sessionId = value; }
export function setChatbotId(value) { chatbotId = value; }
export function setUserConsented(value) { userConsented = value; }
export function setIsLoading(value) { isLoading = value; }
export function setIsRecording(value) { isRecording = value; }
export function setSttLoading(value) { sttLoading = value; }
// export function setTtsLoading(value) { ttsLoading = value; } // Replaced below
export function setChatbotConfig(value) {
    // Clear existing properties
    for (const key in chatbotConfig) {
        delete chatbotConfig[key];
    }
    // Assign new properties
    Object.assign(chatbotConfig, value);
}
export function setSelectedLanguage(value) { selectedLanguage = value; }
export function setMediaRecorder(value) { mediaRecorder = value; }
export function setLocalAudioChunks(value) { localAudioChunks = value; }
// export function setCurrentAudioPlayer(value) { currentAudioPlayer = value; } // Replaced below
export function setWidgetSelectedImageFile(value) { widgetSelectedImageFile = value; }
export function setWidgetImagePreviewUrl(value) { widgetImagePreviewUrl = value; }
export function setWidgetIsUploadingImage(value) { widgetIsUploadingImage = value; }
export function setIsSummarizationMode(value) { isSummarizationMode = value; }
export function setAudioContext(value) { audioContext = value; }
export function setAnalyserNode(value) { analyserNode = value; }
export function setScriptProcessorNode(value) { scriptProcessorNode = value; }
export function setMediaStreamSource(value) { mediaStreamSource = value; }
export function setSilenceTimer(value) { silenceTimer = value; }
export function setVadActive(value) { vadActive = value; }
export function setVadNode(value) { vadNode = value; } // Added setter for vadNode
export function setIsDarkMode(value) { isDarkMode = value; }
export function setLastInputMethod(value) { lastInputMethod = value; }
export function setUserCancelledRecording(value) { userCancelledRecording = value; }

// Audio Playback Setters
// --- START REPLACEMENT: Audio Playback Setters ---
export function setActiveAudioMessageId(value) {
    console.log('[state] setActiveAudioMessageId: New value =', value); // LOG: State change
    activeAudioMessageId = value;
}
export function setPlaybackState(value) {
    console.log('[state] setPlaybackState: New value =', value); // LOG: State change
    playbackState = value;
}
export function setCurrentAudioTime(value) {
    // console.log('[state] setCurrentAudioTime: New value =', value); // LOG: State change (can be very noisy)
    currentAudioTime = value;
}
export function setCurrentAudioDuration(value) {
    console.log(`[state] setCurrentAudioDuration: Setting duration to: ${value}`); // LOG: State change (Focus here for widget controls)
    currentAudioDuration = value;
}
export function setCurrentAudioPlayer(value) {
    console.log('[state] setCurrentAudioPlayer: Setting player to', value ? 'Audio Object' : 'null'); // LOG: State change
    currentAudioPlayer = value;
}
export function setTtsLoading(value) {
    console.log('[state] setTtsLoading: New value =', value); // LOG: State change
    ttsLoading = value;
}
// --- END REPLACEMENT: Audio Playback Setters ---

// Function to assign UI elements
export function setUIElement(key, element) {
    if (key in uiElements) {
        uiElements[key] = element;
    } else {
        console.warn(`Attempted to set unknown UI element key: ${key}`);
    }
}

// Function to get the storage key
export function getStorageKey(chatbotId) {
    return `chatbot_history_${chatbotId}`;
}

// Function to get the consent storage key
export function getConsentStorageKey(chatbotId) {
    return `chatbot_consent_${chatbotId}`;
}

// Function to get the position storage key
export function getPositionStorageKey(chatbotId) {
    return `chatbot_widget_position_${chatbotId}`;
}

// Function to get the session storage key
export function getSessionStorageKey(chatbotId) {
    return `chatbot_session_id_${chatbotId}`;
}

// Function to get the dark mode storage key
export function getDarkModeStorageKey(chatbotId) {
    return `chatbot_dark_mode_${chatbotId}`;
}

// Getter for chatbotId (optional, can access directly, but good practice)
export function getChatbotId() {
    return chatbotId;
}
