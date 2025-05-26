// Functions for interacting with the backend API
import { API_BASE_URL, DEFAULT_CONFIG } from './config.js';
import * as state from './state.js'; // Import all state exports

// --- Fetch Chatbot Configuration ---
export async function fetchChatbotConfig(chatbotId, apiKey) {
    let config = { ...DEFAULT_CONFIG }; // Start with defaults

    if (!apiKey || apiKey === 'YOUR_API_KEY_HERE') {
        console.warn("Widget Warning: API Key missing, using default configuration.");
        return config;
    }
    try {
        const headers = { 'Authorization': `Bearer ${apiKey}`, 'Accept': 'application/json' };
        const response = await fetch(`${API_BASE_URL}/api/chatbots/${chatbotId}/widget-config`, { method: 'GET', headers });
        if (!response.ok) {
            let errorMsg = `HTTP error ${response.status}`;
            try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
            throw new Error(errorMsg);
        }
        const fetchedConfig = await response.json();

        // Merge fetched config with defaults carefully
        config.name = fetchedConfig.name || config.name;
        config.widget_primary_color = fetchedConfig.widget_primary_color || config.widget_primary_color;
        config.widget_text_color = fetchedConfig.widget_text_color || config.widget_text_color;
        config.widget_welcome_message = fetchedConfig.widget_welcome_message || config.widget_welcome_message;
        config.logo_url = fetchedConfig.logo_url || null;
        config.avatar_url = fetchedConfig.avatar_url || null;
        config.widget_background_color = fetchedConfig.widget_background_color || config.widget_background_color;
        config.user_message_color = fetchedConfig.user_message_color || config.user_message_color;
        config.bot_message_color = fetchedConfig.bot_message_color || config.bot_message_color;
        config.input_background_color = fetchedConfig.input_background_color || config.input_background_color;
        config.voice_enabled = fetchedConfig.voice_enabled === true;
        config.voice_input_language = fetchedConfig.voice_input_language || config.voice_input_language;
        config.voice_output_language = fetchedConfig.voice_output_language || config.voice_output_language;
        config.voice_profile = fetchedConfig.voice_profile || config.voice_profile;
        config.voice_speed = typeof fetchedConfig.voice_speed === 'number' ? fetchedConfig.voice_speed : config.voice_speed;
        config.vad_enabled = typeof fetchedConfig.voice_activity_detection_enabled === 'boolean' ? fetchedConfig.voice_activity_detection_enabled : config.vad_enabled;
        config.text_chat_enabled = typeof fetchedConfig.text_chat_enabled === 'boolean' ? fetchedConfig.text_chat_enabled : config.text_chat_enabled;
        config.text_language = fetchedConfig.text_language || config.text_language;
        config.file_uploads_enabled = typeof fetchedConfig.file_uploads_enabled === 'boolean' ? fetchedConfig.file_uploads_enabled : config.file_uploads_enabled;
        config.save_history_enabled = typeof fetchedConfig.save_history_enabled === 'boolean' ? fetchedConfig.save_history_enabled : config.save_history_enabled;
        config.allowed_file_types = fetchedConfig.allowed_file_types || config.allowed_file_types;
        config.max_file_size_mb = typeof fetchedConfig.max_file_size_mb === 'number' ? fetchedConfig.max_file_size_mb : config.max_file_size_mb;
        config.history_retention_days = typeof fetchedConfig.history_retention_days === 'number' ? fetchedConfig.history_retention_days : config.history_retention_days;
        const fetchedHistoryClearing = fetchedConfig.allow_user_history_clearing;
        config.allow_user_history_clearing = (fetchedHistoryClearing === true || fetchedHistoryClearing === 1) ? true : (fetchedHistoryClearing === false ? false : config.allow_user_history_clearing);
        config.feedback_thumbs_enabled = typeof fetchedConfig.feedback_thumbs_enabled === 'boolean' ? fetchedConfig.feedback_thumbs_enabled : config.feedback_thumbs_enabled;
        const fetchedFeedbackEnabled = fetchedConfig.detailed_feedback_enabled;
        config.detailed_feedback_enabled = (fetchedFeedbackEnabled === true || fetchedFeedbackEnabled === 'true' || fetchedFeedbackEnabled === 1) ? true : (fetchedFeedbackEnabled === false ? false : config.detailed_feedback_enabled);
        config.launcher_text = fetchedConfig.launcher_text || config.launcher_text;
        config.widget_position = fetchedConfig.widget_position || config.widget_position;
        config.launcher_icon_url = fetchedConfig.launcher_icon_url || config.launcher_icon_url;
        config.show_widget_header = typeof fetchedConfig.show_widget_header === 'boolean' ? fetchedConfig.show_widget_header : config.show_widget_header;
        config.show_message_timestamps = typeof fetchedConfig.show_message_timestamps === 'boolean' ? fetchedConfig.show_message_timestamps : config.show_message_timestamps;
        config.start_open = typeof fetchedConfig.start_open === 'boolean' ? fetchedConfig.start_open : config.start_open;
        config.show_typing_indicator = typeof fetchedConfig.show_typing_indicator === 'boolean' ? fetchedConfig.show_typing_indicator : config.show_typing_indicator;
        config.default_error_message = fetchedConfig.default_error_message || config.default_error_message;
        config.fallback_message = fetchedConfig.fallback_message || config.fallback_message;
        config.response_delay_ms = typeof fetchedConfig.response_delay_ms === 'number' && fetchedConfig.response_delay_ms >= 0 ? fetchedConfig.response_delay_ms : config.response_delay_ms;
        config.enable_sound_notifications = typeof fetchedConfig.enable_sound_notifications === 'boolean' ? fetchedConfig.enable_sound_notifications : config.enable_sound_notifications;
        config.consent_required = typeof fetchedConfig.consent_required === 'boolean' ? fetchedConfig.consent_required : config.consent_required;
        config.consent_message = fetchedConfig.consent_message || config.consent_message;
        config.image_analysis_enabled = typeof fetchedConfig.image_analysis_enabled === 'boolean' ? fetchedConfig.image_analysis_enabled : config.image_analysis_enabled;
        config.summarization_enabled = typeof fetchedConfig.summarization_enabled === 'boolean' ? fetchedConfig.summarization_enabled : config.summarization_enabled;

        console.log("Widget config fetched:", config);
    } catch (error) {
        console.error('Widget Error fetching config:', error);
        // Keep default config on error
    }
    return config;
}

// --- Send Text Query or Query with Image ---
// Added optional signal parameter for cancellation
export async function sendQuery(chatbotId, apiKey, userMessageContent, imageFileToSend, signal) {
    let requestBody;
    let headers = { 'Authorization': `Bearer ${apiKey}`, 'Accept': 'application/json' };
    let endpoint;

    if (imageFileToSend) {
        // Use FormData for image uploads
        const formData = new FormData();
        formData.append('query', userMessageContent);
        formData.append('session_id', state.sessionId);
        formData.append('language', state.selectedLanguage);
        formData.append('image', imageFileToSend);
        requestBody = formData;
        // Browser sets Content-Type automatically for FormData
        endpoint = `${API_BASE_URL}/api/chatbots/${chatbotId}/query_with_image`;
    } else {
        // Use JSON for text-only queries
        requestBody = JSON.stringify({
            query: userMessageContent,
            session_id: state.sessionId,
            language: state.selectedLanguage
        });
        headers['Content-Type'] = 'application/json'; // Set Content-Type for JSON
        endpoint = `${API_BASE_URL}/api/chatbots/${chatbotId}/query`;
    }

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: headers,
        body: requestBody, // Use the determined body (JSON or FormData)
        signal: signal // Pass the signal to fetch
    });

    if (!response.ok) {
        let errorMsg = `Query failed (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
    }

    return await response.json(); // Return the result object
}

// --- Send Summarization Request ---
// Added optional signal parameter for cancellation
export async function summarizeContent(chatbotId, apiKey, contentToSummarize, signal) {
    console.log('[api:summarizeContent] Called with chatbotId:', chatbotId, 'apiKey:', apiKey, 'contentToSummarize:', contentToSummarize);
    const payload = {
        // session_id: state.sessionId // Removed session_id
    };

    // Determine content type and language
    let isUrl = false;
    try {
        if (contentToSummarize.startsWith('http://') || contentToSummarize.startsWith('https://')) {
           const parsedUrl = new URL(contentToSummarize);
           isUrl = true;
        }
    } catch (_) {
        isUrl = false;
    }

    // Get language from state
    const selectedLang = state.selectedLanguage || 'en';
    console.log(`[widget-api:summarizeContent] Using language for payload: ${selectedLang}. Current state.selectedLanguage: ${state.selectedLanguage}`); // DEBUG LOG

    payload.target_language = selectedLang; // Use 'target_language' key to match backend expectation
    payload.content = contentToSummarize; // Use 'content' key
    if (isUrl) {
        payload.content_type = 'url';
    } else {
        payload.content_type = 'text';
    }

    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
    const apiUrl = `${API_BASE_URL}/api/chatbots/${chatbotId}/summarize`;
    console.log('[api:summarizeContent] API URL:', apiUrl);
    console.log('[api:summarizeContent] Payload:', payload);
    const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload),
        signal: signal // Pass the signal to fetch
    });

    if (!response.ok) {
        let errorMsg = `Summarization failed (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {
            console.error('[api:summarizeContent] Error parsing JSON error:', e);
        }
        console.error('[api:summarizeContent] Summarization failed with error:', errorMsg);
        throw new Error(errorMsg);
    }

    const result = await response.json(); // Return the result object
    console.log('[api:summarizeContent] Summarization API result:', result);
    return result;
}


// --- Send Voice Interaction ---
// Added optional signal parameter for cancellation
export async function sendVoiceInteraction(chatbotId, apiKey, audioBlob, signal) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'voice_input.webm'); // Send blob with a filename
    formData.append('session_id', state.sessionId);
    formData.append('language', state.selectedLanguage); // Send selected language

    const headers = { 'Authorization': `Bearer ${apiKey}`, 'Accept': 'application/json' }; // No Content-Type for FormData
    const response = await fetch(`${API_BASE_URL}/api/voice/chatbots/${chatbotId}/interact`, {
        method: 'POST',
        headers: headers,
        body: formData,
        signal: signal // Pass the signal to fetch
    });

    if (!response.ok) {
        // Handle cancellation specifically
        if (signal?.aborted) {
            throw new DOMException('Voice interaction request cancelled by user.', 'AbortError');
        }
        let errorMsg = `Voice interaction failed (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
    }

    return await response.json(); // Return the result object
}

// --- Send Feedback (Thumbs Up/Down) ---
export async function sendFeedback(apiKey, messageId, feedbackType) {
    // Map 'up'/'down' to 'positive'/'negative'
    const feedbackTypeV2 = feedbackType === 'up' ? 'positive' : 'negative';
    const payload = {
        feedback_type: feedbackTypeV2,
        session_id: state.sessionId
    };
    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
    const response = await fetch(`${API_BASE_URL}/messages/${messageId}/feedback`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        let errorMsg = `Feedback submission failed (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
    }

    return await response.json();
}

// --- Send Detailed Feedback ---
export async function sendDetailedFeedback(apiKey, messageId, feedbackText) {
    const payload = {
        message_id: messageId,
        feedback_text: feedbackText,
        session_id: state.sessionId
    };
    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    };
    const response = await fetch(`${API_BASE_URL}/api/feedback/detailed`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        let errorMsg = `Detailed feedback submission failed (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
    }

    return await response.json();
}

// --- Clear Chat History ---
export async function clearHistory(apiKey) {
    if (!state.sessionId) {
        throw new Error("Cannot clear history: Session ID not available.");
    }
    const headers = {
        'Authorization': `Bearer ${apiKey}`,
        'Accept': 'application/json'
    };
    const response = await fetch(`${API_BASE_URL}/api/chat-sessions/${state.sessionId}/history`, {
        method: 'DELETE',
        headers: headers
    });

    if (!response.ok) {
        let errorMsg = `Failed to clear history (HTTP ${response.status})`;
        try { errorMsg = (await response.json()).error || errorMsg; } catch (e) {}
        throw new Error(errorMsg);
    }
    // No JSON body expected on successful DELETE
    return true; // Indicate success
}
