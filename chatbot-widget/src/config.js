// Configuration constants and default settings
export const API_BASE_URL = 'http://localhost:5001'; // Ensure this matches your backend
export const DEFAULT_PRIMARY_COLOR = '#007bff';
export const DEFAULT_TEXT_COLOR = '#ffffff';
export const DEFAULT_WELCOME_MESSAGE = 'Hello! How can I assist you today?';

// Add other default config values as needed
export const DEFAULT_CONFIG = {
    name: 'Chat Assistant',
    widget_primary_color: DEFAULT_PRIMARY_COLOR,
    widget_text_color: DEFAULT_TEXT_COLOR,
    widget_welcome_message: DEFAULT_WELCOME_MESSAGE,
    logo_url: null,
    avatar_url: null,
    widget_background_color: '#ffffff',
    user_message_color: '#dcf8c6',
    bot_message_color: '#eee',
    input_background_color: '#ffffff',
    voice_enabled: false,
    voice_input_language: 'en-US',
    voice_output_language: 'en-US',
    voice_profile: 'en-US-Standard-C',
    voice_speed: 1.0,
    vad_enabled: true,
    text_chat_enabled: true,
    text_language: 'en',
    file_uploads_enabled: false,
    allowed_file_types: null,
    max_file_size_mb: 10,
    save_history_enabled: true,
    history_retention_days: null,
    allow_user_history_clearing: false,
    feedback_thumbs_enabled: true,
    detailed_feedback_enabled: false,
    launcher_text: '',
    launcher_icon_url: null,
    widget_position: 'bottom-right',
    show_widget_header: true,
    show_message_timestamps: true,
    start_open: false,
    show_typing_indicator: true,
    default_error_message: 'Sorry, an error occurred. Please try again.',
    fallback_message: "Sorry, I can't help with that right now.",
    response_delay_ms: 0,
    enable_sound_notifications: false,
    consent_required: false,
    consent_message: '',
    image_analysis_enabled: false,
    summarization_enabled: true, // Override to true
};

// Supported languages (BCP-47 codes) - simple codes for backend
export const supportedLanguages = [
    { code: 'am', name: 'Amharic' },
    { code: 'en', name: 'English' },
    { code: 'fr', name: 'French' },
    { code: 'ar', name: 'Arabic' },
    { code: 'ru', name: 'Russian' },
    { code: 'uk', name: 'Ukrainian' },
    { code: 'es', name: 'Spanish' },
    { code: 'he', name: 'Hebrew' }
];

// VAD Constants
export const SILENCE_THRESHOLD = -25; // dB, adjust as needed
export const SILENCE_DURATION_MS = 1500; // Stop after 1.5 seconds of silence
