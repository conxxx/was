import React, { useState, useEffect, useRef } from 'react'; // Import useRef
import { motion } from 'motion/react'; // Import motion
import apiService from '../services/api'; // Adjust path if needed
import InteractiveColorPicker from './InteractiveColorPicker'; // Import the new color picker
import InteractiveToggle from './InteractiveToggle'; // Import the new toggle

/**
 * Component to manage chatbot widget customization and core feature settings.
 * Requires chatbotId prop.
 */
function WidgetCustomizationSettings({ chatbotId, onSaveSuccess }) { // Add onSaveSuccess prop
  // Widget Appearance State
  const [welcomeMessage, setWelcomeMessage] = useState('');
  const [primaryColor, setPrimaryColor] = useState('#007bff');
  const [textColor, setTextColor] = useState('#ffffff');
  const [logoFile, setLogoFile] = useState(null); // State for the selected logo file
  const [currentLogoUrl, setCurrentLogoUrl] = useState(null); // State to display current logo
  const [avatarFile, setAvatarFile] = useState(null); // State for the selected avatar file
  const [currentAvatarUrl, setCurrentAvatarUrl] = useState(null); // State to display current avatar
  const [widgetBackgroundColor, setWidgetBackgroundColor] = useState('#ffffff');
  const [userMessageColor, setUserMessageColor] = useState('#e0f7fa');
  const [botMessageColor, setBotMessageColor] = useState('#f1f1f1');
  const [inputBackgroundColor, setInputBackgroundColor] = useState('#ffffff');
  const [widgetPosition, setWidgetPosition] = useState('bottom-right'); // State for widget position
  const [launcherIconFile, setLauncherIconFile] = useState(null); // State for the selected launcher icon file
  const [currentLauncherIconUrl, setCurrentLauncherIconUrl] = useState(null); // State to display current launcher icon

  const [launcherText, setLauncherText] = useState(''); // State for launcher button text

  const [defaultErrorMessage, setDefaultErrorMessage] = useState(''); // State for default error message

  const [fallbackMessage, setFallbackMessage] = useState(''); // State for fallback message
  // Core Feature Toggles State
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [textChatEnabled, setTextChatEnabled] = useState(true); // Default to true
  const [fileUploadsEnabled, setFileUploadsEnabled] = useState(false);
  const [saveHistoryEnabled, setSaveHistoryEnabled] = useState(true); // Default to true
  const [allowUserHistoryClearing, setAllowUserHistoryClearing] = useState(false);
  const [feedbackEnabled, setFeedbackEnabled] = useState(true); // Default to true
  const [detailedFeedbackEnabled, setDetailedFeedbackEnabled] = useState(false);
  const [historyRetentionDays, setHistoryRetentionDays] = useState(0); // Default to 0 (indefinite)
  const [voiceActivityDetectionEnabled, setVoiceActivityDetectionEnabled] = useState(false); // VAD Toggle State
  const [enableSoundNotifications, setEnableSoundNotifications] = useState(false); // Sound Notification Toggle State
  const [imageAnalysisEnabled, setImageAnalysisEnabled] = useState(false); // State for image analysis toggle
  const [summarizationEnabled, setSummarizationEnabled] = useState(false); // State for summarization toggle
  const [allowedScrapingDomains, setAllowedScrapingDomains] = useState(''); // State for allowed domains (string)

  const [advancedRagEnabled, setAdvancedRagEnabled] = useState(false); // State for Advanced RAG toggle
  const [showWidgetHeader, setShowWidgetHeader] = useState(true); // Default to true
  const [showMessageTimestamps, setShowMessageTimestamps] = useState(true); // Default to true

  const [startOpen, setStartOpen] = useState(false); // State for starting widget open
  const [showTypingIndicator, setShowTypingIndicator] = useState(true); // Default to true
  const [responseDelayMs, setResponseDelayMs] = useState(0); // State for simulated response delay

  // Base Instructions State
  const [basePrompt, setBasePrompt] = useState(''); // State for base instructions/system prompt

  const [knowledgeAdherenceLevel, setKnowledgeAdherenceLevel] = useState('strict'); // State for knowledge adherence
  // Data & Privacy State
  const [consentMessage, setConsentMessage] = useState(''); // State for consent message
  const [consentRequired, setConsentRequired] = useState(false); // State for consent required toggle
  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');

  // Refs for 3D effect
  const appearanceCardRef = useRef(null);
  const featuresCardRef = useRef(null);

  // --- 3D Hover Effect Logic (adapted from ChatbotCard) ---
  const handleMouseMove = (e, cardRef) => {
    if (!cardRef.current) return;
    const { left, top, width, height } = cardRef.current.getBoundingClientRect();
    const x = e.clientX - left;
    const y = e.clientY - top;
    const rotateX = (y / height - 0.5) * -15; // Reduced rotation intensity
    const rotateY = (x / width - 0.5) * 15;  // Reduced rotation intensity
    cardRef.current.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.03, 1.03, 1.03)`; // Slightly reduced scale
    cardRef.current.style.transition = 'transform 0.1s ease-out';
  };

  const handleMouseLeave = (cardRef) => {
    if (!cardRef.current) return;
    cardRef.current.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
    cardRef.current.style.transition = 'transform 0.4s ease-in-out';
  };
  // -----------------------------


  // Fetch current settings on component mount
  useEffect(() => {
    if (!chatbotId) {
      setError("Chatbot ID is missing.");
      return;
    }

    const fetchSettings = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const details = await apiService.getChatbotDetails(chatbotId);
        console.log('Received chatbot details:', details); // Log the received details
        // Set Widget Appearance
        setWelcomeMessage(details.widget_welcome_message || 'Hello! How can I help you today?');
        setPrimaryColor(details.widget_primary_color || '#007bff');
        setTextColor(details.widget_text_color || '#ffffff');
        setCurrentLogoUrl(details.logo_url || null);
        setCurrentAvatarUrl(details.avatar_url || null);
        setWidgetBackgroundColor(details.widget_background_color || '#ffffff');
        setUserMessageColor(details.user_message_color || '#e0f7fa');
        setBotMessageColor(details.bot_message_color || '#f1f1f1');
        setInputBackgroundColor(details.input_background_color || '#ffffff');
        setLauncherText(details.launcher_text || '');
        setWidgetPosition(details.widget_position || 'bottom-right');
        setCurrentLauncherIconUrl(details.launcher_icon_url || null);

        // Set Core Feature Toggles & Other Settings
        setVoiceEnabled(details.voice_enabled ?? false);
        setTextChatEnabled(details.text_chat_enabled ?? true);
        setFileUploadsEnabled(details.file_uploads_enabled ?? false);
        setSaveHistoryEnabled(details.save_history_enabled ?? true);
        setAllowUserHistoryClearing(details.allow_user_history_clearing ?? false);
        setFeedbackEnabled(details.feedback_enabled ?? true);
        setDetailedFeedbackEnabled(details.detailed_feedback_enabled ?? false);
        setHistoryRetentionDays(details.history_retention_days ?? 0);
        setVoiceActivityDetectionEnabled(details.voice_activity_detection_enabled ?? false);
        setShowWidgetHeader(details.show_widget_header ?? true);
        setShowMessageTimestamps(details.show_message_timestamps ?? true);
        setStartOpen(details.start_open ?? false);
        setShowTypingIndicator(details.show_typing_indicator ?? true);
        setDefaultErrorMessage(details.default_error_message || 'Sorry, an error occurred. Please try again.');
        setFallbackMessage(details.fallback_message || 'Sorry, I don\'t have an answer for that.');
        setResponseDelayMs(details.response_delay_ms ?? 0);
        setEnableSoundNotifications(details.enable_sound_notifications ?? false);
        setBasePrompt(details.base_prompt || ''); // Fetch base prompt

        setKnowledgeAdherenceLevel(details.knowledge_adherence_level || 'strict'); // Fetch adherence level
        // Fetch Data & Privacy Settings
        setConsentMessage(details.consent_message || ''); // Fetch consent message
        setConsentRequired(details.consent_required ?? false); // Fetch consent required flag
        setImageAnalysisEnabled(details.image_analysis_enabled ?? false); // Fetch image analysis flag
        setSummarizationEnabled(details.summarization_enabled ?? false); // Fetch summarization flag
        setAllowedScrapingDomains(details.allowed_scraping_domains || ''); // Fetch allowed domains
        setAdvancedRagEnabled(details.advanced_rag_enabled ?? false); // Fetch Advanced RAG flag
      } catch (err) { // Single catch block
        console.error("Failed to fetch chatbot settings:", err);
        setError(err.message || 'Failed to load current settings.');
        // Keep default values if fetch fails (state already holds defaults)
      } finally { // Single finally block
        setIsLoading(false);
      }
    }; // End of fetchSettings function definition

    fetchSettings(); // Call the async function

  }, [chatbotId]); // End of useEffect hook

  const handleSave = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccessMessage('');

    // Use FormData to handle file upload along with other fields
    const formData = new FormData();

    // Append Widget Appearance fields
    formData.append('widget_welcome_message', welcomeMessage);
    formData.append('widget_primary_color', primaryColor);
    formData.append('widget_text_color', textColor);
    formData.append('widget_background_color', widgetBackgroundColor);
    formData.append('user_message_color', userMessageColor);
    formData.append('bot_message_color', botMessageColor);
    formData.append('input_background_color', inputBackgroundColor);

    formData.append('launcher_text', launcherText);
    formData.append('widget_position', widgetPosition); // Append widget position
    formData.append('show_widget_header', showWidgetHeader); // Append header toggle
    formData.append('show_message_timestamps', showMessageTimestamps); // Append timestamp toggle

    // Append Core Feature Toggles
    console.log('handleSave: voiceEnabled state before append:', voiceEnabled);
    // Send 1 for true, 0 for false, as FormData often stringifies booleans
    formData.append('voice_enabled', String(voiceEnabled)); // Append boolean as "true" or "false" string
    formData.append('text_chat_enabled', textChatEnabled);
    formData.append('file_uploads_enabled', fileUploadsEnabled);
    formData.append('save_history_enabled', saveHistoryEnabled);
    formData.append('allow_user_history_clearing', allowUserHistoryClearing);
    formData.append('feedback_enabled', feedbackEnabled); // Ensure field name matches backend
    formData.append('detailed_feedback_enabled', detailedFeedbackEnabled);
    formData.append('history_retention_days', parseInt(historyRetentionDays, 10) || 0);
    formData.append('voice_activity_detection_enabled', String(voiceActivityDetectionEnabled)); // Include VAD in save payload (send as string "true"/"false")
    formData.append('start_open', startOpen); // Include start_open in save payload

    formData.append('show_typing_indicator', showTypingIndicator); // Include typing indicator in save payload
    formData.append('default_error_message', defaultErrorMessage); // Include default error message

    formData.append('fallback_message', fallbackMessage); // Include fallback message
    formData.append('response_delay_ms', parseInt(responseDelayMs, 10) || 0); // Include response delay
    formData.append('enable_sound_notifications', enableSoundNotifications); // Include sound notification setting
    formData.append('base_prompt', basePrompt); // Include base prompt

    formData.append('knowledge_adherence_level', knowledgeAdherenceLevel); // Include adherence level
    // Append Data & Privacy fields
    formData.append('consent_message', consentMessage); // Include consent message
    formData.append('consent_required', consentRequired); // Include consent required flag
    formData.append('image_analysis_enabled', String(imageAnalysisEnabled)); // Send boolean as string
    formData.append('summarization_enabled', String(summarizationEnabled)); // Add summarization flag
    formData.append('allowed_scraping_domains', allowedScrapingDomains); // Add allowed domains
    formData.append('advanced_rag_enabled', String(advancedRagEnabled)); // Add Advanced RAG flag
    // Append the logo file if selected
    if (logoFile) {
      formData.append('logo', logoFile, logoFile.name); // Key 'logo' must match backend expectation
    }

    // Append the avatar file if selected
    if (avatarFile) {
      formData.append('avatar', avatarFile, avatarFile.name); // Key 'avatar' must match backend expectation
    }

    // Append the launcher icon file if selected
    if (launcherIconFile) {
      formData.append('launcher_icon', launcherIconFile, launcherIconFile.name); // Key 'launcher_icon' must match backend
    }


    // Append client_id (required by backend for verification)
    const clientId = localStorage.getItem('clientId'); // Assuming client_id is stored in localStorage
    if (clientId) {
        formData.append('client_id', clientId);
    } else {
        setError("Client ID not found. Cannot update settings.");
        setIsLoading(false);
        return; // Stop if client_id is missing
    }

    try {
      // Use a general update function assumed to exist in apiService
      // This function MUST now handle sending FormData
      const updatedDetails = await apiService.updateChatbot(chatbotId, formData, clientId); // Pass FormData and clientId
      // Update the displayed logo URL if it changed
      // Update the displayed logo and avatar URLs if they changed
      setCurrentLogoUrl(updatedDetails.logo_url || null);
      setCurrentAvatarUrl(updatedDetails.avatar_url || null);
      setCurrentLauncherIconUrl(updatedDetails.launcher_icon_url || null); // Update launcher icon URL
      setLogoFile(null); // Clear the file input state after successful upload
      setAvatarFile(null); // Clear the avatar file input state
      setLauncherIconFile(null); // Clear the launcher icon file input state
      onSaveSuccess(); // Call the success handler to navigate
    } catch (err) {
      console.error("Failed to update chatbot settings:", err);
      setError(err.message || 'Failed to save settings.');
    } finally {
      setIsLoading(false);
      // Success message is no longer needed as we navigate away
    }
  };

  return (
    <div className="space-y-8"> {/* Add spacing between sections */}
      {/* Display messages */}
      {error && <p className="text-red-500 dark:text-red-400 mb-4">Error: {error}</p>}
      {successMessage && <p className="text-green-500 dark:text-green-400 mb-4">{successMessage}</p>}

      {/* Form wraps everything */}
      <form onSubmit={handleSave} className="space-y-8">

        {/* Widget Appearance Section - Styled Card */}
        <div
          ref={appearanceCardRef}
          onMouseMove={(e) => handleMouseMove(e, appearanceCardRef)}
          onMouseLeave={() => handleMouseLeave(appearanceCardRef)}
          className="bg-white dark:bg-navy-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-navy-700 transition-transform duration-400 ease-in-out"
          style={{ transformStyle: 'preserve-3d' }} // Needed for perspective
        >
          <div style={{ transform: 'translateZ(20px)' }}> {/* Lift content slightly */}
            <h3 className="text-xl font-semibold text-navy-700 dark:text-white mb-6">Widget Appearance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6"> {/* Grid layout */}

              {/* Welcome Message */}
              <div className="col-span-1 md:col-span-2">
                <label htmlFor="welcomeMessage" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Welcome Message:</label>
                <input
                  type="text"
                  id="welcomeMessage"
                  value={welcomeMessage}
                  onChange={(e) => setWelcomeMessage(e.target.value)}
                  disabled={isLoading}
                  maxLength={255}
                  className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"
                />
              </div>

              {/* Color Pickers */}
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Primary Color:</label>
                {/* Replaced Wheel with InteractiveColorPicker */}
                <InteractiveColorPicker selectedColor={primaryColor} onChange={setPrimaryColor} />
              </div>
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Text Color:</label>
                <InteractiveColorPicker selectedColor={textColor} onChange={setTextColor} />
              </div>
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Widget Background:</label>
                <InteractiveColorPicker selectedColor={widgetBackgroundColor} onChange={setWidgetBackgroundColor} />
              </div>
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">User Message Bubble:</label>
                <InteractiveColorPicker selectedColor={userMessageColor} onChange={setUserMessageColor} />
              </div>
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Bot Message Bubble:</label>
                <InteractiveColorPicker selectedColor={botMessageColor} onChange={setBotMessageColor} />
              </div>
              <div className="space-y-4">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Input Area Background:</label>
                <InteractiveColorPicker selectedColor={inputBackgroundColor} onChange={setInputBackgroundColor} />
              </div>

              {/* UI Toggles */}
              <div className="col-span-1 md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
                 <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-navy-700 rounded-md border border-gray-200 dark:border-navy-600">
                   <label htmlFor="showWidgetHeader" className="text-sm font-medium text-gray-700 dark:text-gray-300">Show Widget Header</label>
                   <input type="checkbox" id="showWidgetHeader" checked={showWidgetHeader} onChange={(e) => setShowWidgetHeader(e.target.checked)} disabled={isLoading} className="h-4 w-4 text-brand-600 border-gray-300 rounded focus:ring-brand-500" />
                 </div>
                 <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-navy-700 rounded-md border border-gray-200 dark:border-navy-600">
                   <label htmlFor="showMessageTimestamps" className="text-sm font-medium text-gray-700 dark:text-gray-300">Show Message Timestamps</label>
                   <input type="checkbox" id="showMessageTimestamps" checked={showMessageTimestamps} onChange={(e) => setShowMessageTimestamps(e.target.checked)} disabled={isLoading} className="h-4 w-4 text-brand-600 border-gray-300 rounded focus:ring-brand-500" />
                 </div>
              </div>

              {/* Launcher Text */}
              <div>
                <label htmlFor="launcherText" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Launcher Button Text:</label>
                <input
                  type="text"
                  id="launcherText"
                  value={launcherText}
                  onChange={(e) => setLauncherText(e.target.value)}
                  disabled={isLoading}
                  maxLength={50}
                  placeholder="Leave empty for default icon"
                  className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"
                />
              </div>

              {/* Widget Position */}
              <div>
                <label htmlFor="widgetPosition" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Initial Widget Position:</label>
                <select
                  id="widgetPosition"
                  value={widgetPosition}
                  onChange={(e) => setWidgetPosition(e.target.value)}
                  disabled={isLoading}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-navy-600 focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm rounded-md bg-white dark:bg-navy-700 text-gray-900 dark:text-white"
                >
                  <option value="bottom-right">Bottom Right</option>
                  <option value="bottom-left">Bottom Left</option>
                  <option value="top-right">Top Right</option>
                  <option value="top-left">Top Left</option>
                </select>
              </div>

              {/* File Uploads (Launcher, Logo, Avatar) */}
               <div className="col-span-1 md:col-span-2 grid grid-cols-1 sm:grid-cols-3 gap-4">
                 <div>
                   <label htmlFor="launcherIconUpload" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Launcher Icon:</label>
                   {currentLauncherIconUrl && <img src={currentLauncherIconUrl} alt="Current Launcher" className="h-10 w-10 mb-2 border rounded"/>}
                   <input type="file" id="launcherIconUpload" accept="image/png, image/svg+xml, image/x-icon" onChange={(e) => setLauncherIconFile(e.target.files[0])} disabled={isLoading} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-50 file:text-brand-700 hover:file:bg-brand-100 dark:file:bg-navy-700 dark:file:text-brand-300 dark:hover:file:bg-navy-600"/>
                   <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">PNG, SVG, ICO. Overrides text.</p>
                 </div>
                 <div>
                   <label htmlFor="logoUpload" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Widget Logo:</label>
                   {currentLogoUrl && <img src={currentLogoUrl} alt="Current Logo" className="h-10 max-w-[100px] mb-2 border rounded"/>}
                   <input type="file" id="logoUpload" accept="image/png, image/jpeg, image/gif, image/svg+xml" onChange={(e) => setLogoFile(e.target.files[0])} disabled={isLoading} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-50 file:text-brand-700 hover:file:bg-brand-100 dark:file:bg-navy-700 dark:file:text-brand-300 dark:hover:file:bg-navy-600"/>
                 </div>
                 <div>
                   <label htmlFor="avatarUpload" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Bot Avatar:</label>
                   {currentAvatarUrl && <img src={currentAvatarUrl} alt="Current Avatar" className="h-10 w-10 mb-2 border rounded-full"/>}
                   <input type="file" id="avatarUpload" accept="image/png, image/jpeg, image/gif" onChange={(e) => setAvatarFile(e.target.files[0])} disabled={isLoading} className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-brand-50 file:text-brand-700 hover:file:bg-brand-100 dark:file:bg-navy-700 dark:file:text-brand-300 dark:hover:file:bg-navy-600"/>
                 </div>
               </div>

              {/* Default Messages */}
              <div>
                <label htmlFor="defaultErrorMessage" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Default Error Message:</label>
                <input type="text" id="defaultErrorMessage" value={defaultErrorMessage} onChange={(e) => setDefaultErrorMessage(e.target.value)} disabled={isLoading} maxLength={255} placeholder="E.g., Sorry, an error occurred." className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"/>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Shown on general errors.</p>
              </div>
              <div>
                <label htmlFor="fallbackMessage" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Fallback Message:</label>
                <input type="text" id="fallbackMessage" value={fallbackMessage} onChange={(e) => setFallbackMessage(e.target.value)} disabled={isLoading} maxLength={255} placeholder="E.g., Sorry, I don't understand." className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"/>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Shown when no specific answer is found.</p>
              </div>
            </div>
          </div>
        </div> {/* End of Widget Appearance Section */}


        {/* Base Instructions Section - Styled Card (No 3D effect needed here) */}
        <div className="bg-white dark:bg-navy-800 p-6 rounded-lg shadow-md border border-gray-200 dark:border-navy-700">
          <h3 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Base Instructions (System Prompt)</h3>
          <div>
            <label htmlFor="basePrompt" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Instructions:</label>
            <textarea
              id="basePrompt"
              value={basePrompt}
              onChange={(e) => setBasePrompt(e.target.value)}
              disabled={isLoading}
              rows={5}
              placeholder="Enter base instructions or system prompt for the chatbot..."
              className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white resize-vertical min-h-[100px]"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Guides the chatbot's overall behavior and personality.</p>
          </div>
        </div> {/* End of Base Instructions Section */}


        {/* Knowledge Settings Section - Styled Card (No 3D effect needed here) */}
        <div className="bg-white dark:bg-navy-800 p-6 rounded-lg shadow-md border border-gray-200 dark:border-navy-700">
          <h3 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Knowledge Settings</h3>
          <div className="space-y-4">
            <div>
              <label htmlFor="knowledgeAdherenceLevel" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Knowledge Adherence Level:</label>
              <select
                id="knowledgeAdherenceLevel"
                value={knowledgeAdherenceLevel}
                onChange={(e) => setKnowledgeAdherenceLevel(e.target.value)}
                disabled={isLoading}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-navy-600 focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm rounded-md bg-white dark:bg-navy-700 text-gray-900 dark:text-white"
              >
                <option value="strict">Strict (Answers only from knowledge)</option>
                <option value="moderate">Moderate (Prefer knowledge, allow some flexibility)</option>
                <option value="flexible">Flexible (Use knowledge as guide, more conversational)</option>
              </select>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Controls how strictly the chatbot must adhere to the provided knowledge sources.</p>
            </div>

            {/* Advanced RAG Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-navy-700 rounded-md border border-gray-200 dark:border-navy-600">
               <div>
                 <label htmlFor="advancedRagEnabled" className="text-sm font-medium text-gray-700 dark:text-gray-300">Enable Advanced RAG Pipeline</label>
                 <p className="text-xs text-gray-500 dark:text-gray-400">Uses a more complex pipeline. May increase latency and cost.</p>
               </div>
               <input type="checkbox" id="advancedRagEnabled" checked={advancedRagEnabled} onChange={(e) => setAdvancedRagEnabled(e.target.checked)} disabled={isLoading} className="h-4 w-4 text-brand-600 border-gray-300 rounded focus:ring-brand-500" />
            </div>
          </div>
        </div> {/* End of Knowledge Settings Section */}


        {/* Data & Privacy Section - Styled Card (No 3D effect needed here) */}
        <div className="bg-white dark:bg-navy-800 p-6 rounded-lg shadow-md border border-gray-200 dark:border-navy-700">
          <h3 className="text-xl font-semibold text-navy-700 dark:text-white mb-4">Data & Privacy</h3>
          <div className="space-y-4">
            <div>
              <label htmlFor="consentMessage" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Consent Message:</label>
              <textarea
                id="consentMessage"
                value={consentMessage}
                onChange={(e) => setConsentMessage(e.target.value)}
                disabled={isLoading}
                rows={4}
                placeholder="Enter the message shown to users to obtain consent..."
                className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white resize-vertical min-h-[80px]"
              />
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Displayed if 'Require Consent' is enabled. Supports basic Markdown.</p>
            </div>

            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-navy-700 rounded-md border border-gray-200 dark:border-navy-600">
               <div>
                 <label htmlFor="consentRequired" className="text-sm font-medium text-gray-700 dark:text-gray-300">Require Consent Before Interaction</label>
                 <p className="text-xs text-gray-500 dark:text-gray-400">User must agree to the message above.</p>
               </div>
               <input type="checkbox" id="consentRequired" checked={consentRequired} onChange={(e) => setConsentRequired(e.target.checked)} disabled={isLoading} className="h-4 w-4 text-brand-600 border-gray-300 rounded focus:ring-brand-500" />
            </div>
          </div>
        </div> {/* End of Data & Privacy Section */}


        {/* Core Features Section - Styled Card */}
        <div
          ref={featuresCardRef}
          onMouseMove={(e) => handleMouseMove(e, featuresCardRef)}
          onMouseLeave={() => handleMouseLeave(featuresCardRef)}
          className="bg-white dark:bg-navy-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-navy-700 transition-transform duration-400 ease-in-out"
          style={{ transformStyle: 'preserve-3d' }} // Needed for perspective
        >
          <div style={{ transform: 'translateZ(20px)' }}> {/* Lift content slightly */}
            <h3 className="text-xl font-semibold text-navy-700 dark:text-white mb-6">Core Features</h3>

            {/* Toggles in a Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {/* Helper function or map for cleaner toggle rendering could be used here */}
              {[
                 { id: 'voiceEnabled', label: 'Enable Voice Interaction', state: voiceEnabled, setState: setVoiceEnabled, description: 'Allow users to interact via voice.' },
                 { id: 'voiceActivityDetectionEnabled', label: 'Enable VAD', state: voiceActivityDetectionEnabled, setState: setVoiceActivityDetectionEnabled, description: 'Automatically detect when the user stops speaking.' },
                 { id: 'textChatEnabled', label: 'Enable Text Chat', state: textChatEnabled, setState: setTextChatEnabled, description: 'Allow users to type messages.' },
                 { id: 'fileUploadsEnabled', label: 'Enable File Uploads', state: fileUploadsEnabled, setState: setFileUploadsEnabled, description: 'Allow users to upload files.' },
                 { id: 'imageAnalysisEnabled', label: 'Enable Image Analysis', state: imageAnalysisEnabled, setState: setImageAnalysisEnabled, description: 'Allow the bot to analyze uploaded images.' },
                 { id: 'saveHistoryEnabled', label: 'Save Chat History', state: saveHistoryEnabled, setState: setSaveHistoryEnabled, description: 'Store conversation history.' },
                 { id: 'allowUserHistoryClearing', label: 'Allow User History Clearing', state: allowUserHistoryClearing, setState: setAllowUserHistoryClearing, description: 'Let users clear their own chat history.' },
                 { id: 'feedbackEnabled', label: 'Enable Feedback Buttons', state: feedbackEnabled, setState: setFeedbackEnabled, description: 'Show thumbs up/down feedback buttons.' }, // Corrected label back
                 { id: 'detailedFeedbackEnabled', label: 'Enable Detailed Feedback', state: detailedFeedbackEnabled, setState: setDetailedFeedbackEnabled, description: 'Allow users to provide written feedback.' },
                 { id: 'enableSoundNotifications', label: 'Enable Sound Notifications', state: enableSoundNotifications, setState: setEnableSoundNotifications, description: 'Play sounds for new messages.' },
                 { id: 'startOpen', label: 'Start Widget Open', state: startOpen, setState: setStartOpen, description: 'Open the widget automatically on page load.' },
                 { id: 'showTypingIndicator', label: 'Show Typing Indicator', state: showTypingIndicator, setState: setShowTypingIndicator, description: 'Show when the bot is "typing".' },
                 { id: 'summarizationEnabled', label: 'Enable Summarization', state: summarizationEnabled, setState: setSummarizationEnabled, description: 'Allow users to request summaries.' },
              ].map(({ id, label, state, setState, description }) => ( // Destructure again, add description
                 <div key={id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-navy-700 rounded-md border border-gray-200 dark:border-navy-600 transition-shadow hover:shadow-md">
                   <div className="flex-grow mr-4">
                     <label htmlFor={id} className="text-sm font-medium text-gray-700 dark:text-gray-300 cursor-pointer">{label}</label>
                     {description && <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{description}</p>}
                   </div>
                   <InteractiveToggle
                     isOn={state}
                     handleToggle={() => !isLoading && setState(!state)} // Prevent toggle when loading
                     labelId={id} // Associate toggle with label for accessibility
                   />
                 </div>
               ))}
            </div>

            {/* Other Core Settings */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="historyRetentionDays" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">History Retention (days):</label>
                <input
                  type="number"
                  id="historyRetentionDays"
                  value={historyRetentionDays}
                  onChange={(e) => setHistoryRetentionDays(Math.max(0, parseInt(e.target.value, 10) || 0))}
                  disabled={isLoading}
                  min="0"
                  className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">0 for indefinite.</p>
              </div>

              <div>
                <label htmlFor="responseDelayMs" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Simulated Response Delay (ms):</label>
                <input
                  type="number"
                  id="responseDelayMs"
                  value={responseDelayMs}
                  onChange={(e) => setResponseDelayMs(Math.max(0, parseInt(e.target.value, 10) || 0))}
                  disabled={isLoading}
                  min="0"
                  className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white"
                />
                 <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">0 for no delay.</p>
              </div>

              {/* Allowed Domains Textarea (conditionally shown) */}
              {summarizationEnabled && (
                <div className="col-span-1 md:col-span-2">
                  <label htmlFor="allowedScrapingDomains" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Allowed Domains for URL Summarization:</label>
                  <textarea
                    id="allowedScrapingDomains"
                    value={allowedScrapingDomains}
                    onChange={(e) => setAllowedScrapingDomains(e.target.value)}
                    disabled={isLoading}
                    rows={4}
                    placeholder="Enter allowed domains, one per line or comma-separated..."
                    className="mt-1 block w-full px-3 py-2 bg-white dark:bg-navy-700 border border-gray-300 dark:border-navy-600 rounded-md shadow-sm focus:outline-none focus:ring-brand-500 focus:border-brand-500 sm:text-sm text-gray-900 dark:text-white resize-vertical"
                  />
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Leave empty to disallow all URL scraping.</p>
                </div>
              )}
            </div>
          </div>
        </div> {/* End of Core Features Section */}

        {/* Save Button - Styled */}
        <div className="flex justify-end mt-8">
          <button
            type="submit"
            disabled={isLoading}
            className="inline-flex justify-center py-2 px-6 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-brand-500 hover:bg-brand-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-500 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-brand-400 dark:hover:bg-brand-500 dark:focus:ring-offset-navy-800"
          >
            {isLoading ? 'Saving...' : 'Save All Settings'}
          </button>
        </div>
      </form> {/* Form ends here */}

      {/* Remove the <style jsx> block */}
    </div>
  );
}

export default WidgetCustomizationSettings;