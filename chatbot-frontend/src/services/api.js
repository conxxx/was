import axios from 'axios';

// TODO: Use environment variable for API URL
const API_URL = 'http://localhost:5001/api'; // Assuming backend runs on 5001
export { API_URL }; // Export the base URL
/**
 * API Service for backend communication (Simplified Client ID Flow)
 */

const apiService = {
  /**
   * Login or create a user with email only.
   * Stores the clientId on success.
   * @param {string} email - User email
   * @returns {Promise<object>} - Promise resolving to response data (e.g., { client_id: '...' })
   */
  login: async (email) => {
    try {
      const payload = { email };
      const response = await axios.post(`${API_URL}/login`, payload);

      if (response.data && response.data.client_id) {
        localStorage.setItem('clientId', response.data.client_id);
        console.log('Client ID stored.');
      } else {
         console.error('Login response missing client_id:', response);
         throw new Error('Login failed: Invalid response from server.');
      }
      return response.data;
    } catch (error) {
      console.error('Login error:', error.response ? error.response.data : error.message);
      localStorage.removeItem('clientId');
      throw new Error(error.response?.data?.error || 'Login failed.');
    }
  },

  /**
   * Get details for a specific chatbot
   * @param {string|number} chatbotId - The ID of the chatbot
   * @returns {Promise<object>} - Promise resolving to chatbot details (potentially including sources)
   */
  getChatbotDetails: async (chatbotId) => {
    if (!chatbotId) {
      throw new Error('Chatbot ID is required to fetch details.');
    }
    const clientId = localStorage.getItem('clientId'); // Get clientId
    if (!clientId) {
      throw new Error('Not logged in.'); // Ensure user is logged in
    }
    try {
      // Add clientId as query parameter for verification
      const response = await axios.get(`${API_URL}/chatbots/${chatbotId}?client_id=${clientId}`);
      // TODO: Confirm if this response includes 'sources' array/object
      console.log(`API Request: Fetching details for chatbot ${chatbotId}`); // Logging
      console.log(`API Success: Fetched details for chatbot ${chatbotId}`, response.data); // Logging
      return response.data;
    } catch (error) {
      console.error('Get chatbot details error:', error.response ? error.response.data : error.message); // Logging
      if (error.response?.status === 403) {
         throw new Error('Unauthorized to view this chatbot.');
      }
      if (error.response?.status === 404) {
         throw new Error('Chatbot not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to fetch chatbot details.');
    }
  },

  /**
   * Logout user by removing the clientId
   */
  logout: () => {
    localStorage.removeItem('clientId');
    console.log('Client ID removed.');
  },

  /**
   * Get all chatbots for the current user (identified by stored clientId)
   * @returns {Promise<Array>} - Promise resolving to the list of chatbots
   */
  getChatbots: async () => {
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
       console.error('getChatbots error: clientId not found in localStorage.');
       throw new Error('Not logged in.');
    }
    try {
      const response = await axios.get(`${API_URL}/chatbots?client_id=${clientId}`);
      return response.data || [];
    } catch (error) {
      console.error('Get chatbots error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 404 && error.response?.data?.error === 'Invalid client_id') {
         apiService.logout();
         throw new Error('Invalid session. Please log in again.');
      }
       if (error.response?.status === 400 && error.response?.data?.error?.includes('client_id query parameter is required')) {
         apiService.logout();
         throw new Error('Session error. Please log in again.');
      }
      throw new Error('Failed to fetch chatbots.');
    }
  },

  /**
   * Create a new chatbot for the current user (identified by stored clientId)
   * @param {Object} chatbotData - Chatbot data (name, sources, etc.)
   * @returns {Promise<object>} - Promise resolving to the creation response (e.g., { chatbot_id: '...' })
   */
  createChatbot: async (chatbotData) => {
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
       console.error('createChatbot error: clientId not found in localStorage.');
       throw new Error('Not logged in.');
    }
    try {
      const formData = new FormData();
      formData.append('client_id', clientId);
      formData.append('name', chatbotData.name);

      formData.append('useUrlSource', chatbotData.useWebsite ? 'true' : 'false');
      if (chatbotData.useWebsite && chatbotData.websiteUrl) {
        formData.append('sourceValueUrl', chatbotData.websiteUrl);
      }
      formData.append('useSitemapSource', chatbotData.useSitemap ? 'true' : 'false');
      if (chatbotData.useSitemap && chatbotData.sitemapUrl) {
        formData.append('sourceValueSitemap', chatbotData.sitemapUrl);
      }
      formData.append('useFiles', chatbotData.useFiles ? 'true' : 'false');

      if (chatbotData.files && chatbotData.files.length > 0) {
         chatbotData.files.forEach(file => {
            formData.append('files', file);
         });
      }

      if (chatbotData.selectedUrls && chatbotData.selectedUrls.length > 0) {
         formData.append('selected_urls', JSON.stringify(chatbotData.selectedUrls));
      } else {
         if (chatbotData.useWebsite || chatbotData.useSitemap) {
             formData.append('selected_urls', '[]');
         }
      }

      const config = {
        headers: { 'Content-Type': 'multipart/form-data' }
      };

      const response = await axios.post(`${API_URL}/chatbots`, formData, config);
      // Store the API key upon successful creation
      if (response.data && response.data.chatbot_id && response.data.api_key) {
          localStorage.setItem(`chatbotApiKey_${response.data.chatbot_id}`, response.data.api_key);
          console.log(`API Key stored for chatbot ${response.data.chatbot_id}`);
      } else {
          console.warn("Chatbot created, but API key was missing in the response.");
      }
      return response.data;
    } catch (error) {
      console.error('Create chatbot error:', error.response ? error.response.data : error.message);
       if (error.response?.status === 404 && error.response?.data?.error === 'Invalid client_id') {
         apiService.logout();
         throw new Error('Invalid session. Please log in again.');
      } else if (error.response?.status === 400 && error.response?.data?.error?.includes('Maximum chatbots')) {
         throw new Error('Maximum chatbot limit reached.');
      }
      throw new Error('Failed to create chatbot.');
    }
  },

  /**
   * Send a query to a specific chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot to query.
   * @param {string} query - The user's query text.
   * @param {Array} [chat_history] - Optional array of previous chat messages
   * @param {string} [languageCode] - Optional BCP-47 language code for the query
   * @param {AbortSignal} [signal] - Optional AbortSignal for cancellation
   * @returns {Promise<object>} - Promise resolving to the chatbot's response (e.g., { response: '...' })
   */
  queryChatbot: async (chatbotId, query, chat_history = null, languageCode = null, signal = null) => { // Added signal parameter
    if (!chatbotId || !query) {
       throw new Error('Chatbot ID and query are required.');
    }
    try {
      // --- Retrieve Plaintext API Key from Local Storage ---
      const apiKey = localStorage.getItem(`chatbotApiKey_${chatbotId}`);

      if (!apiKey) {
        console.error(`QueryChatbot: API key not found in local storage for chatbot ${chatbotId}. Please ensure the chatbot was created successfully and the key was stored.`);
        // Note: This might happen if the user is trying to chat with a bot created before this fix was applied.
        throw new Error("Authentication key not found locally for this chatbot.");
      }
      // --- End Retrieve API Key ---

      const payload = { query };
      if (chat_history && Array.isArray(chat_history) && chat_history.length > 0) {
        payload.chat_history = chat_history;
      }
      // Add language code to payload if provided
      if (languageCode) {
        payload.query_language = languageCode;
      }

     // Add API Key header
     const headers = {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}` // Add the fetched API key
      };

      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/query`, payload, { headers, signal }); // Pass signal in config
      return response.data;
    } catch (error) {
      // Check if the error is due to cancellation first
      if (axios.isCancel(error) || error.name === 'AbortError') {
        console.log('API Service: Query chatbot request cancelled.');
        throw error; // Re-throw the cancellation error
      }
      // If not a cancellation error, proceed with existing error handling
      console.error('Query chatbot error:', error.response ? error.response.data : error.message, error);
      // TODO: Add more specific error handling based on status codes if needed
      throw new Error('Failed to get response from chatbot.');
    }
  },

  /**
   * Send a query and an image to a specific chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot to query.
   * @param {string} query - The user's query text (can be empty if only sending image).
   * @param {Array|null} chat_history - Optional array of previous chat messages.
   * @param {File} imageFile - The image file to send.
   * @param {AbortSignal} [signal] - Optional AbortSignal for cancellation
   * @returns {Promise<object>} - Promise resolving to the chatbot's response (e.g., { response: '...' })
   */
  queryChatbotWithImage: async (chatbotId, query, chat_history, imageFile, signal = null) => { // Added signal parameter
    if (!chatbotId || !imageFile) {
      throw new Error('Chatbot ID and image file are required for image query.');
    }
    // Query can be optional if only sending an image

    try {
      // --- Retrieve Plaintext API Key from Local Storage ---
      const apiKey = localStorage.getItem(`chatbotApiKey_${chatbotId}`);
      if (!apiKey) {
        console.error(`queryChatbotWithImage: API key not found for chatbot ${chatbotId}.`);
        throw new Error("Authentication key not found locally for this chatbot.");
      }
      // --- End Retrieve API Key ---

      const formData = new FormData();
      formData.append('query', query || ''); // Send query, even if empty

      if (chat_history && Array.isArray(chat_history) && chat_history.length > 0) {
        // Backend expects history as a JSON string within FormData
        formData.append('history', JSON.stringify(chat_history));
      } else {
        // Send empty array string if no history
         formData.append('history', '[]');
      }

      // Append the image file
      formData.append('image', imageFile, imageFile.name); // Key 'image', include filename

      const headers = {
        'Authorization': `Bearer ${apiKey}`,
        // 'Content-Type': 'multipart/form-data' // Axios sets this automatically for FormData
      };

      console.log(`Sending query with image to chatbot ${chatbotId}`);
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/query_with_image`, formData, { headers, signal }); // Pass signal in config
      console.log(`Received response for query with image from chatbot ${chatbotId}`);
      return response.data;

    } catch (error) {
      // Check if the error is due to cancellation first
      if (axios.isCancel(error)) {
        console.log('API Service: Query chatbot with image request cancelled.');
        throw error; // Re-throw the cancellation error
      }
      // If not a cancellation error, proceed with existing error handling
      console.error('Query chatbot with image error:', error.response ? error.response.data : error.message, error);
      // Check for structured error first (Plan Section 6.1)
      if (error.response?.data?.error && typeof error.response.data.error === 'object' && error.response.data.error.message) {
         console.log("Structured error received:", error.response.data.error);
         throw new Error(error.response.data.error.message); // Throw with the user-friendly message
      }
      // Fallback to existing status code checks or generic message
       else if (error.response?.status === 403) {
         // This could be IMAGE_ANALYSIS_DISABLED or invalid API key
         throw new Error(error.response?.data?.error || 'Unauthorized or image analysis disabled for this chatbot.');
      }
       else if (error.response?.status === 404) {
         throw new Error('Chatbot not found or image query endpoint unavailable.');
      }
       else if (error.response?.status === 400) {
         // Handle potential validation errors if not structured
         throw new Error(error.response?.data?.error || 'Bad request (e.g., invalid image, missing data).');
      }
       else {
          // Generic fallback
          throw new Error('Failed to get response from chatbot with image.');
       }
    }
  },

  /**
   * Start link discovery process
   * @param {object} discoveryData - { source_url: string, source_type: 'url'|'sitemap' }
   * @returns {Promise<object>} - Promise resolving to task info (e.g., { task_id: '...' })
   */
   discoverLinks: async (discoveryData) => {
    if (!discoveryData?.source_url || !discoveryData?.source_type) {
      throw new Error('Source URL and type are required for discovery.');
    }
    try {
      const response = await axios.post(`${API_URL}/discover-links`, discoveryData);
      return response.data;
    } catch (error) {
      console.error('Discover links error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to start link discovery.');
    }
  },

  /**
   * Get results of a discovery task
   * @param {string} taskId - The ID of the discovery task
   * @returns {Promise<object>} - Promise resolving to discovery results (e.g., { status: '...', discovered_urls: [...] })
   */
  getDiscoveryResults: async (taskId) => {
    if (!taskId) {
      throw new Error('Task ID is required to get discovery results.');
    }
    try {
      const response = await axios.get(`${API_URL}/discover-links/${taskId}`);
      return response.data;
    } catch (error) {
      console.error('Get discovery results error:', error.response ? error.response.data : error.message);
       if (error.response?.status === 404) {
         throw new Error('Discovery task not found or expired.');
       }
      throw new Error(error.response?.data?.error || 'Failed to get discovery results.');
    }
  },

  /**
   * Delete a specific chatbot
   * @param {string|number} chatbotId - The ID of the chatbot to delete
   * @param {string} clientId - The client ID for verification
   * @returns {Promise<object>} - Promise resolving to the deletion response
   */
  deleteChatbot: async (chatbotId, clientId) => { // Added clientId parameter
    if (!chatbotId) {
      throw new Error('Chatbot ID is required for deletion.');
    }
    if (!clientId) { // Verify clientId is passed
      throw new Error('Client ID is required for deletion verification.');
    }
    try {
      // Add clientId as query parameter for verification
      const response = await axios.delete(`${API_URL}/chatbots/${chatbotId}?client_id=${clientId}`);
      // Remove API key on successful deletion
      localStorage.removeItem(`chatbotApiKey_${chatbotId}`);
      console.log(`API Key removed for deleted chatbot ${chatbotId}`);
      return response.data;
    } catch (error) {
      console.error('Delete chatbot error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 403) {
         throw new Error('Unauthorized to delete this chatbot.');
      }
       if (error.response?.status === 404) {
         throw new Error('Chatbot not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to delete chatbot.');
    }
  },

  /**
   * Update an existing chatbot
   * @param {string|number} chatbotId - The ID of the chatbot to update
   * @param {Object} chatbotData - Updated chatbot data (name, sources, etc.)
   * @param {string} clientId - The client ID for verification
   * @returns {Promise<object>} - Promise resolving to the update response
   */
  updateChatbot: async (chatbotId, chatbotData, clientId) => { // Added clientId parameter
    if (!chatbotId) {
      throw new Error('Chatbot ID is required for update.');
    }
    if (!clientId) { // Verify clientId is passed
      throw new Error('Client ID is required for update verification.');
    }
    try {
      // The 'chatbotData' argument is already the FormData object prepared in the component
      // We just need to ensure the client_id is present if the backend requires it in the body
      // (It's already being appended in WidgetCustomizationSettings.jsx, so no need here unless backend changes)
      // if (!chatbotData.has('client_id')) { // Example check if needed
      //    chatbotData.append('client_id', clientId);
      // }

      chatbotData.append('client_id', clientId);

      const config = {
        headers: { 'Content-Type': 'multipart/form-data' }
      };

      // Send the FormData object received from the component
      const response = await axios.put(`${API_URL}/chatbots/${chatbotId}`, chatbotData, config);
      return response.data;
    } catch (error) {
      console.error('Update chatbot error:', error.response ? error.response.data : error.message);
       if (error.response?.status === 403) {
         throw new Error('Unauthorized to update this chatbot.');
      }
       if (error.response?.status === 404) {
         throw new Error('Chatbot not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to update chatbot.');
    }
  },

  /**
   * Update chatbot customization settings (widget colors, welcome message).
   * @param {string|number} chatbotId - The ID of the chatbot to update.
   * @param {object} customizationData - Object containing settings like { widget_primary_color, widget_text_color, widget_welcome_message }.
   * @returns {Promise<object>} - Promise resolving to the update response.
   */
  updateChatbotCustomization: async (chatbotId, customizationData) => {
    if (!chatbotId) {
      throw new Error('Chatbot ID is required for customization update.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in. Client ID is required for update verification.');
    }

    try {
      // Prepare FormData payload
      const formData = new FormData();
      formData.append('client_id', clientId); // Add clientId for verification

      // Append customization data if present
      if (customizationData.widget_primary_color) {
        formData.append('widget_primary_color', customizationData.widget_primary_color);
      }
      if (customizationData.widget_text_color) {
        formData.append('widget_text_color', customizationData.widget_text_color);
      }
      if (customizationData.widget_welcome_message) {
        formData.append('widget_welcome_message', customizationData.widget_welcome_message);
      }

      // Ensure at least one customization field is being sent
      if (!formData.has('widget_primary_color') && !formData.has('widget_text_color') && !formData.has('widget_welcome_message')) {
          console.warn('No customization data provided for update.');
          // Optionally return early or throw an error if no data is provided
          // return { message: 'No customization data provided.' };
      }

      // No explicit Content-Type header needed for FormData; browser sets it.
      const config = {};

      // Send FormData payload to the PUT endpoint
      const response = await axios.put(`${API_URL}/chatbots/${chatbotId}`, formData, config);
      return response.data;
    } catch (error) {
      console.error('Update chatbot customization error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 403) {
        throw new Error('Unauthorized to update this chatbot customization.');
      }
      if (error.response?.status === 404) {
        throw new Error('Chatbot not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to update chatbot customization.');
    }
  },


  /**
   * Create an SSE connection for chatbot status updates
   * Uses clientId in the query parameter as per backend design.
   * @param {string} clientId - Client ID
   * @param {Function} onMessage - Callback for messages
   * @returns {EventSource} - EventSource object
   */
  createStatusStream: (clientId, onMessage) => {
    if (!clientId) {
      console.error("Cannot create status stream without clientId.");
      return { close: () => { console.log("Dummy EventSource closed."); } };
    }

    console.log(`Creating EventSource for: ${API_URL}/chatbots/status-stream?clientId=${clientId}`);
    const eventSource = new EventSource(`${API_URL}/chatbots/status-stream?clientId=${clientId}`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing SSE message:', error, 'Raw data:', event.data);
      }
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
    };

    eventSource.onopen = () => {
      console.log("SSE connection opened.");
    };

    return eventSource;
  },

  /**
   * Synthesize speech from text using the backend TTS endpoint.
   * @param {string} text - The text to synthesize.
   * @param {AbortSignal} [signal] - Optional AbortSignal for cancellation.
   * @returns {Promise<Blob>} - Promise resolving to the audio blob (MP3).
   */
  synthesizeSpeech: async (chatbotId, text, selectedLanguage, signal = null) => {
    if (!text) {
      throw new Error('Text is required for speech synthesis.');
    }
    try {
      const payload = { text };
console.log('[api.js] synthesizeSpeech: Inspecting signal before axios call:', signal);
      console.log('[api.js] synthesizeSpeech: Does signal have addEventListener?', typeof signal?.addEventListener === 'function');
      const response = await axios.post(`${API_URL}/voice/tts`, payload, {
        responseType: 'blob', // Important: Expect binary data (audio blob)
        signal // Pass signal in config
      });
      return response.data; // The audio blob
    } catch (error) {
       if (axios.isCancel(error) || error.name === 'AbortError') {
         console.log('API Service: TTS request cancelled.');
         throw error; // Re-throw cancellation error
       }
      console.error('Synthesize speech error:', error.response ? error.response.data : error.message);
      // Attempt to read error message if the response was JSON (e.g., backend error)
      if (error.response && error.response.data instanceof Blob && error.response.data.type.includes('json')) {
         try {
            const errorJson = JSON.parse(await error.response.data.text());
            throw new Error(errorJson.error || 'TTS synthesis failed.');
         } catch (parseError) {
            console.error('Could not parse error response from TTS:', parseError);
            throw new Error('TTS synthesis failed and error response could not be parsed.');
         }
      }
      throw new Error('TTS synthesis failed.');
    }
  },

  /**
   * Send audio blob to backend for STT and chatbot interaction.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {Blob} audioBlob - The recorded audio blob.
   * @param {string} languageCode - The BCP-47 language code for STT.
   * @param {string} sessionId - The unique ID for the current chat session.
   * @param {string} apiKey - The API key for authentication.
   * @param {AbortSignal} [signal] - Optional AbortSignal for cancellation.
   * @returns {Promise<object>} - Promise resolving to { transcribed_input: '...', text_response: '...', audio_response_base64: '...' }
   */
  interactWithChatbotVoice: async (chatbotId, audioBlob, languageCode, sessionId, apiKey, signal = null) => { // Add apiKey and signal parameters
    if (!chatbotId || !audioBlob || !languageCode || !sessionId) {
      throw new Error('Chatbot ID, audio blob, language code, and session ID are required for voice interaction.');
    }
    if (!apiKey) { // Add check for apiKey
      console.error('interactWithChatbotVoice: apiKey was not provided.');
      throw new Error('API Key is required for voice interaction.');
    }
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice_input.webm'); // Filename is arbitrary but good practice
      formData.append('language', languageCode);
      formData.append('session_id', sessionId);

      // --- Use Standard Client ID Authentication ---
      const clientId = localStorage.getItem('clientId');
      if (!clientId) {
        console.error('interactWithChatbotVoice: clientId not found in localStorage.');
        throw new Error('Not logged in. Client ID is required for voice interaction.');
      }
      // --- End Client ID Authentication ---

      // --- Prepare Headers ---
      const headers = {
          'X-Client-ID': clientId,
          'Authorization': `Bearer ${apiKey}` // Add Authorization header
          // Axios handles 'Content-Type': 'multipart/form-data' automatically for FormData
      };
      // --- End Prepare Headers ---

      const response = await axios.post(`${API_URL}/voice/chatbots/${chatbotId}/interact`, formData, { headers, signal }); // Pass updated headers and signal
      return response.data; // Expects { response_text: '...', response_audio: '...' }
    } catch (error) {
       if (axios.isCancel(error) || error.name === 'AbortError') {
         console.log('API Service: Voice interaction request cancelled.');
         throw error; // Re-throw cancellation error
       }
      console.error('Voice interaction error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 401 || error.response?.status === 403) {
         throw new Error('Unauthorized: Invalid API key or insufficient permissions for voice interaction.');
      }
      if (error.response?.status === 404) {
         throw new Error('Chatbot not found or voice interaction endpoint unavailable.');
      }
      // Add specific error handling if needed (e.g., 403, 404)
      throw new Error(error.response?.data?.error || 'Failed voice interaction with chatbot.');
    }
  },

  /**
   * Add a URL source to an existing chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} url - The URL to add.
   * @returns {Promise<object>} - Promise resolving to the response.
   */
  addChatbotUrl: async (chatbotId, url) => {
    if (!chatbotId || !url) {
      throw new Error('Chatbot ID and URL are required.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      const payload = { client_id: clientId, url: url };
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/sources/url`, payload);
      return response.data;
    } catch (error) {
      console.error('Add chatbot URL error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to add URL source.');
    }
  },

  /**
   * Add a text source to an existing chatbot. (Will be removed later)
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} text - The text content to add.
   * @returns {Promise<object>} - Promise resolving to the response.
   */
  addChatbotText: async (chatbotId, text) => {
    if (!chatbotId || !text) {
      throw new Error('Chatbot ID and text are required.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      const payload = { client_id: clientId, text: text };
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/sources/texts`, payload);
      return response.data;
    } catch (error) {
      console.error('Add chatbot text error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to add text source.');
    }
  },

  /**
   * Add file sources to an existing chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {Array<File>} files - An array of File objects to upload.
   * @returns {Promise<object>} - Promise resolving to the response.
   */
  addChatbotFiles: async (chatbotId, files) => {
    if (!chatbotId || !files || files.length === 0) {
      throw new Error('Chatbot ID and at least one file are required.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      const formData = new FormData();
      formData.append('client_id', clientId);
      files.forEach(file => {
        formData.append('files', file); // Use 'files' as the key for the backend
      });

      const config = {
        headers: { 'Content-Type': 'multipart/form-data' }
      };

      // Assuming the backend endpoint is /chatbots/{id}/sources/files
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/sources/files`, formData, config);
      return response.data;
    } catch (error) {
      console.error('Add chatbot files error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to add file sources.');
    }
  },

  /**
   * Delete a specific data source (URL or Text) from a chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string|number} sourceId - The ID of the source to delete.
   * @returns {Promise<object>} - Promise resolving to the deletion response.
   */
  deleteChatbotSource: async (chatbotId, sourceIdentifier) => {
    if (!chatbotId || !sourceIdentifier) {
      throw new Error('Chatbot ID and Source Identifier are required for deletion.');
    }

    // Retrieve the chatbot-specific API key
    const apiKey = localStorage.getItem(`chatbotApiKey_${chatbotId}`);
    if (!apiKey) {
      console.error(`deleteChatbotSource: API key not found for chatbot ${chatbotId}.`);
      throw new Error("Authentication key not found locally for this chatbot.");
    }

    try {
      const url = `${API_URL}/chatbots/${chatbotId}/sources`;

      // Add API Key header and keep source_identifier in the data payload
      const headers = {
        'Authorization': `Bearer ${apiKey}`
      };

      const response = await axios.delete(url, {
        headers: headers,
        data: {
          source_identifier: sourceIdentifier // Backend needs this to know which source to delete
        }
      });

      // Handle successful response
      console.log('Delete chatbot source response:', response.data);
      return response.data;

    } catch (error) {
      console.error('Delete chatbot source error:', error.response ? error.response.data : error.message);
      // Adjust error handling based on API key usage
      if (error.response?.status === 401 || error.response?.status === 403) {
         // 401 likely means invalid key, 403 might mean key is valid but lacks permission (though less likely for source deletion)
         throw new Error('Unauthorized: Invalid API key or insufficient permissions to delete this source.');
      }
       if (error.response?.status === 404) {
         // Could be chatbot not found OR source_identifier not found for that chatbot
         throw new Error('Chatbot or Source not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to delete source.');
    }
  },

  /**
   * Regenerate the API key for a specific chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} clientId - The client ID for verification.
   * @returns {Promise<object>} - Promise resolving to the response containing the new API key.
   */
  regenerateChatbotApiKey: async (chatbotId, clientId) => {
    if (!chatbotId || !clientId) {
      throw new Error('Chatbot ID and Client ID are required to regenerate the API key.');
    }
    try {
      const payload = { client_id: clientId };
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/regenerate-key`, payload);
      return response.data; // Expects { new_api_key: '...' }
    } catch (error) {
      console.error('Regenerate API key error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 403) {
        throw new Error('Unauthorized to regenerate key for this chatbot.');
      }
      if (error.response?.status === 404) {
        throw new Error('Chatbot not found.');
      }
      throw new Error(error.response?.data?.error || 'Failed to regenerate API key.');
    }
  },

  /**
   * Get summarization settings for a chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @returns {Promise<object>} - Promise resolving to the summarization settings.
   */
  getSummarizationSettings: async (chatbotId) => {
    if (!chatbotId) throw new Error('Chatbot ID is required.');
    const clientId = localStorage.getItem('clientId');
    if (!clientId) throw new Error('Not logged in.');
    try {
      const response = await axios.get(`${API_URL}/chatbots/${chatbotId}/summarization-settings?client_id=${clientId}`);
      return response.data;
    } catch (error) {
      console.error('Get summarization settings error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to get summarization settings.');
    }
  },

  /**
   * Update summarization settings for a chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {object} settings - The settings object (e.g., { enabled, prompt }).
   * @returns {Promise<object>} - Promise resolving to the update response.
   */
  updateSummarizationSettings: async (chatbotId, settings) => {
    if (!chatbotId) throw new Error('Chatbot ID is required.');
    const clientId = localStorage.getItem('clientId');
    if (!clientId) throw new Error('Not logged in.');
    try {
      const payload = { ...settings, client_id: clientId };
      const response = await axios.put(`${API_URL}/chatbots/${chatbotId}/summarization-settings`, payload);
      return response.data;
    } catch (error) {
      console.error('Update summarization settings error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to update summarization settings.');
    } // Removed incorrect comma
  }, // Added correct comma after method definition

  // --- End Voice Service Functions ---

  /**
   * Sends content to the backend for summarization.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} contentType - 'text' or 'url'.
   * @param {string} content - The text or URL to summarize.
   * @param {string} targetLanguage - The desired language code for the summary.
   * @param {AbortSignal} [signal] - Optional AbortSignal for cancellation.
   * @returns {Promise<object>} - Promise resolving to the summarization result (e.g., { summary: '...', original_language: '...', target_language: '...' })
   */
  summarizeContent: async (chatbotId, contentType, content, targetLanguage, signal = null) => { // Added signal parameter
    if (!chatbotId || !contentType || !content || !targetLanguage) {
      throw new Error('Chatbot ID, content type, content, and target language are required for summarization.');
    }
    if (contentType !== 'text' && contentType !== 'url') {
        throw new Error("Invalid content type. Must be 'text' or 'url'.");
    }

    try {
      // Retrieve API Key from Local Storage
      const apiKey = localStorage.getItem(`chatbotApiKey_${chatbotId}`);
      if (!apiKey) {
        console.error(`summarizeContent: API key not found for chatbot ${chatbotId}.`);
        throw new Error("Authentication key not found locally for this chatbot.");
      }

      const payload = {
        content_type: contentType,
        content: content,
        target_language: targetLanguage,
        // source_language could be added here if detected client-side, but backend handles detection
      };

      const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      };

      console.log(`Sending content for summarization to chatbot ${chatbotId}`);
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/summarize`, payload, { headers, signal }); // Pass signal in config
      console.log(`Received summarization response from chatbot ${chatbotId}`);
      return response.data;

    } catch (error) {
       if (axios.isCancel(error) || error.name === 'AbortError') {
         console.log('API Service: Summarization request cancelled.');
         throw error; // Re-throw cancellation error
       }
      console.error('Summarize content error:', error.response ? error.response.data : error.message, error);
      // Handle specific errors based on status code or backend error structure
      if (error.response?.status === 403) {
         // Could be feature disabled or domain not allowed
         throw new Error(error.response?.data?.error || 'Summarization feature disabled or domain not allowed.');
      } else if (error.response?.status === 400) {
         throw new Error(error.response?.data?.error || 'Bad request (e.g., invalid URL, missing data).');
      } else if (error.response?.status === 502 || error.response?.status === 504) {
          throw new Error(error.response?.data?.error || 'Failed to retrieve content from URL.');
      } else {
         // Generic fallback
         throw new Error(error.response?.data?.error || 'Failed to get summary.');
      }
    }
  }, // Re-added trailing comma

  // --- Crawl Functions for Existing Chatbots ---

  /**
   * Start a crawl task for an existing chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} startUrl - The URL to start crawling from.
   * @returns {Promise<object>} - Promise resolving to task info (e.g., { task_id: '...' }).
   */
  startChatbotCrawl: async (chatbotId, startUrl) => {
    if (!chatbotId || !startUrl) {
      throw new Error('Chatbot ID and Start URL are required to start crawl.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      const payload = { client_id: clientId, start_url: startUrl };
      // Assuming endpoint: POST /api/chatbots/{chatbotId}/crawl/start
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/crawl/start`, payload);
      return response.data; // Expects { task_id: '...' }
    } catch (error) {
      console.error('Start chatbot crawl error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to start website crawl.');
    }
  },

  /**
   * Get the status and results of a crawl task for a specific chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {string} taskId - The ID of the crawl task.
   * @returns {Promise<object>} - Promise resolving to crawl status (e.g., { status: '...', result: { urls: [...] }, error: '...' }).
   */
  getCrawlStatus: async (chatbotId, taskId) => {
    if (!chatbotId || !taskId) {
      throw new Error('Chatbot ID and Task ID are required to get crawl status.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      // Assuming endpoint: GET /api/chatbots/{chatbotId}/crawl/status/{taskId}?client_id=...
      const response = await axios.get(`${API_URL}/chatbots/${chatbotId}/crawl/status/${taskId}?client_id=${clientId}`);
      return response.data;
    } catch (error) {
      console.error('Get crawl status error:', error.response ? error.response.data : error.message);
      if (error.response?.status === 404) {
         throw new Error('Crawl task not found or expired.');
      }
      throw new Error(error.response?.data?.error || 'Failed to get crawl status.');
    }
  },

  /**
   * Add selected URLs discovered during a crawl to an existing chatbot.
   * @param {string|number} chatbotId - The ID of the chatbot.
   * @param {Array<string>} urls - An array of URLs to add.
   * @returns {Promise<object>} - Promise resolving to the response.
   */
  addChatbotCrawledUrls: async (chatbotId, urls) => {
    if (!chatbotId || !urls || urls.length === 0) {
      throw new Error('Chatbot ID and a list of URLs are required.');
    }
    const clientId = localStorage.getItem('clientId');
    if (!clientId) {
      throw new Error('Not logged in.');
    }
    try {
      const payload = { client_id: clientId, selected_urls: urls };
      // Assuming endpoint: POST /api/chatbots/{chatbotId}/crawl/add-urls
      const response = await axios.post(`${API_URL}/chatbots/${chatbotId}/crawl/add-urls`, payload);
      return response.data;
    } catch (error) {
      console.error('Add crawled URLs error:', error.response ? error.response.data : error.message);
      throw new Error(error.response?.data?.error || 'Failed to add selected crawled URLs.');
    }
  },

}; // End of apiService object

export default apiService;
