import React, { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { v4 as uuidv4 } from 'uuid'; // For generating session ID
import axios from 'axios'; // Import axios
import apiService from '../services/api';
// TODO: Add styling imports later if needed
// TODO: Add CSS for markdown elements (code blocks, tables etc.)

// Supported languages for voice interaction
const supportedLanguages = [
  { code: 'am', name: 'Amharic' },
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
  { code: 'ar', name: 'Arabic' },
  { code: 'ru', name: 'Russian' },
  { code: 'uk', name: 'Ukrainian' },
  { code: 'es', name: 'Spanish' },
  { code: 'he', name: 'Hebrew' },
];

function ChatPage() {
  const { id: chatbotId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  // const [isLoading, setIsLoading] = useState(false); // For sending text messages - REPLACED by isProcessing
  const [isProcessing, setIsProcessing] = useState(false); // New state for any message sending process
  const currentMessageControllerRef = useRef(null); // Ref to hold the AbortController for cancellation
  const [loadingDetails, setLoadingDetails] = useState(true); // Combined loading state
  const [chatbotDetails, setChatbotDetails] = useState(null); // Store full chatbot details
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);
  const [ttsLoading, setTtsLoading] = useState(null); // Tracks which message ID's TTS is loading
  const [ttsError, setTtsError] = useState('');
  const audioRef = useRef(null); // Ref for playing TTS audio from text or voice response
  const ttsControllerRef = useRef(null); // Ref for TTS AbortController

  // --- Voice Interaction State ---
  const [isRecording, setIsRecording] = useState(false);
  const voiceApiControllerRef = useRef(null); // Ref for Voice API call AbortController
  const [selectedLanguage, setSelectedLanguage] = useState('en'); // Default to English
  const [voiceLoading, setVoiceLoading] = useState(false); // Loading state for voice interaction API call
  const [voiceError, setVoiceError] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [sessionId, setSessionId] = useState(''); // Unique ID for the chat session
  const [vadEnabled, setVadEnabled] = useState(false); // State for VAD setting
  // --- VAD State (AudioWorklet) ---
  const audioContextRef = useRef(null);
  const mediaStreamSourceRef = useRef(null);
  const vadNodeRef = useRef(null); // Ref for the AudioWorkletNode
  const vadStopInitiatedRef = useRef(false); // Flag to prevent multiple VAD stops
  // --- End VAD State ---
  // --- End Voice Interaction State ---

  // --- Image Analysis State ---
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  // const [isUploadingImage, setIsUploadingImage] = useState(false); // Loading state for image upload - REPLACED by isProcessing
  const fileInputRef = useRef(null); // Ref for the hidden file input
  // Removed sendButtonClickLockedRef
  // --- End Image Analysis State ---

  // --- Summarization State ---
  const [isSummarizeModalOpen, setIsSummarizeModalOpen] = useState(false);
  const [summarizeContentType, setSummarizeContentType] = useState('url'); // 'url' or 'text'
  const [summarizeContentInput, setSummarizeContentInput] = useState('');
  const [summarizeTargetLanguage, setSummarizeTargetLanguage] = useState('en'); // Default target language
  const [isSummarizing, setIsSummarizing] = useState(false); // Loading state for summarization API call
  const [summarizeError, setSummarizeError] = useState(''); // Error state for summarization modal
  const summarizationControllerRef = useRef(null); // Ref for Summarization AbortController
  // --- End Summarization State ---

  // --- Audio Playback State ---
  const [activeAudioMessageId, setActiveAudioMessageId] = useState(null); // ID of the message whose audio is playing/paused
  const [playbackState, setPlaybackState] = useState('stopped'); // 'playing', 'paused', 'stopped'
  const [currentAudioTime, setCurrentAudioTime] = useState(0);
  const [currentAudioDuration, setCurrentAudioDuration] = useState(0);
  // --- End Audio Playback State ---

  const [lastInputMethod, setLastInputMethod] = useState(null); // 'text', 'voice', or null

  // Effect for initial setup (runs once on mount or when chatbotId changes)
  useEffect(() => {
    console.log("[DEBUG] Initial setup useEffect running...");
    // Generate a unique session ID
    setSessionId(uuidv4());

    // Set initial welcome message ONLY if messages are currently empty
    setMessages(prevMessages => {
      if (prevMessages.length === 0) {
        console.log("[DEBUG] Setting initial welcome message.");
        return [{ id: 'welcome', content: 'Hello! How can I help you today?', isUser: false, timestamp: Date.now() }];
      }
      return prevMessages; // Don't change if messages already exist
    });

    // Fetch chatbot details
    if (chatbotId) {
      setLoadingDetails(true);
      apiService.getChatbotDetails(chatbotId) // clientId handled by service
        .then(data => {
          setChatbotDetails(data); // Store all details
          // Set initial welcome message from chatbot settings if available
          if (data.widget_settings?.widget_welcome_message) {
             setMessages([
               { id: 'welcome', content: data.widget_settings.widget_welcome_message, isUser: false, timestamp: Date.now() }
             ]);
          }
          // Set VAD enabled state
          setVadEnabled(data.voice_activity_detection_enabled ?? false); // Use the correct key from backend response
        })
        .catch(err => {
          console.error("Error fetching chatbot details:", err);
          setError(err.message || 'Failed to load chatbot details.');
           if (err.message.includes('Unauthorized') || err.message.includes('Not logged in')) {
             apiService.logout();
             navigate('/login');
          }
        })
        .finally(() => {
          setLoadingDetails(false);
        });
    } else {
       setLoadingDetails(false); // No ID, no loading
       setError('No Chatbot ID provided.');
    }

    // Cleanup function for this initial setup effect
    return () => {
      console.log("[DEBUG] Initial setup useEffect cleanup running...");
      // Stop recording if active (moved from previous combined cleanup)
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        console.log("Stopping media recorder during initial setup cleanup.");
        mediaRecorderRef.current.stop();
      }
      // Ensure VAD is stopped explicitly if it was active
      if (vadNodeRef.current) {
         console.log("Stopping VAD during initial setup cleanup.");
         stopVADWorklet(); // Use the new cleanup function
      }
      // Stop TTS audio if playing
      if (audioRef.current) {
        console.log("Pausing TTS audio during initial setup cleanup.");
        audioRef.current.pause();
        audioRef.current = null;
      }
      // Note: Image preview URL cleanup is handled in its dedicated effect
    };
  }, [chatbotId, navigate]); // Dependencies: chatbotId, navigate (imagePreviewUrl removed)

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Effect specifically for cleaning up the image preview object URL
  useEffect(() => {
    // This cleanup function runs when imagePreviewUrl changes OR when the component unmounts
    return () => {
      if (imagePreviewUrl) {
        console.log("[DEBUG] Revoking image preview Object URL:", imagePreviewUrl);
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]); // Only depends on imagePreviewUrl

  const handleSendMessage = async (e) => {
    e.preventDefault(); // Prevent default button action (though type="button")
    e.stopPropagation(); // Stop the event from bubbling up

    // --- Handle "Stop" Button Click ---
    // If already processing, this click is intended to abort the current request.
    if (isProcessing && currentMessageControllerRef.current) {
      console.log("ChatPage: 'Stop' button clicked. Aborting current request...");
      currentMessageControllerRef.current.abort();
      // State cleanup (isProcessing=false, currentMessageControllerRef=null)
      // will happen in the finally block of the *aborted* request.
      return; // Stop execution here, do not proceed to send a new message.
    }
    // ---------------------------------

    // --- Proceed with sending a NEW message ---
    // This part only runs if isProcessing was false.

    console.log("[ChatPage:handleSendMessage] Setting lastInputMethod to 'text'."); // LOG: lastInputMethod
    setLastInputMethod('text'); // Set input method for text/image queries
    // Set processing state *immediately* to prevent race conditions if triggered again quickly
    setIsProcessing(true);
    setError(''); // Clear previous errors

    const userQuery = input.trim();
    const imageFileToSend = selectedImageFile; // Store the image state at the start

    // Allow sending only an image without text
    if (!userQuery && !imageFileToSend) return; // Nothing to send

    // Construct user message content (text + image indicator if present)
    let userMessageContent = userQuery;
    if (selectedImageFile) {
        // Add a placeholder or indicator that an image was sent.
        // The actual image isn't displayed *in* the message bubble here,
        // but this confirms it was part of the submission.
        userMessageContent += `\n[Image: ${selectedImageFile.name}]`;
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      content: userMessageContent,
      isUser: true,
      timestamp: Date.now()
    };

    // Add user message and clear input
    // Only add if there's text or an image to show
    if (userMessageContent.trim()) {
        setMessages(prev => [...prev, userMessage]);
    }
    setInput('');
    // Don't clear the image here yet, wait for successful send or explicit cancel

    // Create a new AbortController for the current request
    const controller = new AbortController();
    currentMessageControllerRef.current = controller; // Store the new controller

    console.log("ChatPage: Initiating new query...");

    try {
      // --- Format Chat History ---
      // Exclude the initial welcome message and any error messages
      const historyToSend = messages
        .filter(msg => msg.id !== 'welcome' && !msg.id.startsWith('error-'))
        .map(msg => ({
          role: msg.isUser ? 'user' : 'assistant',
          content: msg.content
        }));
      // -------------------------

      let response;
      // --- API Call Logic ---
      if (imageFileToSend) { // Use the stored state
        // Call the new API endpoint for image uploads
        console.log("[DEBUG] Attempting to call queryChatbotWithImage API...");
        response = await apiService.queryChatbotWithImage(
          chatbotId,
          userQuery, // Send query even if empty, backend might handle it
          historyToSend,
          imageFileToSend, // Use the stored state
          controller.signal // Pass the signal
        );
        console.log("[DEBUG] queryChatbotWithImage API call successful. Response:", response);
      } else {
        // Call the existing endpoint for text-only queries
        console.log("Calling queryChatbot API");
        response = await apiService.queryChatbot(
            chatbotId,
            userQuery,
            historyToSend,
            null, // languageCode - assuming default for now
            controller.signal // Pass the signal
        );
        console.log("queryChatbot API response received");
      }
      // --------------------
      const botMessage = {
        id: `bot-${Date.now()}`,
        content: response.answer || response.response || "Sorry, I couldn't get a response.", // Check both possible keys
        isUser: false,
        timestamp: Date.now(),
        isVoiceResponse: false, // Explicitly not a voice response
        // TODO: Handle sources if backend provides them (response.sources)
      };
      console.log("[ChatPage:handleSendMessage] Created bot message, isVoiceResponse:", botMessage.isVoiceResponse); // LOG: isVoiceResponse
      setMessages(prev => [...prev, botMessage]);
      console.log("[DEBUG] Bot message added successfully.");

      // TTS for text/image responses is NOT auto-played here as per new requirement.
      // Manual playback controls will also NOT be shown due to isVoiceResponse: false.

    } catch (err) {
       // console.error("[DEBUG] Error caught in handleSendMessage:", err); // Suppress logging the full CanceledError object
       if (axios.isCancel(err) || err.name === 'AbortError') { // Check for axios cancel or native AbortError
            console.log('[DEBUG] Fetch aborted by user.'); // Keep this log to confirm cancellation
            // Optionally add a system message to the chat?
            // const abortMessage = { id: `abort-${Date.now()}`, content: "Request cancelled.", isUser: false, timestamp: Date.now() };
            // setMessages(prev => [...prev, abortMessage]);
       } else {
          console.log("[DEBUG] Handling non-abort error in handleSendMessage.");
          // Existing error handling logic...
          const displayError = err.message || 'Failed to send message.';
          setError(displayError); // Set the error state for potential display elsewhere
          const errorMessage = {
            id: `error-${Date.now()}`,
            content: `Error: ${displayError}`, // Display the potentially structured error message
            isUser: false,
            timestamp: Date.now()
          };
          setMessages(prev => [...prev, errorMessage]);
          console.log("[DEBUG] Error message added to chat.");
           // Handle session errors specifically if needed
          if (err.message.includes('Not logged in') || err.message.includes('Invalid session')) {
             console.log("[DEBUG] Session error detected, navigating to login.");
             apiService.logout();
             navigate('/login');
          } else {
             console.log("[DEBUG] Non-session error handled.");
          }
        } // <-- Added closing brace for 'else'
    } // <-- Added closing brace for 'catch'
    finally { // <-- Correctly positioned 'finally'
      console.log("[DEBUG] Entering finally block in handleSendMessage.");
      setIsProcessing(false); // Reset processing state
      currentMessageControllerRef.current = null; // Clear the controller ref
      // Clear the selected image and preview URL after the attempt
      // Always reset processing state and clear the controller ref in the finally block
      // The initial check in the function handles the abort logic.
      setIsProcessing(false);
      currentMessageControllerRef.current = null;
      // Only clear image if one was selected at the start of this function call
      if (imageFileToSend) {
        handleCancelImage(); // Use the cancel handler to ensure cleanup
        console.log("ChatPage: Cleared image state and processing status after API call.");
      } else {
        console.log("ChatPage: Cleared processing status after API call (no image involved).");
      }
    }
  };

  // Removed handleSendButtonClick wrapper function

  // --- VAD Functions (AudioWorklet) ---
  const handleVadMessage = (event) => {
    if (event.data.type === 'silenceDetected') {
      console.log('[ChatPage] Silence detected message received from VAD processor.');
      // Check the actual recorder state *and* the VAD stop flag
      if (mediaRecorderRef.current?.state === 'recording' && !vadStopInitiatedRef.current) {
        console.log('[ChatPage] Recorder state is "recording" and VAD stop not initiated, initiating stopRecording...');
        // console.log(`[handleVadMessage] MediaRecorder state before stop: ${mediaRecorderRef.current?.state}`); // Log is redundant now
        vadStopInitiatedRef.current = true; // Mark VAD stop as initiated
        // DO NOT set setIsRecording(false) here. Let onstop handle it.
        stopRecording(); // Call stopRecording
      } else {
        // Log why it was ignored, including the actual recorder state
        console.log(`[ChatPage] VAD silence detected but ignored. Recorder State: ${mediaRecorderRef.current?.state}, VAD Stop Initiated: ${vadStopInitiatedRef.current}`);
      }
    } else if (event.data.type === 'error') {
      console.error('[ChatPage] Error message received from VAD processor:', event.data.detail);
      setVoiceError(`VAD Processor Error: ${event.data.detail}`);
      stopVADWorklet(); // Attempt cleanup on processor error
    }
  };

  const startVADWorklet = async (stream) => {
    if (!vadEnabled || vadNodeRef.current) return; // Only start if enabled and not already active

    console.log("[ChatPage] Starting VAD Worklet...");
    try {
      // Ensure AudioContext is initialized and resumed
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // Load the processor module
      try {
        // Use absolute path from root for Vite compatibility
        await audioContextRef.current.audioWorklet.addModule('/src/vad-processor.js');
        console.log("[ChatPage] VAD processor module added.");
      } catch (moduleError) {
        console.error("[ChatPage] Error adding VAD worklet module:", moduleError);
        setVoiceError(`Failed to load VAD module: ${moduleError.message}`);
        return; // Stop if module fails to load
      }

      // Create the AudioWorkletNode
      vadNodeRef.current = new AudioWorkletNode(audioContextRef.current, 'vad-processor');
      console.log("[ChatPage] VAD AudioWorkletNode created.");

      // Set up message listener
      vadNodeRef.current.port.onmessage = handleVadMessage;
      vadNodeRef.current.onprocessorerror = (event) => {
        // This handles errors thrown *inside* the processor's process method
        console.error('[ChatPage] VAD processor internal error:', event);
        setVoiceError('VAD processor encountered an internal error.');
        stopVADWorklet(); // Attempt cleanup
      };

      // Create media stream source if it doesn't exist
      if (!mediaStreamSourceRef.current) {
         mediaStreamSourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
         console.log("[ChatPage] MediaStreamSource created.");
      }

      // Connect the source to the VAD node
      mediaStreamSourceRef.current.connect(vadNodeRef.current);
      // No need to connect VAD node to destination for analysis only
      console.log("[ChatPage] MediaStreamSource connected to VAD node.");

      console.log("[ChatPage] VAD Worklet Started Successfully.");

    } catch (error) {
      console.error("[ChatPage] Error starting VAD Worklet:", error);
      setVoiceError(`VAD Start Error: ${error.message}`);
      stopVADWorklet(); // Attempt cleanup if start failed
    }
  };

  const stopVADWorklet = () => {
    if (!vadNodeRef.current) return; // Only stop if active
    console.log("[ChatPage] Stopping VAD Worklet...");

    try {
      // Clean up message listener and error handler
      vadNodeRef.current.port.onmessage = null;
      vadNodeRef.current.onprocessorerror = null;

      // Disconnect the source node from the VAD node
      if (mediaStreamSourceRef.current) {
        try {
           mediaStreamSourceRef.current.disconnect(vadNodeRef.current);
           console.log("[ChatPage] Disconnected MediaStreamSource from VAD node.");
        } catch (disconnectError) {
           console.warn("[ChatPage] Error disconnecting MediaStreamSource (might already be disconnected):", disconnectError);
        }
        // Don't nullify mediaStreamSourceRef here if the stream might be reused by MediaRecorder
        // It gets stopped/cleaned in MediaRecorder.onstop
      }

      // Close the port
      vadNodeRef.current.port.close();
      console.log("[ChatPage] VAD node port closed.");

    } catch (cleanupError) {
        console.error("[ChatPage] Error during VAD Worklet cleanup:", cleanupError);
    } finally {
        vadNodeRef.current = null; // Nullify the ref
        console.log("[ChatPage] VAD Worklet Stopped.");
        // Consider closing AudioContext only when completely done (e.g., component unmount)
        // if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        //   audioContextRef.current.close().then(() => console.log("AudioContext closed."));
        //   audioContextRef.current = null;
        // }
    }
  };
  // --- End VAD Functions ---

  // --- Voice Recording Handlers ---
  const startRecording = async () => {
    if (isRecording) return; // Prevent starting if already recording

    vadStopInitiatedRef.current = false; // Reset VAD stop flag for new recording

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorderRef.current = new MediaRecorder(stream);
        audioChunksRef.current = []; // Clear previous chunks

        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorderRef.current.onstop = async () => {
          console.log("MediaRecorder stopped.");
          // Set recording state to false definitively here
          setIsRecording(false);

          // Ensure all tracks are stopped first
          stream.getTracks().forEach(track => track.stop());
          console.log("Microphone stream tracks stopped.");

          // Stop VAD Worklet now that tracks are stopped
          stopVADWorklet(); // Use new cleanup function

          // Clean up the MediaStreamSource node as the stream is now stopped
          if (mediaStreamSourceRef.current) {
              try {
                  mediaStreamSourceRef.current.disconnect(); // Disconnect from any remaining connections
              } catch (e) {
                  console.warn("Error disconnecting MediaStreamSource in onstop:", e);
              }
              mediaStreamSourceRef.current = null;
              console.log("Cleaned up MediaStreamSource node.");
          }

          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' }); // Adjust type if needed
          const hasAudioData = audioBlob.size > 0;

          // Only proceed if there's actual audio data
          if (!hasAudioData) {
              console.warn("No audio data recorded, skipping API call.");
              // Ensure voice loading is false if no data
              if (voiceLoading) setVoiceLoading(false);
              return;
          }

          // Set loading state for API call
          if (hasAudioData) { // This check is slightly redundant now but safe
              setVoiceLoading(true);
              setVoiceError('');
          }
          // setIsRecording(false); // Moved to the top of onstop

          // Only proceed if there's actual audio data
          if (!hasAudioData) {
              console.warn("No audio data recorded, skipping API call.");
              return;
          }

          // Create and store AbortController for the API call
          const controller = new AbortController();

          voiceApiControllerRef.current = controller;
          console.log("Initiating voice API call...");
          console.log("[ChatPage:mediaRecorder.onstop] Setting lastInputMethod to 'voice'."); // LOG: lastInputMethod
          setLastInputMethod('voice'); // Set input method for voice queries
          const currentInteractionIsVoice = true; // Local flag for this specific interaction
          console.log("[ChatPage:mediaRecorder.onstop] Set local flag currentInteractionIsVoice to true."); // LOG: Local flag

          try {
            // --- Retrieve Plaintext API Key from Local Storage ---
            const apiKey = localStorage.getItem(`chatbotApiKey_${chatbotId}`);
            console.log("[DEBUG] Retrieved API Key in onstop:", apiKey ? 'Found' : 'NOT FOUND'); // Log API key status
            if (!apiKey) {
              console.error(`ChatPage: API key not found in local storage for chatbot ${chatbotId}.`);
              setVoiceError("Authentication key not found locally. Cannot initiate voice chat.");
              setVoiceLoading(false);
              return; // Stop execution if key is missing
            }
            // --- End Retrieve API Key ---

            console.log("[DEBUG] Audio Blob size:", audioBlob.size); // Log blob size

            // Call the API endpoint, now including the correct API key and signal
            const response = await apiService.interactWithChatbotVoice(
              chatbotId,
              audioBlob,
              selectedLanguage,
              sessionId, // Pass the session ID
              apiKey, // Pass the retrieved plaintext API key
              controller.signal // Pass the signal
            );

            // Add transcribed input as a user message
            if (response.transcribed_input) {
              const userVoiceMessage = {
                id: `user-voice-${Date.now()}`, // Unique ID for the message
                content: response.transcribed_input,
                isUser: true, // Mark as a user message
                timestamp: Date.now()
              };
              setMessages(prev => [...prev, userVoiceMessage]); // Add to message history
            }

            // Add bot's text response to chat
            const botTextMessage = {
              id: `bot-voice-text-${Date.now()}`,
              content: response.text_response || "Sorry, I couldn't process the voice input.",
              isUser: false,
              timestamp: Date.now(),
              isVoiceResponse: true, // This is a response to a voice input
            };
            console.log("[ChatPage:mediaRecorder.onstop] Created bot message, isVoiceResponse:", botTextMessage.isVoiceResponse); // LOG: isVoiceResponse
            setMessages(prev => [...prev, botTextMessage]);

            // Handle audio response
            if (response.audio_response_base64) {
              // Use the local flag currentInteractionIsVoice for the auto-play decision
              console.log(`[ChatPage:mediaRecorder.onstop] Checking conditions for auto-play. currentInteractionIsVoice: ${currentInteractionIsVoice}, audio_response_base64 exists: true`); // LOG: Auto-play check
              if (currentInteractionIsVoice) { // <--- CHANGED TO USE LOCAL FLAG
                console.log("[ChatPage:mediaRecorder.onstop] Auto-playing TTS response as currentInteractionIsVoice is true. Message ID:", botTextMessage.id); // LOG: Auto-play action
                playBase64Audio(response.audio_response_base64, botTextMessage.id);
              } else {
                // This else block should now definitely not be hit in this path.
                console.log(`[ChatPage:mediaRecorder.onstop] TTS response received, but NOT auto-playing because currentInteractionIsVoice was false. This is highly unexpected for a voice response path.`);
              }
            } else {
              console.log("[ChatPage:mediaRecorder.onstop] No audio_response_base64 in response. Cannot auto-play TTS."); // LOG: No audio data
            }

          } catch (err) {
             if (axios.isCancel(err) || err.name === 'AbortError') {
               console.log('Voice interaction API request cancelled.');
               // Clear error state if cancellation was intentional
               setVoiceError('');
             } else {
               console.error("Voice interaction API error (Full Error):", err); // Log the full error object
               const errorMsg = err.response?.data?.error || err.message || 'Failed to interact with chatbot via voice.';
               setVoiceError(errorMsg);
               const errorMessage = {
                 id: `error-voice-${Date.now()}`,
                 content: `Voice Error: ${err.message || 'Could not process voice input.'}`,
                 isUser: false,
                 timestamp: Date.now()
               };
               setMessages(prev => [...prev, errorMessage]);
             }
          } finally {
            setVoiceLoading(false);
            // Clear the controller ref if it matches the one used for this request
            if (voiceApiControllerRef.current === controller) {
                voiceApiControllerRef.current = null;
            }
          }
        };

        // Start recording
        mediaRecorderRef.current.start();
        setIsRecording(true);
        setVoiceError(''); // Clear previous errors
        console.log("MediaRecorder started.");

        // Start VAD Worklet if enabled
        if (vadEnabled) {
          await startVADWorklet(stream); // Use the new async function
        }

      } catch (err) {
        console.error("Error accessing microphone:", err);
        setVoiceError(`Could not start recording: ${err.message}`);
        setIsRecording(false);
        // Ensure VAD Worklet is stopped if an error occurred during startup
        stopVADWorklet();
      }
    } else {
      setVoiceError("Audio recording is not supported by your browser.");
    }
  };

  const stopRecording = (manualStop = false) => { // Add flag for manual stop
    console.log(`[stopRecording] Called. Manual: ${manualStop}, Recorder State: ${mediaRecorderRef.current?.state}, VAD Stop Initiated: ${vadStopInitiatedRef.current}`);

    // Prevent redundant stop calls if already stopping or inactive
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
        console.log("[stopRecording] Ignored: Recorder not active or not initialized.");
        // Ensure VAD is stopped if called unexpectedly while not recording
        if (!isRecording) { // Check state variable as well
             stopVADWorklet();
        }
        return;
    }

    // If VAD initiated the stop, and this call isn't manual, don't call stop() again
    // (The first call from handleVadMessage should be sufficient)
    // However, we still need the manualStop logic to set the flag correctly.
    // Let's simplify: always set the flag if manual, and always call stop() if recording.
    // The onstop handler is the single source of truth for cleanup and state change.

    if (manualStop) {
        console.log("[stopRecording] Manual stop: Setting VAD stop initiated flag.");
        vadStopInitiatedRef.current = true; // Ensure VAD doesn't interfere
    }

    // Explicitly check state before calling stop()
    if (mediaRecorderRef.current.state === 'recording') {
        console.log("[stopRecording] Recorder is 'recording', calling mediaRecorderRef.current.stop().");
        mediaRecorderRef.current.stop(); // This triggers the onstop handler
    } else {
        console.log(`[stopRecording] Ignored stop() call as recorder state is '${mediaRecorderRef.current.state}'.`);
    }
    // VAD Worklet cleanup and setIsRecording(false) are handled within onstop

    // Removed the 'else' block as the initial check handles non-recording states.
    // The explicit stopVADWorklet call in the initial check handles cases where
    // stopRecording might be called erroneously when not recording.
    /*
    } else {
       console.log("Stop recording called, but not currently recording.");
       // Explicitly stop VAD Worklet here too, in case it got stuck active without recording
       stopVADWorklet();
       // Ensure recording state is false if we reach here unexpectedly
       if (isRecording) {
           setIsRecording(false);
       }
    }
    */
    // Erroneous lines removed here
  };

  const handleMicButtonClick = () => {
    // If currently processing the *API call* after recording, abort it
    if (voiceLoading && voiceApiControllerRef.current) {
        console.log("Voice API call in progress. Aborting...");
        voiceApiControllerRef.current.abort();
        // State reset happens in the finally block of the API call
    }
    // If currently recording, stop the recording manually
    else if (isRecording) {
      stopRecording(true); // Pass true for manual stop
    }
    // Otherwise, start recording
    else {
      startRecording();
    }
  };
  // --- End Voice Recording Handlers ---

  // --- Time Formatting Helper ---
  const formatTime = (timeInSeconds) => {
    if (isNaN(timeInSeconds) || timeInSeconds === Infinity) {
      return '0:00';
    }
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };
  // --- End Time Formatting Helper ---

  // --- Audio Playback Helper (for Base64) ---
  // TODO: Integrate this with handlePlayTTS to avoid duplication
  // --- START REPLACEMENT: playBase64Audio ---
  const playBase64Audio = (base64String, messageId = `voice-response-${Date.now()}`) => {
      console.log(`[ChatPage:playBase64Audio] Entered. messageId: ${messageId}, base64String provided: ${!!base64String}`); // LOG: Entry
      if (!base64String) {
          console.error("[ChatPage:playBase64Audio] No base64 audio string provided."); // LOG: Error
          setTtsError('No audio data received.');
          setTtsLoading(null); // Clear loading for this message
          return;
      }

      try {
          // Stop any currently playing audio
          if (audioRef.current) {
              console.log("[ChatPage:playBase64Audio] Existing audio player found. Pausing and cleaning up."); // LOG: Cleanup
              audioRef.current.pause();
              // Remove previous listeners
              audioRef.current.onloadedmetadata = null;
              audioRef.current.ontimeupdate = null;
              audioRef.current.onended = null;
              audioRef.current.onerror = null;
              audioRef.current.onplay = null;
              audioRef.current.onpause = null;
          }
          // Reset state before creating new player
          console.log("[ChatPage:playBase64Audio] Resetting playback state."); // LOG: State Reset
          setActiveAudioMessageId(null);
          setPlaybackState('stopped');
          setCurrentAudioTime(0);
          setCurrentAudioDuration(0);

          const byteCharacters = atob(base64String);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
              byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: 'audio/mpeg' }); // Assuming MP3, adjust if different
          const audioUrl = URL.createObjectURL(blob);

          console.log(`[ChatPage:playBase64Audio] Created new Audio object for messageId: ${messageId} with URL: ${audioUrl}`); // LOG: Action
          const newPlayer = new Audio(audioUrl);
          audioRef.current = newPlayer; // Assign to ref

          // --- Attach Event Listeners ---
          newPlayer.onloadedmetadata = () => {
              console.log(`[ChatPage:onloadedmetadata] Triggered for messageId: ${messageId}. Duration: ${newPlayer.duration}`); // LOG: Event
              if (audioRef.current === newPlayer) { // Check if still the active player
                  console.log(`[ChatPage:onloadedmetadata] Updating state for ${messageId}.`); // LOG: State Update
                  setCurrentAudioDuration(newPlayer.duration);
                  setActiveAudioMessageId(messageId); // Set active ID now
                  setCurrentAudioTime(0);
                  setTtsLoading(null); // Clear loading for this message
              } else {
                   console.log(`[ChatPage:onloadedmetadata] Event for ${messageId}, but player has changed. Ignoring.`); // LOG: Stale Event
              }
          };

          newPlayer.ontimeupdate = () => {
              // console.log(`[ChatPage:ontimeupdate] Triggered for messageId: ${messageId}. Time: ${newPlayer.currentTime}`); // LOG: Event (noisy)
              if (audioRef.current === newPlayer) {
                  setCurrentAudioTime(newPlayer.currentTime);
              }
          };

          newPlayer.onplay = () => {
              console.log(`[ChatPage:onplay] Triggered for messageId: ${messageId}`); // LOG: Event
              if (audioRef.current === newPlayer) {
                  console.log(`[ChatPage:onplay] Updating state to 'playing' for ${messageId}.`); // LOG: State Update
                  setPlaybackState('playing');
                  setActiveAudioMessageId(messageId); // Ensure active ID
              } else {
                   console.log(`[ChatPage:onplay] Event for ${messageId}, but player has changed. Ignoring.`); // LOG: Stale Event
              }
          };

          newPlayer.onpause = () => {
              // Log when the pause event is triggered for a specific message
              console.log(`[ChatPage:onpause] Triggered for messageId: ${messageId}.`);
              
              // Check if the event is for the currently active audio player instance
              if (audioRef.current === newPlayer) {
                  // If it's the current player, update the state to 'paused'
                  console.log(`[ChatPage:onpause] Updating state to 'paused' for ${messageId}.`);
                  setPlaybackState('paused');
              } else {
                  // If the player has changed since this handler was attached, ignore the event
                  console.log(`[ChatPage:onpause] Event for ${messageId}, but player has changed. Ignoring state update.`);
              }
          };

          newPlayer.onended = () => {
              console.log(`[ChatPage:onended] Triggered for messageId: ${messageId}`); // LOG: Event
              if (audioRef.current === newPlayer) {
                  console.log(`[ChatPage:onended] Updating state to 'stopped' and cleaning up for ${messageId}.`); // LOG: State Update & Cleanup
                  setPlaybackState('stopped');
                  setActiveAudioMessageId(null);
                  setCurrentAudioTime(0);
                  // Don't reset duration here
                  URL.revokeObjectURL(audioUrl); // Clean up the object URL
                  console.log(`[ChatPage:onended] Revoked Object URL: ${audioUrl}`); // LOG: Cleanup Detail
                  audioRef.current = null; // Clear the ref
              } else {
                   console.log(`[ChatPage:onended] Event for ${messageId}, but player has changed. Ignoring.`); // LOG: Stale Event
              }
          };

          newPlayer.onerror = (e) => {
              console.error(`[ChatPage:onerror] Triggered for messageId: ${messageId}. Error:`, e); // LOG: Event & Error
              if (audioRef.current === newPlayer) {
                  console.log(`[ChatPage:onerror] Updating state to 'stopped' and cleaning up due to error for ${messageId}.`); // LOG: State Update & Cleanup
                  setTtsError(`Error playing audio: ${e.message || 'Unknown error'}`);
                  setPlaybackState('stopped');
                  setActiveAudioMessageId(null);
                  setCurrentAudioTime(0);
                  setCurrentAudioDuration(0); // Reset duration on error
                  URL.revokeObjectURL(audioUrl);
                  console.log(`[ChatPage:onerror] Revoked Object URL: ${audioUrl}`); // LOG: Cleanup Detail
                  audioRef.current = null;
                  setTtsLoading(null); // Clear loading for this message
              } else {
                   console.log(`[ChatPage:onerror] Event for ${messageId}, but player has changed. Ignoring.`); // LOG: Stale Event
              }
          };
          // --- End Event Listeners ---

          // Attempt to play
          console.log(`[ChatPage:playBase64Audio] Calling play() for messageId ${messageId}`); // LOG: Action
          newPlayer.play().catch(e => {
              console.error(`[ChatPage:playBase64Audio] Error starting playback for ${messageId}:`, e); // LOG: Error
              setTtsError(`Error starting audio: ${e.message || 'Unknown error'}`);
              if (audioRef.current === newPlayer) { // Check if it's still the same player before resetting
                  console.log(`[ChatPage:playBase64Audio] Resetting state after play() failed for ${messageId}.`); // LOG: State Reset
                  setPlaybackState('stopped');
                  setActiveAudioMessageId(null);
                  setCurrentAudioTime(0);
                  setCurrentAudioDuration(0);
                  URL.revokeObjectURL(audioUrl);
                  audioRef.current = null;
                  setTtsLoading(null); // Clear loading for this message
              }
          });

      } catch (error) {
          console.error("[ChatPage:playBase64Audio] Error processing base64 audio:", error); // LOG: Error
          setTtsError('Failed to process audio data.');
          setTtsLoading(null); // Clear loading for this message
      }
      console.log(`[ChatPage:playBase64Audio] Exiting for messageId: ${messageId}`); // LOG: Exit
  };
  // --- END REPLACEMENT: playBase64Audio ---
  // --- End Audio Playback Helper ---

  // --- Text-to-Speech Handler ---
  // --- START REPLACEMENT: handlePlayTTS ---
  const handlePlayTTS = async (messageId, text) => {
    console.log(`[ChatPage] handlePlayTTS called for messageId: ${messageId}`);
    setTtsLoading(messageId); // Set loading state for this specific message
    setTtsError(''); // Clear previous TTS errors
    setActiveAudioMessageId(null); // Stop any currently playing audio
    setPlaybackState('stopped');
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // Abort previous TTS request if any
    if (ttsControllerRef.current) {
      ttsControllerRef.current.abort();
      console.log("[ChatPage] Aborted previous TTS request.");
    }

    // Create a new AbortController for the current TTS request
    const controller = new AbortController();
    ttsControllerRef.current = controller;

    try {
      console.log(`[ChatPage] Requesting TTS for text: "${text.substring(0, 50)}..."`);
      const response = await apiService.synthesizeSpeech(
        chatbotId,
        text,
        selectedLanguage, // Use the selected language
        controller.signal // Pass the signal
      );

      // --- FIXED BLOB HANDLING ---
      console.log(`[ChatPage] TTS API Response received. Type: ${response?.constructor?.name}`); // Log the type of response

      if (response instanceof Blob) {
        console.log("[ChatPage] Received Blob data, converting to base64...");
        const reader = new FileReader();
        reader.onloadend = () => {
          try {
            const base64DataUrl = reader.result;
            // Example data URL: "data:audio/mpeg;base64,SUQzBAAAAA..."
            // Remove the prefix to get only the base64 string
            const base64String = base64DataUrl?.split(',')[1]; // Use optional chaining for safety

            if (base64String) {
              console.log("[ChatPage] Blob converted successfully, playing audio.");
              playBase64Audio(base64String, messageId);
            } else {
              console.error("[ChatPage] Failed to extract base64 string from data URL. Data URL:", base64DataUrl);
              setTtsError('Failed to process audio data (invalid format).');
              setTtsLoading(null); // Clear loading on processing error
            }
          } catch (e) {
             console.error("[ChatPage] Error processing FileReader result:", e);
             setTtsError('Failed to process audio data.');
             setTtsLoading(null); // Clear loading on processing error
          }
        };
        reader.onerror = (error) => {
          console.error("[ChatPage] FileReader error converting Blob:", error);
          setTtsError('Failed to read audio data.');
          setTtsLoading(null); // Clear loading on reader error
        };
        reader.readAsDataURL(response); // Start the conversion
      } else {
        console.error("[ChatPage] TTS response was not a Blob as expected. Received:", response);
        setTtsError('Received unexpected audio data format from the server.');
        setTtsLoading(null); // Clear loading if response is not a Blob
      }
      // --------------------------

    } catch (err) {
      // Use axios.isCancel for Axios cancellations, err.name === 'AbortError' for native fetch AbortError
      if (axios.isCancel(err) || err.name === 'AbortError') {
        console.log('[ChatPage] TTS request aborted.');
        setTtsLoading(null); // Clear loading on abort
        // No user-facing error needed for abort
      } else {
        console.error("[ChatPage] Error fetching TTS:", err);
        setTtsError(err.response?.data?.message || err.message || 'Failed to synthesize speech.'); // Prefer server error message
        setTtsLoading(null); // Clear loading on fetch error
      }
    } finally {
      // Note: setTtsLoading(null) is now handled within the async branches (onloadend, onerror, catch)
      // to ensure it's cleared *after* processing is done or fails.
      ttsControllerRef.current = null; // Clear the controller ref regardless
    }
  };
  // --- END REPLACEMENT: handlePlayTTS ---
  // --- End Text-to-Speech Handler ---

   // --- Playback Control Handlers ---
   // --- START REPLACEMENT: handlePlayPauseClick ---
   const handlePlayPauseClick = (messageId, text) => {
       console.log(`[ChatPage:handlePlayPauseClick] Entered. messageId: ${messageId}, activeAudioMessageId: ${activeAudioMessageId}, playbackState: ${playbackState}`); // LOG: Entry & State
       if (activeAudioMessageId === messageId) {
           // Audio for this message is active (playing or paused)
           if (playbackState === 'playing') {
               console.log(`[ChatPage:handlePlayPauseClick] Pausing audio for messageId: ${messageId}`); // LOG: Action
               audioRef.current?.pause();
               // State update ('paused') handled by onpause listener
           } else if (playbackState === 'paused' || playbackState === 'stopped') {
                // If paused or stopped (but player exists for this ID), try playing
               if (audioRef.current) {
                    console.log(`[ChatPage:handlePlayPauseClick] Resuming/Playing audio for messageId: ${messageId}`); // LOG: Action
                    audioRef.current.play().catch(e => console.error(`[ChatPage:handlePlayPauseClick] Error resuming playback:`, e)); // LOG: Error
                    // State update ('playing') handled by onplay listener
               } else {
                    // Player doesn't exist, need to fetch TTS again (treat as initial play)
                    console.log(`[ChatPage:handlePlayPauseClick] Player not found for paused/stopped state. Fetching TTS again for messageId: ${messageId}`); // LOG: Action
                    handlePlayTTS(messageId, text);
               }
           }
       } else {
           // No audio active, or different audio active. Start TTS for this message.
           console.log(`[ChatPage:handlePlayPauseClick] No active audio or different message active. Fetching TTS for messageId: ${messageId}`); // LOG: Action
           handlePlayTTS(messageId, text);
       }
        console.log(`[ChatPage:handlePlayPauseClick] Exiting for messageId: ${messageId}`); // LOG: Exit
   };
   // --- END REPLACEMENT: handlePlayPauseClick ---

   // --- START REPLACEMENT: handleStopClick ---
   const handleStopClick = () => {
       console.log(`[ChatPage:handleStopClick] Entered. activeAudioMessageId: ${activeAudioMessageId}`); // LOG: Entry & State
       if (audioRef.current && activeAudioMessageId) {
           console.log(`[ChatPage:handleStopClick] Stopping audio for messageId: ${activeAudioMessageId}`); // LOG: Action
           audioRef.current.pause();
           audioRef.current.currentTime = 0;
           // State updates (stopped, null activeId, 0 time) handled by onended/onerror or manually here
           setPlaybackState('stopped'); // Explicitly set state
           setActiveAudioMessageId(null);
           setCurrentAudioTime(0);
           // URL cleanup happens in onended/onerror
           audioRef.current = null; // Clear ref after stopping
       } else {
           console.log(`[ChatPage:handleStopClick] No active audio player to stop.`); // LOG: Info
       }
       console.log(`[ChatPage:handleStopClick] Exiting.`); // LOG: Exit
   };
   // --- END REPLACEMENT: handleStopClick ---

   // --- START REPLACEMENT: handleSeekChange ---
   const handleSeekChange = (event) => {
       const newTime = parseFloat(event.target.value);
       console.log(`[ChatPage:handleSeekChange] Entered. New time value: ${newTime}, activeAudioMessageId: ${activeAudioMessageId}`); // LOG: Entry & State
       if (audioRef.current && activeAudioMessageId && !isNaN(newTime)) {
           console.log(`[ChatPage:handleSeekChange] Seeking audio for messageId ${activeAudioMessageId} to ${newTime}`); // LOG: Action
           audioRef.current.currentTime = newTime;
           setCurrentAudioTime(newTime); // Update state immediately
       } else {
            console.log(`[ChatPage:handleSeekChange] Seek ignored. Player: ${!!audioRef.current}, ActiveID: ${activeAudioMessageId}, Time: ${newTime}`); // LOG: Info
       }
        console.log(`[ChatPage:handleSeekChange] Exiting.`); // LOG: Exit
   };
   // --- END REPLACEMENT: handleSeekChange ---
   // --- End Playback Control Handlers ---

  // --- Image Upload Handlers ---
  const handleImageUploadClick = () => {
    // Trigger the hidden file input
    fileInputRef.current?.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      console.log("Image selected:", file.name);
      setSelectedImageFile(file);

      // Revoke previous URL if exists
      if (imagePreviewUrl) {
        console.log("Revoking previous Object URL:", imagePreviewUrl);
        URL.revokeObjectURL(imagePreviewUrl);
      }

      // Create and set new preview URL
      const newPreviewUrl = URL.createObjectURL(file);
      console.log("Created new Object URL:", newPreviewUrl);
      setImagePreviewUrl(newPreviewUrl);
    } else {
      console.log("No file selected or file is not an image.");
      // Clear previous selection if the new selection is invalid
      if (selectedImageFile) {
         handleCancelImage(); // Reuse cancel logic
      }
      if (file) {
          setError('Please select a valid image file.'); // Inform user
      }
    }
     // Reset the file input value to allow selecting the same file again
     if (event.target) {
        event.target.value = null;
     }
  };

  const handleCancelImage = () => {
    console.log("Cancelling image selection.");
    setSelectedImageFile(null);
    if (imagePreviewUrl) {
      console.log("Revoking Object URL on cancel:", imagePreviewUrl);
      URL.revokeObjectURL(imagePreviewUrl);
    }
    setImagePreviewUrl(null);
     // Also clear the file input ref's value if needed, though it might already be cleared by handleFileChange
     if (fileInputRef.current) {
        fileInputRef.current.value = null;
     }
 };
 // --- End Image Upload Handlers ---

 // --- Summarization Handlers ---
 const handleOpenSummarizeModal = () => {
   setSummarizeError(''); // Clear previous errors
   setSummarizeContentInput(''); // Clear previous input
   setSummarizeContentType('url'); // Reset to default type
   setSummarizeTargetLanguage('en'); // Reset to default language
   setIsSummarizeModalOpen(true);
 };

 const handleCloseSummarizeModal = () => {
   setIsSummarizeModalOpen(false);
   // Optionally reset state here if needed, though handleOpen does it too
 };

 const handleGenerateSummary = async () => {
   if (!summarizeContentInput.trim()) {
     setSummarizeError('Please enter content (URL or text) to summarize.');
     return;
   }
   // Basic URL validation if type is URL
   if (summarizeContentType === 'url') {
       try {
           new URL(summarizeContentInput); // Check if it's a valid URL structure
      } catch (_) { // eslint-disable-line no-unused-vars
          setSummarizeError('Please enter a valid URL.');
           return;
       }
   }

   // If already summarizing, abort the current request
   if (isSummarizing && summarizationControllerRef.current) {
     console.log("Summarization request in progress. Aborting...");
     summarizationControllerRef.current.abort();
     // State reset happens in finally block
     return;
   }

   setIsSummarizing(true);
   setSummarizeError('');

   // Create and store AbortController
   const controller = new AbortController();
   summarizationControllerRef.current = controller;
   console.log("Initiating summarization...");

   try {
     const result = await apiService.summarizeContent(
       chatbotId,
       summarizeContentType,
       summarizeContentInput,
       summarizeTargetLanguage,
       controller.signal // Pass signal
     );

     // Add summary to chat messages
     const summaryMessage = {
       id: `summary-${Date.now()}`,
       // Use Markdown for formatting
       content: `**Summary (${result.target_language}):**\n\n${result.summary}`,
       isUser: false, // Display as a bot message/response
       timestamp: Date.now(),
       isVoiceResponse: false, // Summaries are not direct voice responses
     };
     console.log("[ChatPage:handleGenerateSummary] Created summary message, isVoiceResponse:", summaryMessage.isVoiceResponse); // LOG: isVoiceResponse
     setMessages(prev => [...prev, summaryMessage]);

     handleCloseSummarizeModal(); // Close modal on success

   } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError') {
        console.log('Summarization request cancelled.');
        // Optionally set a specific message, or just clear the error
        setSummarizeError(''); // Clear any previous errors
      } else {
        console.error("Summarization failed:", err);
        // Check for specific backend error (e.g., token limit)
        if (err.response?.status === 500 && err.response?.data?.error?.includes('MAX_TOKENS')) {
            setSummarizeError('Summarization failed: The provided text or URL content is too long.');
        } else {
            setSummarizeError(err.message || 'Failed to generate summary.');
        }
      }
   } finally {
     setIsSummarizing(false);
     if (summarizationControllerRef.current === controller) { // Ensure it's the same controller
       summarizationControllerRef.current = null;
     }
   }
 };
 // --- End Summarization Handlers ---

  // --- Summarization Handlers ---
  return (
    // TODO: Replace with styled components later
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', height: '90vh' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h1>
           Chat with {loadingDetails ? '...' : (chatbotDetails?.name || `Bot ${chatbotId}`)}
        </h1>
        <button onClick={() => navigate('/dashboard')}>Back to Dashboard</button>
      </div>

      {/* Message Display Area */}
      <div style={{ flexGrow: 1, border: '1px solid #ccc', marginBottom: '1rem', padding: '1rem', overflowY: 'auto' }}>
        {messages.map(msg => {
          // LOG: Check msg.isVoiceResponse for each message during rendering
          // console.log(`[ChatPage:renderMessages] Msg ID: ${msg.id}, isUser: ${msg.isUser}, isVoiceResponse: ${msg.isVoiceResponse}, voice_enabled: ${chatbotDetails?.voice_enabled}`);
          return (
            <div key={msg.id} style={{ textAlign: msg.isUser ? 'right' : 'left', marginBottom: '0.5rem' }}>
              <span style={{
                display: 'inline-block',
                padding: '0.5rem 1rem',
                borderRadius: '10px',
                backgroundColor: msg.isUser ? '#dcf8c6' : '#eee',
                maxWidth: '70%',
                overflowWrap: 'break-word', // Ensure long words wrap
              }}>
                {/* Container for message content and controls */}
                <div>
                  {/* Render message content with Markdown */}
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>

                  {/* --- Modern TTS Playback Controls (Now conditional on msg.isVoiceResponse) --- */}
                  {!msg.isUser && msg.content && chatbotDetails?.voice_enabled && msg.isVoiceResponse && (
                    <div
                      className={`audio-controls modern-audio-controls ${activeAudioMessageId === msg.id || (currentAudioDuration > 0 && msg.id === activeAudioMessageId) ? 'active' : ''}`} // Add active class based on state
                      data-audio-controls-for={msg.id}
                    >
                      {/* Play/Pause Button */}
                    <button
                      className="audio-button play-pause-button"
                      onClick={() => handlePlayPauseClick(msg.id, msg.content)}
                      disabled={ttsLoading === msg.id || (!currentAudioDuration && activeAudioMessageId === msg.id && playbackState !== 'paused')} // Disable during synth or if duration not loaded (unless paused)
                      title={activeAudioMessageId === msg.id && playbackState === 'playing' ? "Pause" : "Play"}
                    >
                      {/* Use Bootstrap Icons */}
                      {ttsLoading === msg.id ? (
                        <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> // Simple spinner
                      ) : (
                        <i className={`bi ${activeAudioMessageId === msg.id && playbackState === 'playing' ? 'bi-pause-fill' : 'bi-play-fill'}`}></i>
                      )}
                    </button>

                    {/* Current Time */}
                    <span className="time-display current-time" data-audio-time="current">
                      {formatTime(activeAudioMessageId === msg.id ? currentAudioTime : 0)}
                    </span>

                    {/* Seek Bar Container */}
                    <div className="seek-bar-container">
                      <input
                        className="seek-bar"
                        type="range"
                        min="0"
                        max={activeAudioMessageId === msg.id ? (currentAudioDuration || 0) : 0}
                        value={activeAudioMessageId === msg.id ? (currentAudioTime || 0) : 0}
                        onChange={handleSeekChange}
                        disabled={activeAudioMessageId !== msg.id || playbackState === 'stopped' || !currentAudioDuration || currentAudioDuration === Infinity}
                        title="Seek"
                        data-audio-seek="seek"
                      />
                    </div>

                    {/* Duration Time */}
                    <span className="time-display duration-time" data-audio-time="duration">
                       {formatTime(activeAudioMessageId === msg.id ? (currentAudioDuration || 0) : 0)}
                    </span>

                    {/* Stop Button (Show only if playing/paused for this message) */}
                    {activeAudioMessageId === msg.id && playbackState !== 'stopped' && (
                      <button
                        className="audio-button stop-button"
                        onClick={handleStopClick}
                        title="Stop"
                        data-audio-action="stop"
                      >
                        <i className="bi bi-stop-fill"></i>
                      </button>
                    )}
                  </div>
                )}
                {/* --- End Modern TTS Playback Controls --- */}
              </div>
            </span>
            {/* Timestamp */}
            <div style={{ fontSize: '0.75rem', color: '#888', marginTop: '0.2rem', textAlign: msg.isUser ? 'right' : 'left' }}>
              {new Date(msg.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ); // Closing parenthesis for return statement in map
      })}
        {/* Dummy div to scroll to */}
        <div ref={messagesEndRef} />
        {/* Removed the specific isLoading indicator, now handled by the combined isProcessing indicator below */}
        {/* {isProcessing && <p style={{ textAlign: 'center', color: '#888' }}>Processing...</p>} */} {/* Replaced isLoading with isProcessing and generic text */}
        {voiceLoading && <p style={{ textAlign: 'center', color: '#888' }}>Processing voice...</p>}
      </div>

      {/* Input Area */}
      {/* Input Area - Conditionally render voice elements */}
      {/* Image Preview Area */}
      {imagePreviewUrl && (
        <div style={{ position: 'relative', marginBottom: '0.5rem', maxWidth: '100px', maxHeight: '100px' }}>
          <img
            src={imagePreviewUrl}
            alt="Selected preview"
            style={{ maxWidth: '100%', maxHeight: '100px', display: 'block', borderRadius: '4px' }}
          />
          <button
            type="button"
            onClick={handleCancelImage}
            disabled={isProcessing} // Disable cancel during processing
            style={{
              position: 'absolute',
              top: '2px',
              right: '2px',
              background: 'rgba(0,0,0,0.6)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '20px',
              height: '20px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '0.8rem',
              lineHeight: '1',
            }}
            title="Remove image"
          >
            &times; {/* Simple 'x' */}
          </button>
        </div>
      )}

      {/* Input Form */}
      {/* Replaced form with div to prevent potential refresh issues */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
         {/* Hidden File Input */}
         <input
           type="file"
           ref={fileInputRef}
           onChange={handleFileChange}
           accept="image/*" // Accept only image files
           style={{ display: 'none' }} // Keep it hidden
           disabled={isProcessing || voiceLoading || isRecording} // Disable if processing or voice active
         />

        {/* Image Upload Button (Conditionally Rendered) */}
        {chatbotDetails?.image_analysis_enabled && (
          <button
            type="button" // Prevent form submission
            onClick={handleImageUploadClick}
            disabled={isProcessing || voiceLoading || isRecording} // Disable if processing or voice active
            style={{ padding: '0.75rem', cursor: 'pointer' }}
            title="Attach Image"
          >
            📎 {/* Placeholder icon */}
          </button>
        )}

        {/* Text Input */}
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={chatbotDetails?.image_analysis_enabled ? "Type message or attach image..." : "Type your message..."}
          disabled={isProcessing || isRecording || voiceLoading} // Disable input during processing or voice activity
          style={{ flexGrow: 1, padding: '0.75rem' }}
        />
            {/* Summarize Button (Conditionally Rendered) */}
            {chatbotDetails?.summarization_enabled && (
              <button
                type="button" // Prevent form submission
                onClick={handleOpenSummarizeModal}
                disabled={isProcessing || isRecording || voiceLoading} // Disable if processing or voice active
                title="Summarize Content"
                style={{ padding: '8px 12px', marginLeft: '5px', cursor: 'pointer' }} // Basic styling
              >
                📄✨ {/* Document with sparkles icon */}
              </button>
            )}

        {/* Changed type to "button" and call handleSendMessage directly */}
        <button
          type="button"
          onClick={handleSendMessage} // Call original handler directly
          // Disable send ONLY if NOT processing AND (no text AND no image)
          // OR if recording/voice loading
          // Allow click when processing to trigger cancellation
          disabled={
            ( !isProcessing && (!input.trim() && !selectedImageFile) ) ||
            isRecording ||
            voiceLoading
          }
          style={{ padding: '0.75rem 1.5rem', cursor: 'pointer' }}
        >
          {isProcessing ? 'Stop' : 'Send'} {/* Change text when processing */}
        </button>

        {/* Conditionally render voice controls if enabled */}
        {chatbotDetails?.voice_enabled && (
          <>
            {/* Language Selector */}
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              disabled={isProcessing || isRecording || voiceLoading} // Disable if processing or voice active
              style={{ padding: '0.75rem', cursor: 'pointer' }}
              title="Select interaction language"
            >
              {supportedLanguages.map(lang => (
                <option key={lang.code} value={lang.code}>{lang.name} ({lang.code})</option>
              ))}
            </select>

            {/* Microphone Button */}
            <button
              type="button" // Important: prevent form submission
              onClick={handleMicButtonClick}
              // Disable only if main text/image processing is happening
              // Allow click during recording (to stop) or voiceLoading (to cancel API)
              disabled={isProcessing}
              style={{
                padding: '0.75rem',
                cursor: 'pointer',
                // Change background if recording OR if voice API call is loading
                backgroundColor: (isRecording || voiceLoading) ? 'red' : '#eee', // Red when recording OR processing voice API
                color: (isRecording || voiceLoading) ? 'white' : 'black',
                border: 'none',
                borderRadius: '50%',
                width: '40px',
                height: '40px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              // Update title based on state: Stop Recording, Stop Processing, Start Recording
              title={voiceLoading ? "Stop Processing Voice" : (isRecording ? "Stop Recording" : "Start Recording")}
            >
              {/* Show Stop icon if recording OR processing voice API, Mic otherwise */}
              {(isRecording || voiceLoading) ? '⏹️' : '🎤'}
            </button>
          </>
        )}
      </div>
       {/* Combined Loading Indicators */}
       {isProcessing && ( // Use unified processing state
         <p style={{ textAlign: 'center', color: '#888', marginTop: '0.5rem' }}>
           Processing...
         </p>
       )}
       {voiceLoading && !isProcessing && <p style={{ textAlign: 'center', color: '#888', marginTop: '0.5rem' }}>Processing voice...</p>} {/* Show voice loading only if not general processing */}
       {/* Error Messages */}
       {error && <p style={{ color: 'red', marginTop: '0.5rem' }}>Chat Error: {error}</p>}
       {voiceError && <p style={{ color: 'red', marginTop: '0.5rem' }}>Voice Error: {voiceError}</p>}
       {ttsError && <p style={{ color: 'orange', marginTop: '0.5rem' }}>TTS Error: {ttsError}</p>}

     {/* Summarization Modal */}
     {isSummarizeModalOpen && (
       <div style={modalOverlayStyle}>
         <div style={modalContentStyle}>
           <button onClick={handleCloseSummarizeModal} style={modalCloseButtonStyle} title="Close">
             &times;
           </button>
           <h2>Summarize Content</h2>

           {/* Content Type Selector */}
           <div style={{ marginBottom: '1rem' }}>
             <label>
               <input
                 type="radio"
                 name="summarizeType"
                 value="url"
                 checked={summarizeContentType === 'url'}
                 onChange={(e) => setSummarizeContentType(e.target.value)}
                 disabled={isSummarizing}
               /> URL
             </label>
             <label style={{ marginLeft: '1rem' }}>
               <input
                 type="radio"
                 name="summarizeType"
                 value="text"
                 checked={summarizeContentType === 'text'}
                 onChange={(e) => setSummarizeContentType(e.target.value)}
                 disabled={isSummarizing}
               /> Text
             </label>
           </div>

           {/* Content Input */}
           <div style={{ marginBottom: '1rem' }}>
             <label htmlFor="summarizeInput">
               {summarizeContentType === 'url' ? 'Enter URL:' : 'Paste Text:'}
             </label>
             {summarizeContentType === 'url' ? (
               <input
                 type="url"
                 id="summarizeInput"
                 value={summarizeContentInput}
                 onChange={(e) => setSummarizeContentInput(e.target.value)}
                 placeholder="https://example.com"
                 disabled={isSummarizing}
                 style={{ width: '100%', padding: '0.5rem', marginTop: '0.25rem' }}
               />
             ) : (
               <textarea
                 id="summarizeInput"
                 value={summarizeContentInput}
                 onChange={(e) => setSummarizeContentInput(e.target.value)}
                 placeholder="Enter the text you want to summarize..."
                 rows={5}
                 disabled={isSummarizing}
                 style={{ width: '100%', padding: '0.5rem', marginTop: '0.25rem' }}
               />
             )}
           </div>

           {/* Target Language Selector */}
           <div style={{ marginBottom: '1rem' }}>
             <label htmlFor="targetLanguage">Target Language:</label>
             <select
               id="targetLanguage"
               value={summarizeTargetLanguage}
               onChange={(e) => setSummarizeTargetLanguage(e.target.value)}
               disabled={isSummarizing}
               style={{ marginLeft: '0.5rem', padding: '0.5rem' }}
             >
               {/* Add more languages as needed */}
               <option value="en">English</option>
               <option value="es">Spanish</option>
               <option value="fr">French</option>
               <option value="de">German</option>
               <option value="ja">Japanese</option>
               <option value="zh">Chinese</option>
               {/* Add other supported languages from your backend/service */}
             </select>
           </div>

           {/* Error Display */}
           {summarizeError && (
             <p style={{ color: 'red', marginBottom: '1rem' }}>Error: {summarizeError}</p>
           )}

           {/* Action Button */}
           <button
             onClick={handleGenerateSummary}
             // Disable only if not summarizing AND input is empty
             // Allow click while summarizing to trigger cancellation
             disabled={!isSummarizing && !summarizeContentInput.trim()}
             style={{ padding: '0.75rem 1.5rem', cursor: 'pointer' }}
           >
             {isSummarizing ? 'Stop Summarizing' : 'Generate Summary'}
           </button>
         </div>
       </div>
     )}
     {/* End Summarization Modal */}

   </div>
  );
}

// Basic Modal Styles (can be moved to CSS later)
const modalOverlayStyle = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  backgroundColor: 'rgba(0, 0, 0, 0.5)',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  zIndex: 1000, // Ensure it's above other content
};

const modalContentStyle = {
  backgroundColor: '#fff',
  padding: '2rem',
  borderRadius: '8px',
  minWidth: '400px',
  maxWidth: '600px',
  position: 'relative',
  boxShadow: '0 4px 15px rgba(0, 0, 0, 0.2)',
};

const modalCloseButtonStyle = {
  position: 'absolute',
  top: '10px',
  right: '10px',
  background: 'none',
  border: 'none',
  fontSize: '1.5rem',
  cursor: 'pointer',
};

// Basic Modal Styles (can be moved to CSS later)
export default ChatPage;
