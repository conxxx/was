// Functions related to voice input, recording, and VAD
import * as state from './state.js';
import * as ui from './ui.js';
import * as api from './api.js';
// SILENCE_THRESHOLD and SILENCE_DURATION_MS might be unused now if vad-processor handles its own thresholds/timing
// import { SILENCE_THRESHOLD, SILENCE_DURATION_MS } from './config.js';
// Import necessary state setters for playback control
import {
    setActiveAudioMessageId,
    setPlaybackState,
    setCurrentAudioTime,
    setCurrentAudioDuration,
    setCurrentAudioPlayer // Keep using this to store the Audio object itself
} from './state.js';


let chatbotId = null; // Will be set during initialization
let apiKey = null; // Will be set during initialization
let voiceApiController = null; // Controller for cancelling the voice API call

export function setCredentials(id, key) {
    chatbotId = id;
    apiKey = key;
}

// --- Voice Input Handling ---
export async function startRecording() {
    if (state.isRecording) {
        console.log("[voice:startRecording] Already recording.");
        return;
    }

    // --- Audio Context Management ---
    // Ensure AudioContext is ready before getUserMedia
    let audioCtx;
    try {
        if (state.audioContext && state.audioContext.state === 'running') {
            audioCtx = state.audioContext;
            console.log("[voice:startRecording] Reusing existing AudioContext.");
        } else if (state.audioContext && state.audioContext.state === 'suspended') {
             await state.audioContext.resume();
             audioCtx = state.audioContext;
             console.log("[voice:startRecording] Resumed existing suspended AudioContext.");
        } else {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            state.setAudioContext(audioCtx); // Store the context immediately
            console.log("[voice:startRecording] Created new AudioContext.");
        }
        // Resume context if it's suspended (often needed after page load)
        if (audioCtx.state === 'suspended') {
            await audioCtx.resume();
            console.log("[voice:startRecording] AudioContext resumed.");
        }
    } catch (ctxErr) {
        console.error("[voice:startRecording] Error ensuring AudioContext is running:", ctxErr);
        ui.updateVoiceStatus('Audio system error.', true);
        return; // Cannot proceed without a running audio context
    }
    // --- End Audio Context Management ---


    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const newMediaRecorder = new MediaRecorder(stream);
        state.setMediaRecorder(newMediaRecorder); // Store recorder immediately
        state.setLocalAudioChunks([]); // Reset local chunks

        // --- VAD Setup (AudioWorklet) ---
        let mediaStreamSource = null;
        let vadNode = null;
        if (state.chatbotConfig.vad_enabled === true && audioCtx) {
             console.log("[voice:startRecording] Setting up VAD (AudioWorklet)...");
             try {
                 // Create MediaStreamSource
                 mediaStreamSource = audioCtx.createMediaStreamSource(stream);
                 state.setMediaStreamSource(mediaStreamSource); // Store source
                 console.log("[voice:startRecording] MediaStreamSource created for VAD.");

                 // Add AudioWorklet module
                 try {
                     const workletPath = 'dist/vad-processor.js'; // Correct path relative to the HTML file
                     console.log('[voice:startRecording] PRE-TRY: Attempting to load AudioWorklet module from:', workletPath);
                     await audioCtx.audioWorklet.addModule(workletPath)
                         .then(() => {
                             console.log("[voice:startRecording] SUCCESS: AudioWorklet module added successfully.");
                         })
                         .catch((moduleError) => {
                             console.error('[voice:startRecording] CATCH within addModule promise: Error loading AudioWorklet module:', moduleError);
                             // Re-throw or handle as an error to be caught by the outer try-catch
                             throw moduleError; 
                         });
                     console.log("[voice:startRecording] POST-AWAIT: addModule call completed or promise handled.");
                 } catch (error) { // This is the outer catch
                     console.error('[voice:startRecording] OUTER CATCH: Error during VAD module loading/setup:', error);
                     ui.updateVoiceStatus('VAD Error: Load failed (outer catch).', true);
                     // Don't block recording, just log the error and continue without VAD
                     if (mediaStreamSource) {
                        mediaStreamSource.disconnect(); // Disconnect source if module failed
                        state.setMediaStreamSource(null);
                     }
                     mediaStreamSource = null; // Ensure VAD node isn't created below
                 }

                 // Create VAD node if module loaded and source exists
                 if (mediaStreamSource) {
                     vadNode = new AudioWorkletNode(audioCtx, 'vad-processor');
                     state.setVadNode(vadNode); // Store the node in state
                     console.log("[voice:startRecording] AudioWorkletNode (VAD) created.");
                     console.log('[voice:startRecording] VAD node created.'); // Added log

                     // Setup message listener
                     vadNode.port.onmessage = (event) => {
                         console.log('[voice:vadMessageHandler] Received message from VAD worklet:', event.data); // Added detailed log
                         // Check type and ensure recording is still active
                         if (event.data.type === 'silenceDetected' && state.isRecording) {
                             console.log('[voice:vadNode] Silence detected by VAD worklet, stopping recording.');
                             stopRecording(); // Call the existing stop function
                         } else if (event.data.type === 'speech') {
                             console.log('[voice:vadNode] Speech detected by VAD worklet.');
                             // Optional: Handle speech start if needed
                         } else if (event.data.type === 'error') {
                             console.error('[voice:vadNode] Error message from VAD processor:', event.data.message);
                             ui.updateVoiceStatus(`VAD Error: ${event.data.message}`, true);
                             // Optionally stop recording or just log
                         }
                     };
                     vadNode.port.onmessageerror = (error) => {
                         console.error('[voice:vadNode] Error receiving message from VAD processor:', error);
                     };
                     console.log("[voice:startRecording] VAD message listener attached.");

                     // Connect source to VAD node
                     console.log('[voice:startRecording] Connecting MediaStreamSource to vadNode...'); // Added log
                     mediaStreamSource.connect(vadNode);
                     // Do NOT connect vadNode to destination, it's just for analysis
                     console.log("[voice:startRecording] MediaStreamSource connected to vadNode.");
                     // Note: vadNode is not connected to destination, only source -> vadNode.
                     console.log("[voice:startRecording] VAD setup complete.");
                 }

             } catch (vadSetupError) {
                 console.error("[voice:startRecording] Error setting up VAD:", vadSetupError);
                 ui.updateVoiceStatus('VAD setup error.', true);
                 // Clean up any partial VAD setup
                 if (vadNode) {
                     vadNode.port.onmessage = null;
                     vadNode.port.onmessageerror = null;
                     vadNode.port.close();
                     vadNode.disconnect();
                     state.setVadNode(null);
                 }
                 if (mediaStreamSource) {
                     mediaStreamSource.disconnect();
                     state.setMediaStreamSource(null);
                 }
                 // Continue recording without VAD
             }
        } else {
            console.log("[voice:startRecording] VAD is disabled or AudioContext not available. Skipping VAD setup.");
        }
        // --- End VAD Setup ---


        newMediaRecorder.ondataavailable = event => {
            state.localAudioChunks.push(event.data);
        };

        // --- START mediaRecorder.onstop ---
        newMediaRecorder.onstop = async () => {
            console.log("[voice:onstop] Recording stopped (onstop triggered).");
            console.log(`[voice:onstop:entry] Event triggered. Current isRecording: ${state.isRecording}, userCancelledRecording: ${state.userCancelledRecording}, audioChunks.length: ${state.localAudioChunks.length}`);

            console.log(`[voice:onstop:pre-cancel-check] About to check userCancelledRecording. Value: ${state.userCancelledRecording}`);
            if (state.userCancelledRecording) {
                console.log(`[voice:onstop:cancel-path] userCancelledRecording is true. Current audioChunks.length: ${state.localAudioChunks.length}`);
                console.log("[voice:onstop] Recording was cancelled by user. Discarding audio.");
                state.setLocalAudioChunks([]); // Clear any potentially recorded chunks
                state.setUserCancelledRecording(false); // Reset the flag
                console.log(`[voice:onstop:cancel-path] userCancelledRecording flag reset to false.`);
                ui.updateVoiceStatus('Recording cancelled.');
                // Ensure STT loading is false if it was somehow set
                if (state.sttLoading) state.setSttLoading(false);
                ui.updateInputDisabledState(); // Re-enable inputs
                console.log(`[voice:onstop:cancel-path] Exiting onstop early due to user cancellation.`);
                return; // Exit early, do not process audio
            }

            const wasRecording = state.isRecording; // Capture state before async stopRecording call potentially changes it

            // VAD cleanup is now handled in stopRecording()

            if (state.localAudioChunks.length === 0) {
                console.warn("[voice:onstop] No audio data recorded.");
                // Ensure state is correct if stopRecording wasn't called for some reason
                // (e.g., immediate stop before any data)
                if (state.isRecording) {
                    console.warn("[voice:onstop] Still marked as recording despite no data, forcing state update.");
                    state.setIsRecording(false);
                    ui.updateInputDisabledState();
                }
                if (wasRecording) { // Only show status if recording was actually active
                   ui.updateVoiceStatus('No audio detected.', true);
                }
                // VAD cleanup handled in stopRecording() which should have been called
                return;
            }

            ui.updateVoiceStatus('Processing...'); // Indicate processing
            console.log(`[voice:onstop:process-path] Not cancelled. Proceeding to create audioBlob. audioChunks.length: ${state.localAudioChunks.length}`);
            const audioBlob = new Blob(state.localAudioChunks, { type: 'audio/webm' });
            console.log(`[voice:onstop] Audio blob created. Size: ${audioBlob.size}`);
            state.setLocalAudioChunks([]); // Clear chunks after creating blob

            // VAD cleanup handled in stopRecording()

            // Send audio for STT and interaction
            state.setSttLoading(true);
            ui.updateInputDisabledState();

            // Create and store AbortController for the API call
            voiceApiController = new AbortController();
            const signal = voiceApiController.signal;
            console.log("[voice:onstop] Widget: Initiating voice API call...");

            state.setLastInputMethod('voice'); // Set input method for voice
            console.log("[voice:onstop] lastInputMethod set to 'voice'.");

            try {
                // Call API function, passing the signal
                console.log(`[voice:onstop:process-path] About to call api.sendVoiceInteraction. sttLoading: ${state.sttLoading}`);
                const result = await api.sendVoiceInteraction(chatbotId, apiKey, audioBlob, signal);
                console.log('[voice:onstop] Backend voice interaction response:', result);

                // Add transcribed text and assistant response via UI module
                if (result.transcribed_text || result.transcribed_input) { // Check both possible keys
                    ui.addMessage(result.transcribed_text || result.transcribed_input, 'user');
                }

                // Convert base64 audio to Blob URL if present
                if (result.audio_response_base64) {
                    console.log('[voice:onstop] Found audio_response_base64, attempting to create Blob URL.');
                    try {
                        const byteCharacters = atob(result.audio_response_base64);
                        const byteNumbers = new Array(byteCharacters.length);
                        for (let i = 0; i < byteCharacters.length; i++) {
                            byteNumbers[i] = byteCharacters.charCodeAt(i);
                        }
                        const byteArray = new Uint8Array(byteNumbers);
                        // Using 'audio/webm' as a common type for Opus/Vorbis from backend TTS
                        const audioBlob = new Blob([byteArray], { type: 'audio/webm' });
                        result.audio_url = URL.createObjectURL(audioBlob);
                        console.log('[voice:onstop] Successfully created Blob URL:', result.audio_url);
                    } catch (e) {
                        console.error('[voice:onstop] Error creating Blob URL from base64:', e);
                        result.audio_url = null; // Ensure it's null if conversion fails
                    }
                }

                // Pass audio_url to addMessage and capture the returned DOM message ID
                const assistantMessageId = ui.addMessage(result.text_response, 'assistant', false, false, result.message_id, result.audio_url);
                console.log(`[voice:onstop] Assistant message added to UI with ID: ${assistantMessageId}`); // LOG: Added message ID

                // --- Handle TTS Playback with Controls ---
                console.log('[voice:onstop] Attempting to play audio from URL:', result.audio_url); // LOG: TTS Check
                if (result.audio_url && !result.audio_url.includes('example.com')) {
                    state.setTtsLoading(true); // Indicate loading/buffering
                    ui.updateInputDisabledState();
                    ui.updateVoiceStatus('Speaking...');

                    // Stop any currently playing audio and reset state
                    if (state.currentAudioPlayer) {
                        console.log(`[voice:onstop] Stopping previous player (messageId: ${state.activeAudioMessageId})`);
                        state.currentAudioPlayer.pause();
                        // Remove previous listeners
                        state.currentAudioPlayer.onloadedmetadata = null;
                        state.currentAudioPlayer.ontimeupdate = null;
                        state.currentAudioPlayer.onended = null;
                        state.currentAudioPlayer.onerror = null;
                        state.currentAudioPlayer.onplay = null;
                        state.currentAudioPlayer.onpause = null;
                        if (state.currentAudioPlayer.src && state.currentAudioPlayer.src.startsWith('blob:')) {
                            URL.revokeObjectURL(state.currentAudioPlayer.src);
                            console.log(`[voice:onstop] Revoked Object URL for previous player: ${state.currentAudioPlayer.src}`);
                        }
                    }
                    console.log('[voice:onstop] Resetting audio state before new playback.');
                    setActiveAudioMessageId(null);
                    setPlaybackState('stopped');
                    setCurrentAudioTime(0);
                    setCurrentAudioDuration(0);

                    const newPlayer = new Audio(result.audio_url);
                    // ** CRUCIAL FIX: Use the captured assistantMessageId consistently **
                    const currentMessageId = assistantMessageId;
                    console.log(`[voice:onstop] Created new Audio player for message ${currentMessageId}`);
                    setCurrentAudioPlayer(newPlayer); // Store the new player

                    // --- Attach Event Listeners using currentMessageId ---
                    console.log(`[voice:onstop] Attaching event listeners for message ${currentMessageId}`);
                    newPlayer.onloadedmetadata = () => {
                        console.log(`[voice:onstop:onloadedmetadata] Triggered for ${currentMessageId}. Duration: ${newPlayer.duration}`); // LOG: Event
                        // Only update if this is still the intended player
                        if (state.currentAudioPlayer === newPlayer) {
                            console.log(`[voice:onstop:onloadedmetadata] Updating state for ${currentMessageId}. Setting duration: ${newPlayer.duration}`); // LOG: State Update
                            setCurrentAudioDuration(newPlayer.duration);
                            setActiveAudioMessageId(currentMessageId); // Use captured ID
                            setCurrentAudioTime(0);
                            state.setTtsLoading(false); // Loading finished
                            ui.updateInputDisabledState();
                            // --- DEBUG LOG: Before UI Update ---
                            console.log(`[DEBUG:Controls] Before updateMessageAudioControlsUI for ${currentMessageId}. State: activeId=${state.activeAudioMessageId}, playback=${state.playbackState}, duration=${state.currentAudioDuration}`);
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                            // --- DEBUG LOG: After UI Update ---
                            console.log(`[DEBUG:Controls] After updateMessageAudioControlsUI for ${currentMessageId}.`);
                        } else {
                             console.log(`[voice:onstop:onloadedmetadata] Event for ${currentMessageId}, but player has changed. Ignoring.`);
                        }
                    };

                    newPlayer.ontimeupdate = () => {
                        if (state.currentAudioPlayer === newPlayer) {
                            // console.log(`[voice:onstop:ontimeupdate] Time update for ${currentMessageId}: ${newPlayer.currentTime}`); // Too noisy
                            setCurrentAudioTime(newPlayer.currentTime);
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                        }
                    };

                    newPlayer.onplay = () => {
                        if (state.currentAudioPlayer === newPlayer) {
                            console.log(`[voice:onstop:onplay] Audio playing for ${currentMessageId}`);
                            setPlaybackState('playing');
                            setActiveAudioMessageId(currentMessageId); // Use captured ID
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                        } else {
                            console.log(`[voice:onstop:onplay] Event for ${currentMessageId}, but player has changed. Ignoring.`);
                        }
                    };

                    newPlayer.onpause = () => {
                        // Only set to 'paused' if it wasn't intentionally stopped
                        if (state.currentAudioPlayer === newPlayer && state.playbackState !== 'stopped') {
                            console.log(`[voice:onstop:onpause] Audio paused for ${currentMessageId}`);
                            setPlaybackState('paused');
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                        } else {
                             console.log(`[voice:onstop:onpause] Event for ${currentMessageId}, but player changed or state is 'stopped'. Ignoring. Current state: ${state.playbackState}`);
                        }
                    };

                    newPlayer.onended = () => {
                        console.log(`[voice:onstop:onended] Audio ended for ${currentMessageId}`);
                        if (state.currentAudioPlayer === newPlayer) {
                            console.log(`[voice:onstop:onended] Updating state for ended audio ${currentMessageId}`);
                            setPlaybackState('stopped');
                            setActiveAudioMessageId(null);
                            setCurrentAudioTime(0);
                            // Don't reset duration here, keep it for display

                            // Clean up Blob URL if it exists before nullifying player
                            if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                                URL.revokeObjectURL(newPlayer.src);
                                console.log(`[voice:onstop:onended] Revoked Object URL on audio end: ${newPlayer.src}`);
                            }

                            setCurrentAudioPlayer(null); // Clear the player
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                            ui.updateVoiceStatus(''); // Clear general status
                            state.setTtsLoading(false); // Explicitly clear TTS loading state
                            ui.updateInputDisabledState(); // Update overall input state
                        } else {
                            console.log(`[voice:onstop:onended] Event for ${currentMessageId}, but player has changed. Ignoring.`);
                        }
                    };

                    newPlayer.onerror = (e) => {
                        console.error(`[voice:onstop:onerror] Audio error for ${currentMessageId}:`, e);
                        if (state.currentAudioPlayer === newPlayer) {
                            console.log(`[voice:onstop:onerror] Updating state for errored audio ${currentMessageId}`);
                            ui.updateVoiceStatus('Error playing audio.', true);
                            setPlaybackState('stopped');
                            setActiveAudioMessageId(null);
                            setCurrentAudioTime(0);
                            setCurrentAudioDuration(0); // Reset duration on error

                            // Clean up Blob URL if it exists before nullifying player
                            if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                                URL.revokeObjectURL(newPlayer.src);
                                console.log(`[voice:onstop:onerror] Revoked Object URL on audio error: ${newPlayer.src}`);
                            }

                            setCurrentAudioPlayer(null);
                            state.setTtsLoading(false);
                            ui.updateInputDisabledState();
                            ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                        } else {
                             console.log(`[voice:onstop:onerror] Event for ${currentMessageId}, but player has changed. Ignoring.`);
                        }
                    };
                    // --- End Event Listeners ---

                    // Conditional Attempt to play
                    if (state.lastInputMethod === 'voice') {
                        console.log(`[voice:onstop] Attempting to AUTO-PLAY TTS audio for ${currentMessageId} because lastInputMethod was 'voice'.`);
                        newPlayer.play().catch(e => {
                            console.error(`[voice:onstop] Error starting AUTO-PLAY TTS audio playback for ${currentMessageId}:`, e);
                            ui.updateVoiceStatus('Error playing audio.', true);
                            if (state.currentAudioPlayer === newPlayer) {
                                console.log(`[voice:onstop] Updating state after TTS auto-playback start error for ${currentMessageId}`);
                                setPlaybackState('stopped');
                                setActiveAudioMessageId(null);
                                setCurrentAudioPlayer(null);
                                state.setTtsLoading(false);
                                ui.updateInputDisabledState();
                                ui.updateMessageAudioControlsUI(currentMessageId); // Use captured ID
                            } else {
                                 console.log(`[voice:onstop] TTS Auto-playback start error for ${currentMessageId}, but player has changed. Ignoring.`);
                            }
                        });
                    } else {
                        // This case should ideally not be hit if we just set lastInputMethod to 'voice' above,
                        // but it's here for robustness.
                        console.log(`[voice:onstop] TTS audio ready for ${currentMessageId}, but NOT auto-playing (lastInputMethod: ${state.lastInputMethod}). Manual play enabled.`);
                    }

                } else {
                     console.warn("[voice:onstop] Skipping TTS setup due to missing or placeholder audio URL:", result.audio_url); // LOG: Skip TTS
                     ui.updateVoiceStatus(''); // Clear status if no audio URL
                     state.setTtsLoading(false); // Ensure loading state is reset
                     ui.updateInputDisabledState();
                }
                // --- End TTS Playback ---

            } catch (error) {
                 if (error.name === 'AbortError') {
                    console.log('[voice:onstop] Voice interaction API call cancelled.');
                    ui.updateVoiceStatus('Processing cancelled.', true); // Inform user
                 } else {
                    console.error('[voice:onstop] Widget Error during voice interaction API call:', error);
                    ui.addMessage(state.chatbotConfig.default_error_message || `Voice Error: ${error.message}`, 'assistant', true);
                    ui.updateVoiceStatus('Error processing audio.', true);
                 }
            } finally {
                 console.log('[voice:onstop] Voice API call finally block.');
                 state.setSttLoading(false);
                 // Don't update input disabled state here if TTS is still loading/playing
                 if (!state.ttsLoading && state.playbackState === 'stopped') {
                    ui.updateInputDisabledState();
                 }
                 voiceApiController = null; // Clear the controller for the API call
                 // Don't clear voice status here if TTS is active
            }
        };
        // --- END mediaRecorder.onstop ---

         newMediaRecorder.onerror = (event) => {
            console.error("[voice:newMediaRecorder.onerror] MediaRecorder encountered an error:", event); // MODIFIED LOG
            // Ensure recording stops and cleanup happens
            stopRecording(); // Call stopRecording to handle cleanup
            ui.updateVoiceStatus(`Recording error: ${event.error ? event.error.name : 'Unknown error'}`, true);
        };

        newMediaRecorder.start();
        state.setIsRecording(true);
        ui.updateInputDisabledState();
        ui.updateVoiceStatus('Recording...');

        // VAD setup happens above now

        console.log("[voice:startRecording] Recording started successfully.");

    } catch (err) {
        console.error("[voice:startRecording] Error accessing microphone:", err);
        ui.updateVoiceStatus(`Mic access error: ${err.message}`, true);
        state.setIsRecording(false);
        ui.updateInputDisabledState();
        // Clean up audio context if created but mic failed
        if (state.audioContext && state.audioContext.state === 'running') {
            state.audioContext.close().then(() => console.log("[voice:micError] Closed audio context after mic error."));
            state.setAudioContext(null);
        }
    }
}

export function stopRecording() {
    console.log(`[voice:stopRecording:entry] Called. Current isRecording: ${state.isRecording}, userCancelledRecording: ${state.userCancelledRecording}`);
    console.log("[voice] Widget: stopRecording() called.");

    // --- Set Recording State ---
    // Set state immediately to prevent race conditions (e.g., VAD message arriving after stop)
    const wasRecording = state.isRecording;
    if (wasRecording) {
        state.setIsRecording(false);
        ui.updateInputDisabledState(); // Update UI based on new state
        console.log("[voice:stopRecording] Set isRecording to false.");
    } else {
        console.log("[voice:stopRecording] stopRecording called but already not recording.");
    }


    // --- VAD Cleanup ---
    if (state.vadNode) {
        console.log('[voice:stopRecording] Cleaning up VAD node.');
        try {
            state.vadNode.port.onmessage = null; // Remove listener first
            state.vadNode.port.onmessageerror = null;
            state.vadNode.port.close();
            state.vadNode.disconnect(); // Disconnects from source and any downstream
        } catch (e) {
            console.error('[voice:stopRecording] Error during VAD node cleanup:', e);
        } finally {
            state.setVadNode(null);
        }
    } else {
        console.log('[voice:stopRecording] No VAD node found in state to clean up.');
    }

    // --- MediaStreamSource Cleanup ---
    // Disconnect source AFTER VAD node is disconnected from it
    if (state.mediaStreamSource) {
        console.log('[voice:stopRecording] Cleaning up MediaStreamSource.');
        try {
            state.mediaStreamSource.disconnect(); // Disconnect from all connections
        } catch (e) {
            console.error('[voice:stopRecording] Error disconnecting MediaStreamSource:', e);
        } finally {
            state.setMediaStreamSource(null);
        }
    } else {
         console.log('[voice:stopRecording] No MediaStreamSource found in state to clean up.');
    }

    // --- MediaRecorder Stop ---
    // Check state *before* stopping, as onstop handler relies on it
    const recorder = state.mediaRecorder; // Get reference before potentially nullifying
    console.log(`[voice:stopRecording] MediaRecorder instance: ${recorder ? 'Exists' : 'Null'}, State: ${recorder ? recorder.state : 'N/A'}`); // ADDED LOG
    if (recorder && recorder.state === 'recording') {
        console.log("[voice:stopRecording] Stopping MediaRecorder (state: recording)...");
        // The onstop handler will deal with processing chunks etc.
        console.log(`[voice:stopRecording:action] Calling state.mediaRecorder.stop().`);
        recorder.stop();
    } else if (recorder && recorder.state === 'paused') { // Specifically handle 'paused' state if possible
         console.log(`[voice:stopRecording] MediaRecorder state is 'paused'. Attempting stop.`);
         console.log(`[voice:stopRecording:action] Calling state.mediaRecorder.stop().`);
         recorder.stop();
    } else if (recorder && recorder.state !== 'inactive') {
         console.log(`[voice:stopRecording] MediaRecorder exists but state is '${recorder.state}' (not 'recording' or 'paused'). Attempting stop.`);
         try {
            console.log(`[voice:stopRecording:action] Calling state.mediaRecorder.stop().`);
            recorder.stop(); // Try stopping if state is weird
         } catch (e) {
             console.error("[voice:stopRecording] Error stopping MediaRecorder in non-standard state:", e);
         }
    } else {
        console.log("[voice:stopRecording] No active or stoppable MediaRecorder found.");
    }

    // --- MediaStream Track Stop ---
    // Stop the tracks associated with the stream AFTER recorder and source are done
    // Ensure we have the stream reference before trying to stop tracks
    const stream = recorder ? recorder.stream : null; // Use the captured recorder reference
    if (stream) {
        console.log("[voice:stopRecording] Stopping MediaStream tracks...");
        stream.getTracks().forEach(track => track.stop());
        console.log("[voice:stopRecording] MediaStream tracks stopped.");
    } else {
        console.log("[voice:stopRecording] No stream found on mediaRecorder to stop tracks.");
    }
    // Clear recorder reference after stopping tracks
    state.setMediaRecorder(null);


    // --- AudioContext Cleanup ---
    // Consider closing the context here or keeping it alive for future recordings
    // Closing ensures cleaner state but might add latency to next recording start.
    // Let's close it for now.
    if (state.audioContext && state.audioContext.state === 'running') {
        console.log("[voice:stopRecording] Closing AudioContext...");
        state.audioContext.close().then(() => {
            console.log("[voice:stopRecording] AudioContext closed successfully.");
        }).catch(e => console.error("[voice:stopRecording] Error closing AudioContext:", e))
        .finally(() => {
             state.setAudioContext(null); // Nullify even if close fails
        });
    } else {
        console.log("[voice:stopRecording] AudioContext not running or not found, skipping close.");
        if (state.audioContext) { // Nullify if it exists but wasn't running
             state.setAudioContext(null);
        }
    }

    console.log("[voice:stopRecording] Cleanup finished.");
    console.log(`[voice:stopRecording:exit] Exiting. Final isRecording: ${state.isRecording}, userCancelledRecording: ${state.userCancelledRecording}`);
}


// --- Function to cancel the ongoing voice API call ---
export function cancelVoiceApiCall() {
    if (voiceApiController) {
        console.log("[voice] Widget: Cancelling ongoing voice API call...");
        voiceApiController.abort();
        voiceApiController = null; // Clear immediately
        // UI updates (like status message) should happen in the catch/finally block where the call was made
    } else {
        console.log("[voice] Widget: No active voice API call to cancel.");
    }
}

// --- Time Formatting Helper ---
// (Could be moved to utils.js if needed elsewhere)
function formatTime(timeInSeconds) {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}


// --- Audio Playback Controls ---
// Function to initiate playback from a URL and update UI
export function playAudioFromUrl(audioUrl, messageId) {
    console.log(`[voice:playAudioFromUrl] Request to play audio for message ${messageId} from URL: ${audioUrl}`);

    // Stop any currently playing audio first
    if (state.currentAudioPlayer) {
        console.log(`[voice:playAudioFromUrl] Stopping existing player (messageId: ${state.activeAudioMessageId}) before playing new one.`);
        state.currentAudioPlayer.pause();
        // Remove listeners from the old player
        state.currentAudioPlayer.onloadedmetadata = null;
        state.currentAudioPlayer.ontimeupdate = null;
        state.currentAudioPlayer.onended = null;
        state.currentAudioPlayer.onerror = null;
        state.currentAudioPlayer.onplay = null;
        state.currentAudioPlayer.onpause = null;
        if (state.currentAudioPlayer.src && state.currentAudioPlayer.src.startsWith('blob:')) {
            URL.revokeObjectURL(state.currentAudioPlayer.src);
             console.log(`[voice:playAudioFromUrl] Revoked Object URL for previous player: ${state.currentAudioPlayer.src}`);
        }
        // Reset state associated with the old player
        const oldMessageId = state.activeAudioMessageId;
        setActiveAudioMessageId(null);
        setPlaybackState('stopped');
        setCurrentAudioTime(0);
        // Don't reset duration immediately, let the new player load its own
        setCurrentAudioPlayer(null);
        if (oldMessageId) {
            ui.updateMessageAudioControlsUI(oldMessageId); // Update UI for the stopped message
        }
    } else {
        console.log("[voice:playAudioFromUrl] No existing player to stop.");
    }

    // Reset state before creating the new player
    console.log('[voice:playAudioFromUrl] Resetting audio state before new playback.');
    setActiveAudioMessageId(null);
    setPlaybackState('stopped');
    setCurrentAudioTime(0);
    setCurrentAudioDuration(0); // Reset duration for the new track

    const newPlayer = new Audio(audioUrl);
    console.log(`[voice:playAudioFromUrl] Created new Audio player for message ${messageId}`);
    setCurrentAudioPlayer(newPlayer); // Store the new player
    setActiveAudioMessageId(messageId); // Tentatively set active ID
    setPlaybackState('loading'); // Indicate loading
    ui.updateMessageAudioControlsUI(messageId); // Show loading state

    // --- Attach Event Listeners ---
    console.log(`[voice:playAudioFromUrl] Attaching event listeners for message ${messageId}`);
    newPlayer.onloadedmetadata = () => {
        console.log(`[voice:playAudioFromUrl:onloadedmetadata] Triggered for ${messageId}. Duration: ${newPlayer.duration}`);
        if (state.currentAudioPlayer === newPlayer) {
            console.log(`[voice:playAudioFromUrl:onloadedmetadata] Updating state for ${messageId}. Duration: ${newPlayer.duration}`);
            setCurrentAudioDuration(newPlayer.duration);
            setCurrentAudioTime(0); // Ensure time is 0
            // Playback state will change on 'onplay'
            ui.updateMessageAudioControlsUI(messageId);
        } else {
            console.log(`[voice:playAudioFromUrl:onloadedmetadata] Event for ${messageId}, but player changed. Ignoring.`);
        }
    };

    newPlayer.ontimeupdate = () => {
        if (state.currentAudioPlayer === newPlayer) {
            // console.log(`[voice:playAudioFromUrl:ontimeupdate] Time update for ${messageId}: ${newPlayer.currentTime}`); // Too noisy
            setCurrentAudioTime(newPlayer.currentTime);
            ui.updateMessageAudioControlsUI(messageId);
        }
    };

    newPlayer.onplay = () => {
        if (state.currentAudioPlayer === newPlayer) {
            console.log(`[voice:playAudioFromUrl:onplay] Audio playing for ${messageId}`);
            setPlaybackState('playing');
            ui.updateMessageAudioControlsUI(messageId);
        } else {
             console.log(`[voice:playAudioFromUrl:onplay] Event for ${messageId}, but player changed. Ignoring.`);
        }
    };

    newPlayer.onpause = () => {
        // Only set to 'paused' if it wasn't intentionally stopped (e.g., by calling pauseAudio)
        if (state.currentAudioPlayer === newPlayer && state.playbackState !== 'stopped') {
            console.log(`[voice:playAudioFromUrl:onpause] Audio paused for ${messageId}`);
            setPlaybackState('paused');
            ui.updateMessageAudioControlsUI(messageId);
        } else {
             console.log(`[voice:playAudioFromUrl:onpause] Event for ${messageId}, but player changed or state is 'stopped'. Ignoring. Current state: ${state.playbackState}`);
        }
    };

    newPlayer.onended = () => {
        console.log(`[voice:playAudioFromUrl:onended] Audio ended for ${messageId}`);
        if (state.currentAudioPlayer === newPlayer) {
            console.log(`[voice:playAudioFromUrl:onended] Updating state for ended audio ${messageId}`);
            setPlaybackState('stopped');
            setActiveAudioMessageId(null);
            setCurrentAudioTime(0);
            // Keep duration

            // Clean up Blob URL if it exists
            if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                URL.revokeObjectURL(newPlayer.src);
                console.log(`[voice:playAudioFromUrl:onended] Revoked Object URL on audio end: ${newPlayer.src}`);
            }

            setCurrentAudioPlayer(null);
            ui.updateMessageAudioControlsUI(messageId);
            ui.updateVoiceStatus(''); // Clear general status if needed
            state.setTtsLoading(false); // Ensure TTS loading is false
            ui.updateInputDisabledState();
        } else {
             console.log(`[voice:playAudioFromUrl:onended] Event for ${messageId}, but player changed. Ignoring.`);
        }
    };

    newPlayer.onerror = (e) => {
        console.error(`[voice:playAudioFromUrl:onerror] Audio error for ${messageId}:`, e);
        if (state.currentAudioPlayer === newPlayer) {
            console.log(`[voice:playAudioFromUrl:onerror] Updating state for errored audio ${messageId}`);
            ui.updateVoiceStatus('Error playing audio.', true);
            setPlaybackState('stopped');
            setActiveAudioMessageId(null);
            setCurrentAudioTime(0);
            setCurrentAudioDuration(0);

            // Clean up Blob URL if it exists
            if (newPlayer.src && newPlayer.src.startsWith('blob:')) {
                URL.revokeObjectURL(newPlayer.src);
                console.log(`[voice:playAudioFromUrl:onerror] Revoked Object URL on audio error: ${newPlayer.src}`);
            }

            setCurrentAudioPlayer(null);
            state.setTtsLoading(false);
            ui.updateInputDisabledState();
            ui.updateMessageAudioControlsUI(messageId);
        } else {
             console.log(`[voice:playAudioFromUrl:onerror] Event for ${messageId}, but player changed. Ignoring.`);
        }
    };
    // --- End Event Listeners ---

    // Attempt to play
    console.log(`[voice:playAudioFromUrl] Attempting to play audio for ${messageId}`);
    newPlayer.play().catch(e => {
        console.error(`[voice:playAudioFromUrl] Error starting audio playback for ${messageId}:`, e);
        // Don't show generic error if it's just an interruption
        if (e.name !== 'AbortError') {
            ui.updateVoiceStatus('Error playing audio.', true);
        }
        if (state.currentAudioPlayer === newPlayer) {
            console.log(`[voice:playAudioFromUrl] Updating state after playback start error for ${messageId}`);
            setPlaybackState('stopped');
            setActiveAudioMessageId(null);
            setCurrentAudioPlayer(null);
            state.setTtsLoading(false);
            ui.updateInputDisabledState();
            ui.updateMessageAudioControlsUI(messageId);
        } else {
             console.log(`[voice:playAudioFromUrl] Playback start error for ${messageId}, but player changed. Ignoring.`);
        }
    });
}

// Pause the currently playing audio
export function pauseAudio() {
    if (state.currentAudioPlayer && state.playbackState === 'playing') {
        console.log(`[voice:pauseAudio] Pausing audio for message ${state.activeAudioMessageId}`);
        state.currentAudioPlayer.pause();
        // State update happens in the 'onpause' handler
    } else {
        console.log("[voice:pauseAudio] No audio playing to pause.");
    }
}

// Resume the currently paused audio
export function resumeAudio() {
    if (state.currentAudioPlayer && state.playbackState === 'paused') {
        console.log(`[voice:resumeAudio] Resuming audio for message ${state.activeAudioMessageId}`);
        state.currentAudioPlayer.play().catch(e => {
            console.error(`[voice:resumeAudio] Error resuming audio for ${state.activeAudioMessageId}:`, e);
            ui.updateVoiceStatus('Error resuming audio.', true);
            // Reset state if resume fails
            setPlaybackState('stopped');
            setActiveAudioMessageId(null);
            setCurrentAudioPlayer(null);
            state.setTtsLoading(false);
            ui.updateInputDisabledState();
            if (state.activeAudioMessageId) ui.updateMessageAudioControlsUI(state.activeAudioMessageId);
        });
        // State update happens in the 'onplay' handler
    } else {
        console.log("[voice:resumeAudio] No paused audio to resume.");
    }
}

// Seek the currently playing/paused audio
export function seekAudio(time) {
    if (state.currentAudioPlayer && state.currentAudioDuration > 0) {
        const seekTime = Math.max(0, Math.min(time, state.currentAudioDuration));
        console.log(`[voice:seekAudio] Seeking audio for message ${state.activeAudioMessageId} to ${seekTime}`);
        state.currentAudioPlayer.currentTime = seekTime;
        setCurrentAudioTime(seekTime); // Update state immediately for responsiveness
        ui.updateMessageAudioControlsUI(state.activeAudioMessageId);
    } else {
        console.log("[voice:seekAudio] Cannot seek: No active player or duration unknown.");
    }
}
