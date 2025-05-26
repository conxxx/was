// chatbot-frontend/src/vad-processor.js

const SILENCE_THRESHOLD = 10; // Same threshold as before
const SILENCE_DURATION_MS = 1500; // Same duration as before
// Note: sampleRate is available globally in AudioWorkletGlobalScope
const SAMPLES_PER_BUFFER = 128; // Standard buffer size for AudioWorkletProcessor process method

class VadProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super(options);
    this._speaking = false;
    this._silentFramesCount = 0;
    // Calculate how many consecutive silent buffers are needed
    const silentDurationSec = SILENCE_DURATION_MS / 1000;
    const buffersPerSecond = sampleRate / SAMPLES_PER_BUFFER;
    this._requiredSilentFrames = Math.ceil(silentDurationSec * buffersPerSecond);

    this.port.onmessage = (event) => {
      // Handle messages from the main thread if needed in the future
      // console.log('[VadProcessor] Message received:', event.data);
    };
    console.log(`[VadProcessor] Initialized. Sample Rate: ${sampleRate}, Required Silent Frames: ${this._requiredSilentFrames}`);
  }

  process(inputs, outputs, parameters) {
    // We only expect one input connection
    const inputChannel = inputs[0]?.[0]; // Get the first channel of the first input

    // If no input data, keep processing but do nothing else
    if (!inputChannel) {
      return true; // Keep processor alive
    }

    // Calculate RMS (Root Mean Square) or average amplitude for the buffer slice
    let sumSquares = 0;
    for (let i = 0; i < inputChannel.length; i++) {
      sumSquares += inputChannel[i] * inputChannel[i];
    }
    const rms = Math.sqrt(sumSquares / inputChannel.length);
    // Convert RMS to a pseudo-amplitude value (0-255 range is not directly applicable here)
    // Let's scale RMS (typically 0-1) to a more comparable range, e.g., 0-100
    const scaledAmplitude = rms * 300; // Adjust scaling factor based on testing

    // console.log(`[VadProcessor] Scaled Amplitude: ${scaledAmplitude.toFixed(2)}`); // Debugging

    if (scaledAmplitude > SILENCE_THRESHOLD) {
      // Speech detected
      if (!this._speaking) {
        // console.log('[VadProcessor] Speech started.');
        this._speaking = true;
      }
      this._silentFramesCount = 0; // Reset silence counter
    } else if (this._speaking) {
      // Silence detected *after* speech
      this._silentFramesCount++;
      // console.log(`[VadProcessor] Silent Frames Count: ${this._silentFramesCount}`);

      if (this._silentFramesCount >= this._requiredSilentFrames) {
        console.log('[VadProcessor] Silence duration met (frame count). Posting message.');
        this.port.postMessage({ type: 'silenceDetected' });
        this._speaking = false; // Reset speaking state
        this._silentFramesCount = 0; // Reset counter
        // Note: We don't stop the processor itself here, the main thread handles MediaRecorder stop
      }
    }
    // If !this._speaking and amplitude is low, do nothing (initial silence)

    // Return true to keep the processor alive
    return true;
  }
}

try {
  // Name the processor uniquely
  registerProcessor('vad-processor', VadProcessor);
} catch (e) {
  console.error('Error registering VadProcessor:', e);
  // Post error back to main thread if possible, or just log
  // self.postMessage({ type: 'error', detail: 'Failed to register processor' });
}
