# chatbot-backend/app/services/voice_service.py
import os
import io
import time # Added for timing
from pydub import AudioSegment
from google.cloud import texttospeech, speech
from flask import current_app

# --- TTS ---
# Mapping from short codes to Google Cloud TTS language codes
# Ref: https://cloud.google.com/text-to-speech/docs/voices
TTS_LANGUAGE_MAP = {
    'am': 'am-ET', # Amharic (Ethiopia)
    'en': 'en-US', # English (United States)
    'fr': 'fr-FR', # French (France)
    'ar': 'ar-XA', # Arabic (Standard)
    'ru': 'ru-RU', # Russian (Russia)
    'uk': 'uk-UA', # Ukrainian (Ukraine)
    'es': 'es-ES', # Spanish (Spain)
    'he': 'he-IL', # Hebrew (Israel)
}

# Mapping from short codes to specific Google Cloud TTS voice names
# Ensure these voices support the corresponding language codes in TTS_LANGUAGE_MAP
# Ref: https://cloud.google.com/text-to-speech/docs/voices
TTS_VOICE_NAME_MAP = {
    'am': 'am-ET-Standard-A', # Amharic (Ethiopia) - Female
    'en': 'en-US-Neural2-A',  # English (United States) - Female (Neural2)
    'fr': 'fr-FR-Neural2-A',  # French (France) - Female (Neural2)
    'ar': 'ar-XA-Standard-A', # Arabic (Standard) - Female
    'ru': 'ru-RU-Standard-A', # Russian (Russia) - Female
    'uk': 'uk-UA-Standard-A', # Ukrainian (Ukraine) - Female
    'es': 'es-ES-Neural2-A',  # Spanish (Spain) - Female (Neural2)
    'he': 'he-IL-Wavenet-A',  # Hebrew (Israel) - Female (Wavenet) - Reverted back from Standard
}
 
def synthesize_speech_google(text: str, language_short_code: str = 'en') -> bytes | None:
    """
    Synthesizes speech from the input string of text using Google Cloud TTS
    for the specified language.

    Args:
        text: The text to synthesize.
        language_short_code: The short language code (e.g., 'en', 'fr', 'am').
                             Defaults to 'en'.

    Returns:
        The audio content in bytes (MP3 format), or None if an error occurs.
    """
    overall_start_time = time.time()
    current_app.logger.info(f"TTS: Starting synthesis for lang '{language_short_code}', text: '{text[:50]}...'")
    try:
        google_lang_code = TTS_LANGUAGE_MAP.get(language_short_code.lower())
        google_voice_name = None # Initialize voice name

        if not google_lang_code:
            current_app.logger.warning(f"Unsupported language code '{language_short_code}' for TTS. Falling back to 'en-US'.")
            google_lang_code = 'en-US' # Fallback language code
            google_voice_name = TTS_VOICE_NAME_MAP.get('en') # Fallback voice name
        else:
            # Language code is supported, try to find a specific voice
            google_voice_name = TTS_VOICE_NAME_MAP.get(language_short_code.lower())
            if not google_voice_name:
                current_app.logger.warning(f"No specific voice name found for '{language_short_code}'. Using language default if possible.")
                # If no specific voice is found, we'll rely on language_code and gender only
                # This might default to a standard voice or fail if none is available.

        current_app.logger.info(f"Attempting TTS synthesis with Language Code: '{google_lang_code}', Voice Name: '{google_voice_name or 'Default/Neutral'}'") # Log final chosen params

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Construct voice parameters
        voice_params = { "language_code": google_lang_code }

        if google_voice_name:
            # If we have a specific voice name, use it and omit gender
            voice_params["name"] = google_voice_name
        else:
            # If no specific voice name, fall back to NEUTRAL gender for the language code
            # This might work for some languages, or fail if no default neutral voice exists
            current_app.logger.warning(f"No specific voice name for lang '{language_short_code}', requesting NEUTRAL gender.")
            voice_params["ssml_gender"] = texttospeech.SsmlVoiceGender.NEUTRAL

        voice = texttospeech.VoiceSelectionParams(**voice_params)

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        api_call_start_time = time.time()
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        current_app.logger.info(f"PERF: TTS API call for lang '{language_short_code}' took {time.time() - api_call_start_time:.4f} seconds.")
        
        current_app.logger.info(f"PERF: TTS Overall synthesis for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds.")
        return response.audio_content
    except Exception as e:
        current_app.logger.error(f"Error synthesizing speech with Google Cloud TTS for lang '{language_short_code}': {e}")
        current_app.logger.info(f"PERF: TTS Overall synthesis for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds (Error).")
        return None

# --- STT ---
# Mapping from short codes to Google Cloud Speech language codes
# Ref: https://cloud.google.com/speech-to-text/docs/languages
STT_LANGUAGE_MAP = {
    'am': 'am-ET', # Amharic (Ethiopia)
    'en': 'en-US', # English (United States)
    'fr': 'fr-FR', # French (France)
    'ar': 'ar-XA', # Arabic (Modern Standard)
    'ru': 'ru-RU', # Russian (Russia)
    'uk': 'uk-UA', # Ukrainian (Ukraine)
    'es': 'es-ES', # Spanish (Spain)
    'he': 'he-IL', # Hebrew (Israel)
}

def transcribe_audio_google(
    audio_content: bytes,
    language_short_code: str = 'en'
) -> str | None:
    """
    Transcribes the given audio content using Google Cloud Speech-to-Text after
    converting it to LINEAR16, 16000 Hz, mono.

    Args:
        audio_content: The audio data bytes (format will be inferred by pydub).
        language_short_code: The short language code (e.g., 'en', 'fr', 'am').
                             Defaults to 'en'.

    Returns:
        The transcribed text as a string, or None if an error occurs (including
        conversion errors) or no speech is detected.
    """
    overall_start_time = time.time()
    current_app.logger.info(f"STT: Starting transcription for lang '{language_short_code}'.")
    try:
        google_lang_code = STT_LANGUAGE_MAP.get(language_short_code.lower())
        if not google_lang_code:
            current_app.logger.warning(f"Unsupported language code '{language_short_code}' for STT. Falling back to 'en-US'.")
            google_lang_code = 'en-US'

        # --- Audio Transcoding with pydub ---
        conversion_start_time = time.time()
        try:
            audio_io = io.BytesIO(audio_content)
            audio_segment = AudioSegment.from_file(audio_io) # pydub infers format

            # Convert to LINEAR16 (PCM 16-bit), 16kHz, Mono
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            converted_audio_bytes = audio_segment.raw_data
            current_app.logger.info(f"Audio successfully converted for STT (lang: {language_short_code}).")
            current_app.logger.info(f"PERF: STT Audio Conversion for lang '{language_short_code}' took {time.time() - conversion_start_time:.4f} seconds.")
        except Exception as e:
            current_app.logger.error(f"Audio conversion failed for lang '{language_short_code}': {e}")
            current_app.logger.info(f"PERF: STT Audio Conversion for lang '{language_short_code}' took {time.time() - conversion_start_time:.4f} seconds (Error).")
            current_app.logger.info(f"PERF: STT Overall transcription for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds (Conversion Error).")
            return None
        # --- End Audio Transcoding ---

        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(content=converted_audio_bytes) # Use converted audio
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Hardcoded encoding
            sample_rate_hertz=16000, # Hardcoded sample rate
            language_code=google_lang_code,
            # Enable automatic punctuation (optional, but often helpful)
            enable_automatic_punctuation=True
        )

        api_call_start_time = time.time()
        # Detects speech in the audio file
        response = client.recognize(config=config, audio=audio)
        current_app.logger.info(f"PERF: STT API call for lang '{language_short_code}' took {time.time() - api_call_start_time:.4f} seconds.")

        if response.results:
            # Concatenate results if needed, but usually the first result is sufficient for short audio
            transcript = response.results[0].alternatives[0].transcript
            current_app.logger.info(f"Transcription successful for lang '{language_short_code}'.")
            current_app.logger.info(f"PERF: STT Overall transcription for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds.")
            return transcript
        else:
            current_app.logger.info(f"No speech detected in audio for lang '{language_short_code}'.")
            current_app.logger.info(f"PERF: STT Overall transcription for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds (No Speech Detected).")
            return "" # Return empty string if no speech detected

    except Exception as e:
        current_app.logger.error(f"Error transcribing audio with Google Cloud STT for lang '{language_short_code}': {e}")
        current_app.logger.info(f"PERF: STT Overall transcription for lang '{language_short_code}' took {time.time() - overall_start_time:.4f} seconds (Error).")
        return None


# --- Placeholder for coordinating function ---
# def process_voice_interaction(...)
#     pass
