# chatbot-backend/app/api/voice_routes.py
import base64
from flask import Blueprint, request, jsonify, make_response, current_app, g
from app.services.voice_service import synthesize_speech_google, transcribe_audio_google
# Language detection removed from language_service
from app.models import Chatbot, ChatMessage # Import Chatbot and ChatMessage
from app.api.routes import require_api_key, get_rag_service # Import necessary items from main routes
from app import db # Import db for saving messages
import uuid # For generating unique IDs for audio files
import os   # For potential path operations
import time # Added for timing

# Define the blueprint
voice_bp = Blueprint('voice_bp', __name__, url_prefix='/api/voice')

# Language service functions are imported directly, no instantiation needed.

def _upload_audio_and_get_url(audio_content, chatbot_id, session_id):
    """
    Uploads audio content to a local directory and returns a URL.
    """
    # ... (keep this function exactly as is) ...
    upload_dir = os.path.join(current_app.root_path, 'uploads', 'audio')
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{chatbot_id}_{session_id}_{uuid.uuid4()}.mp3"
    filepath = os.path.join(upload_dir, filename)
    try:
        with open(filepath, 'wb') as f:
            f.write(audio_content)
        current_app.logger.info(f"Audio saved to {filepath}")
        backend_base_url = current_app.config.get('BACKEND_BASE_URL', 'http://localhost:5001')
        static_audio_path = '/uploads/audio'
        audio_url = f"{backend_base_url}{static_audio_path}/{filename}"
        current_app.logger.info(f"Generated audio URL: {audio_url}")
        return audio_url
    except Exception as e:
        current_app.logger.error(f"Failed to upload or save audio file {filename}: {e}")
        return None

@voice_bp.route('/tts', methods=['POST'])
def text_to_speech():
    """
    API endpoint to synthesize text to speech using Google Cloud TTS.
    """
    # TODO: Consider if this endpoint needs API key protection like the interact endpoint
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    text = data.get('text')
    language = data.get('language', 'en') # Get language from request, default to 'en'
    if not text:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    try:
        # Language detection removed. Use provided language or default.
        current_app.logger.info(f"Using language '{language}' for TTS input text.")

        audio_content = synthesize_speech_google(text, language_short_code=language) # Pass provided/default language
        if audio_content:
            response = make_response(audio_content)
            response.headers.set('Content-Type', 'audio/mpeg')
            return response
        else:
            return jsonify({"error": "Failed to synthesize speech"}), 500
    except Exception as e:
        current_app.logger.error(f"Unexpected error in /tts endpoint: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


@voice_bp.route('/chatbots/<int:chatbot_id>/interact', methods=['POST'])
@require_api_key # Use the decorator from routes.py to authenticate
def voice_interaction(chatbot_id):
    """
    Handles the full STT -> RAG -> TTS voice interaction flow.
    """
    overall_start_time = time.time()
    current_app.logger.info(f"VoiceInteraction: START for chatbot {chatbot_id}")
    chatbot = g.chatbot # API key decorator puts chatbot here

    if not chatbot.voice_enabled:
        current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Voice Disabled).")
        return jsonify({"error": "Voice interaction is not enabled for this chatbot."}), 403

    if 'audio' not in request.files:
        return jsonify({"error": "Missing 'audio' file part in the request."}), 400

    audio_file = request.files['audio']
    language = request.form.get('language') # Language selected by user for STT/TTS
    session_id = request.form.get('session_id')
    current_app.logger.debug(f"Raw language value from form: '{language}'") # ADDED LOGGING

    if not language:
        return jsonify({"error": "Missing 'language' form field."}), 400
    if not session_id:
        return jsonify({"error": "Missing 'session_id' form field."}), 400
    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file."}), 400

    supported_langs = {'am', 'en', 'fr', 'ar', 'ru', 'uk', 'es', 'he'}
    if language.lower() not in supported_langs:
         return jsonify({"error": f"Unsupported language: {language}. Supported: {', '.join(supported_langs)}"}), 400

    try:
        audio_content = audio_file.read()

        # 1. Perform STT (using the language specified in the request)
        stt_start_time = time.time()
        current_app.logger.info(f"Performing STT for chatbot {chatbot_id}, session {session_id}, lang {language}")
        transcribed_text = transcribe_audio_google(audio_content, language_short_code=language)
        current_app.logger.info(f"PERF: STT (transcribe_audio_google) for chatbot {chatbot_id} took {time.time() - stt_start_time:.4f} seconds.")

        if transcribed_text is None:
            current_app.logger.error(f"STT failed for chatbot {chatbot_id}, session {session_id}")
            current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (STT Fail).")
            return jsonify({"error": "Speech-to-text conversion failed."}), 500
        elif not transcribed_text.strip():
             current_app.logger.info(f"STT resulted in empty text for chatbot {chatbot_id}, session {session_id}")
             current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (STT Empty).")
             return jsonify({ "text_response": "", "audio_url": None, "detail": "No speech detected or transcription was empty." }), 200

        current_app.logger.info(f"STT Result for chatbot {chatbot_id}, session {session_id}: '{transcribed_text}'")
        current_app.logger.debug(f"STT Raw Output (type: {type(transcribed_text)}, length: {len(transcribed_text)}): {transcribed_text!r}")
 
         # 2. Retrieve and Format Chat History
        history_start_time = time.time()
        history_for_rag = [] 
        if chatbot.save_history_enabled:
            try:
                recent_messages_objects = ChatMessage.query.filter_by(
                    chatbot_id=chatbot_id, session_id=session_id
                ).order_by(ChatMessage.timestamp.desc()).limit(10).all()
                recent_messages_objects.reverse() 

                history_for_rag = [
                    {'role': msg.role, 'content': msg.content}
                    for msg in recent_messages_objects
                ]
                current_app.logger.debug(f"Retrieved and formatted {len(history_for_rag)} messages for history (chatbot {chatbot_id}, session {session_id})")
            except Exception as hist_e:
                 current_app.logger.error(f"Error retrieving/formatting chat history for chatbot {chatbot_id}, session {session_id}: {hist_e}")
        current_app.logger.info(f"PERF: Chat History Retrieval for chatbot {chatbot_id} took {time.time() - history_start_time:.4f} seconds.")

        # 3. Call RAG Pipeline (passing transcribed text and formatted history)
        rag_call_start_time = time.time()
        rag_service = get_rag_service()
        if not rag_service:
             current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (RAG Service Unavailable).")
             return jsonify({"error": "RAG service is unavailable."}), 503

        current_app.logger.info(f"Calling RAG pipeline for chatbot {chatbot_id}, session {session_id}")
        rag_result = rag_service.execute_pipeline(
            chatbot_id=chatbot_id,
            client_id=chatbot.client_id,
            query=transcribed_text, 
            chat_history=history_for_rag 
        )
        current_app.logger.info(f"PERF: RAG Pipeline execution for chatbot {chatbot_id} took {time.time() - rag_call_start_time:.4f} seconds.")

        # 4. Process RAG Response
        actual_response_text = None
        if isinstance(rag_result, dict) and rag_result.get('error') is None:
            actual_response_text = rag_result.get('answer')
            if not actual_response_text:
                 current_app.logger.warning(f"RAG pipeline returned success but no answer text for chatbot {chatbot_id}, session {session_id}.")
                 actual_response_text = "I could not find an answer based on the provided information." 
        else:
            error_detail = rag_result.get('error') if isinstance(rag_result, dict) else str(rag_result)
            current_app.logger.error(f"RAG pipeline failed for chatbot {chatbot_id}, session {session_id}. Response/Error: {error_detail}")
            current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (RAG Error).")
            return jsonify({"error": f"Failed to get response: {error_detail or 'RAG pipeline error'}"}), 500

        current_app.logger.info(f"RAG Response for chatbot {chatbot_id}, session {session_id}: '{actual_response_text[:100]}...'")

        # 5. Save messages to history (if enabled)
        if chatbot.save_history_enabled:
            db_save_start_time = time.time()
            try:
                user_message = ChatMessage(chatbot_id=chatbot_id, session_id=session_id, role='user', content=transcribed_text)
                assistant_message = ChatMessage(chatbot_id=chatbot_id, session_id=session_id, role='assistant', content=actual_response_text)
                db.session.add(user_message)
                db.session.add(assistant_message)
                db.session.commit()
                current_app.logger.debug(f"Saved user and assistant messages for chatbot {chatbot_id}, session {session_id}")
            except Exception as db_e:
                db.session.rollback()
                current_app.logger.error(f"Failed to save chat messages for chatbot {chatbot_id}, session {session_id}: {db_e}")
            current_app.logger.info(f"PERF: DB Message Save for chatbot {chatbot_id} took {time.time() - db_save_start_time:.4f} seconds.")

        # 6. Perform TTS (using the original language specified in the request)
        tts_call_start_time = time.time()
        current_app.logger.info(f"Attempting TTS for chatbot {chatbot_id}, session {session_id}...")
        current_app.logger.debug(f"TTS Input Text (first 100 chars): '{actual_response_text[:100]}...'")
        current_app.logger.debug(f"TTS Language Code: '{language}'")
        tts_audio_content = synthesize_speech_google(actual_response_text, language_short_code=language)
        current_app.logger.info(f"PERF: TTS (synthesize_speech_google) for chatbot {chatbot_id} took {time.time() - tts_call_start_time:.4f} seconds.")

        if tts_audio_content is None:
            current_app.logger.error(f"TTS failed for chatbot {chatbot_id}, session {session_id}. Returning text only.")
            current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (TTS Fail).")
            return jsonify({
                "text_response": actual_response_text,
                "audio_url": None,
                "detail": "Text response generated, but audio synthesis failed."
            }), 200

        # 7. Encode audio to base64
        base64_start_time = time.time()
        audio_response_base64 = None
        if tts_audio_content:
            try:
                audio_response_base64 = base64.b64encode(tts_audio_content).decode('utf-8')
                current_app.logger.info(f"Successfully base64 encoded TTS audio for chatbot {chatbot_id}, session {session_id}.")
            except Exception as enc_e:
                current_app.logger.error(f"Failed to base64 encode TTS audio for chatbot {chatbot_id}, session {session_id}: {enc_e}")
        else:
            current_app.logger.info(f"TTS content was None for chatbot {chatbot_id}, session {session_id}, so audio_response_base64 will be None.")
        current_app.logger.info(f"PERF: Base64 Encoding for chatbot {chatbot_id} took {time.time() - base64_start_time:.4f} seconds.")

        # 8. Return successful response with base64 audio
        current_app.logger.debug(f"Returning voice interaction response. audio_response_base64 is None: {audio_response_base64 is None}")
        current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Success).")
        return jsonify({
            "text_response": actual_response_text,
            "audio_response_base64": audio_response_base64, 
            "transcribed_input": transcribed_text
        }), 200

    except Exception as e:
        current_app.logger.error(f"Unexpected error in voice interaction for chatbot {chatbot_id}, session {session_id}: {e}", exc_info=True)
        current_app.logger.info(f"PERF: VoiceInteraction for chatbot {chatbot_id} took {time.time() - overall_start_time:.4f} seconds (Exception).")
        return jsonify({"error": "An internal server error occurred during voice interaction."}), 500
