# --- GEVENT PATCHING ---
# Do this BEFORE any other imports that might use network/ssl
try:
    import gevent.monkey
    gevent.monkey.patch_all()
    # print("Gevent monkey patching applied.") # Made silent
except ImportError:
    pass # print("Gevent not installed, skipping monkey patching.") # Made silent
# -----------------------

import os
# --- Explicitly set GOOGLE_GENAI_USE_VERTEXAI for Celery worker ---
# This ensures the google-generativeai SDK targets Vertex AI.
# It's set here to take precedence and ensure it's active before task modules are loaded.
env_var_name = "GOOGLE_GENAI_USE_VERTEXAI"
current_value = os.getenv(env_var_name)
if current_value != 'True':
    os.environ[env_var_name] = 'True'
    # print(f"Celery Worker: Programmatically SET {env_var_name} to 'True'. Previous value: '{current_value}'") # Made silent
# else:
    # print(f"Celery Worker: {env_var_name} already set to 'True'. Value: '{current_value}'") # Made silent
from celery import Celery
from config import Config # Import Config directly

# Define the Celery application instance here
# It reads configuration directly from the Config object
celery_app = Celery(
    'chatbot_backend_tasks', # Give it a name (can be anything)
    backend=Config.CELERY_RESULT_BACKEND,
    broker=Config.CELERY_BROKER_URL,
    include=[
        'app.services.ingestion',
        'app.services.discovery',
        'cleanup_tasks',  # Add the cleanup tasks module
        'app.tasks.deletion_tasks' # Add the new deletion tasks module
    ] # Tell Celery where to find tasks
)

# Address CPendingDeprecationWarning for Celery 6.0+ compatibility
celery_app.conf.broker_connection_retry_on_startup = True
# Load optional Celery config from Flask config class (optional settings)
# celery_app.config_from_object(Config) # This might re-read broker/backend, maybe redundant

# Set optional configurations if needed (example)
# celery_app.conf.update(
#     task_serializer='json',
#     result_serializer='json',
#     accept_content=['json'],
#     timezone='UTC',
#     enable_utc=True,
# )

# --- Celery Beat Schedule ---
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'delete-old-chat-messages-daily': {
        'task': 'cleanup_tasks.cleanup_old_chat_messages', # Task name as defined in cleanup_tasks.py
        'schedule': crontab(minute=0, hour=3),  # Run daily at 3:00 AM UTC
        #'args': (), # No arguments needed for this task
    },
}
celery_app.conf.timezone = 'UTC' # Ensure timezone is set for schedule clarity
# --------------------------

# IMPORTANT: The ContextTask setup is now moved to run.py or wherever the Flask app is run
# This file only defines the core Celery instance for the worker to use.
