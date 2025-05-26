# config.py
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess' # CHANGE IN PRODUCTION!
    # Use SQLite for simplicity in MVP
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or os.environ.get('SECRET_KEY')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(seconds=int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 86400)))
    
    # OAuth Configuration
    GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID')
    APPLE_CLIENT_ID = os.environ.get('APPLE_CLIENT_ID')
    APPLE_TEAM_ID = os.environ.get('APPLE_TEAM_ID')
    APPLE_KEY_ID = os.environ.get('APPLE_KEY_ID')
    
    # Email Configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', 'yes', '1')
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    
    # GCP configurations
    PROJECT_ID = os.environ.get('PROJECT_ID', "roo-code-459017")
    REGION = os.environ.get('REGION', "us-central1")
    BUCKET_NAME = os.environ.get('BUCKET_NAME', "was-bucket4")
    INDEX_ENDPOINT_ID = os.environ.get('INDEX_ENDPOINT_ID', "7916908131475521536")
    INDEX_ID = os.environ.get('DEPLOYED_INDEX_ID', "deployed_1746727342706")


    # --- Model Names ---
    # NEW: Update the default or ensure the environment variable is set to the new model
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "text-embedding-large-exp-03-07") # TDD_ANCHOR: test_config_loads_correct_embedding_model
    GENERATION_MODEL_NAME = os.environ.get('GENERATION_MODEL_NAME', "gemini-2.5-flash-preview-04-17")
    #REPHRASE_MODEL_NAME = os.environ.get('REPHRASE_MODEL_NAME', "gemini-2.5-flash-preview-04-17")
    
# --- RAG & Generation Configuration ---
    GENERATION_MAX_TOKENS = int(os.environ.get('GENERATION_MAX_TOKENS', 2048)) # Increased default from 512
    GENERATION_TEMPERATURE = float(os.environ.get('GENERATION_TEMPERATURE', 0.3))
    RAG_TOP_K = int(os.environ.get('RAG_TOP_K', 5)) # Number of chunks to retrieve
    MAX_CONTEXT_CHARS = int(os.environ.get('MAX_CONTEXT_CHARS', 8000)) # Max chars for context in prompt
    # --- Google Generative AI Configuration ---
    GOOGLE_GEMINI_API_KEY = os.environ.get('GOOGLE_GEMINI_API_KEY')
    # When using the google-generativeai SDK with Vertex AI, this environment variable
    # must be set to 'True'. The SDK reads this directly from the environment.
    # Adding it to the Flask config for completeness and awareness.
    # Ensure GOOGLE_GENAI_USE_VERTEXAI=True is in your .env file.
    GOOGLE_GENAI_USE_VERTEXAI = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() in ('true', '1', 'yes')

    # --- Celery Configuration ---
    # Use Redis as the message broker and result backend (adjust URL if Redis is elsewhere)
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    # Optional: Configure task serialization, timezone, etc.
    # CELERY_TASK_SERIALIZER = 'json'
    # CELERY_RESULT_SERIALIZER = 'json'
    # CELERY_ACCEPT_CONTENT = ['json']
    # CELERY_TIMEZONE = 'UTC'
    # CELERY_ENABLE_UTC = True

    # --- Rate Limiting Configuration ---
    # Defaults to in-memory for development if not set in environment
    RATELIMIT_STORAGE_URI = os.environ.get('RATELIMIT_STORAGE_URI', 'memory://')
    DEFAULT_RATE_LIMIT_PER_MINUTE = os.environ.get('DEFAULT_RATE_LIMIT_PER_MINUTE', '60')
    DEFAULT_RATE_LIMIT_PER_HOUR = os.environ.get('DEFAULT_RATE_LIMIT_PER_HOUR', '200')
    DEFAULT_RATE_LIMIT_PER_DAY = os.environ.get('DEFAULT_RATE_LIMIT_PER_DAY', '1000')
    DEFAULT_RATE_LIMIT = "200 per day" # Default rate limit for limiter decorators

    # --- File Upload Configuration ---
    MAX_IMAGE_SIZE_BYTES = int(os.environ.get('MAX_IMAGE_SIZE_BYTES', 10 * 1024 * 1024)) # Default 10MB
    MAX_IMAGE_SIZE_MB = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
    ALLOWED_IMAGE_MIME_TYPES = set(os.environ.get('ALLOWED_IMAGE_MIME_TYPES', 'image/jpeg,image/png,image/webp,image/gif').split(','))
