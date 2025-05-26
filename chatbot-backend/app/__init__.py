# app/__init__.py
from flask import Flask, send_from_directory
from flask import Flask, send_from_directory
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_login import LoginManager
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from flask_mail import Mail
from flask_limiter import Limiter # Import Limiter
from flask_limiter.util import get_remote_address # Import strategy for IP identification
# from celery import Celery, Task # Removed Celery import
import logging
import os

# Initialize extensions without app context
db = SQLAlchemy()
migrate = Migrate()
cors = CORS()
login_manager = LoginManager()
jwt = JWTManager()
bcrypt = Bcrypt()
mail = Mail()
limiter = Limiter( # Initialize Limiter globally - Storage URI will be set from config in create_app
    key_func=get_remote_address, # Identify users by IP address
) # storage_uri is now set dynamically in create_app using app.config

# --- Removed Celery Initialization/make_celery function ---

# --- Flask App Factory ---
def create_app(config_class=Config):

    app = Flask(__name__)
    app.config.from_object(config_class)
    # --- Create Upload Directories ---
    # Ensure necessary upload directories exist after config is loaded
    try:
        # Get paths from config, providing defaults based on previous structure
        # os.makedirs creates paths relative to the Current Working Directory (usually project root)
        upload_dirs_to_ensure = [
            app.config.get('UPLOAD_FOLDER', 'uploads'),
            app.config.get('LOGO_UPLOAD_FOLDER', os.path.join('app', 'static', 'logos')),
            app.config.get('AVATAR_UPLOAD_FOLDER_BASE', os.path.join('app', 'static', 'avatars')),
            app.config.get('LAUNCHER_ICON_UPLOAD_FOLDER_BASE', os.path.join('app', 'static', 'launcher_icons'))
        ]
        for dir_path in upload_dirs_to_ensure:
            if dir_path: # Proceed only if path is defined
                os.makedirs(dir_path, exist_ok=True)
                app.logger.info(f"Ensured directory exists: {dir_path}")
    except Exception as e:
        app.logger.error(f"CRITICAL: Failed to create initial upload directories defined in config: {e}", exc_info=True)
        # Decide if this should be a fatal error
        # raise RuntimeError(f"Failed to create essential directory: {e}")
    # -----------------------------


    # --- Configure Logging ---
    # Set the overall app logger level to DEBUG
    # This will allow INFO and DEBUG messages to be processed
    app.logger.setLevel(logging.DEBUG)

    # Optional: Configure a handler if needed (Flask usually adds one by default)
    # You might want to customize the format later
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
         # Ensure logging is set up properly, especially if not in Flask debug mode
         # Or just rely on the default handler added by Flask/Werkzeug
         pass
         # Example handler setup (uncomment and adjust if default isn't working):
         # handler = logging.StreamHandler()
         # handler.setLevel(logging.DEBUG)
         # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         # handler.setFormatter(formatter)
         # if not app.logger.handlers: # Add handler only if none exist
         #     app.logger.addHandler(handler)
    # ------------------------

    # Initialize extensions with app context
    db.init_app(app)
    migrate.init_app(app, db)
    # cors.init_app(app, resources={r"/api/*": {"origins": ["http://localhost:5173"], "supports_credentials": True}}) # Moved down
    
    # Initialize authentication extensions (if still needed)
    # login_manager.init_app(app) # Assuming JWT/client_id is primary auth now
    jwt.init_app(app)
    bcrypt.init_app(app)
    mail.init_app(app)
    # Initialize limiter with the app and storage URI from config
    limiter.init_app(app) 
    app.config.setdefault('RATELIMIT_STORAGE_URI', 'memory://') # Ensure default if not in config obj
    limiter.storage_uri = app.config['RATELIMIT_STORAGE_URI']
    # Set default rate limits from config
    default_limits = [
        f"{app.config['DEFAULT_RATE_LIMIT_PER_MINUTE']} per minute",
        f"{app.config['DEFAULT_RATE_LIMIT_PER_HOUR']} per hour",
        f"{app.config['DEFAULT_RATE_LIMIT_PER_DAY']} per day"
    ]
    limiter.default_limits = default_limits
    app.logger.info(f"Default rate limits set to: {', '.join(default_limits)}")

    app.logger.info(f"Rate limiter initialized with storage: {limiter.storage_uri}")


    # --- Removed Celery Configuration Logic ---
    # Celery instance defined in celery_worker.py
    # ContextTask setup moved to run.py (or similar entry point)
    # -----------------------------------------
    
    # Configure login manager (if using Flask-Login features)
    # login_manager.login_view = 'auth.login' # Adjust if needed
    # login_manager.login_message_category = 'info'
    
    # @login_manager.user_loader # Keep if Flask-Login is used elsewhere
    # def load_user(user_id):
    #     from app.models import User
    #     return User.query.get(int(user_id))

    # Initialize CORS *before* registering blueprints
    cors.init_app(app,
                  resources={r"/api/.*": {"origins": ["http://localhost:5173", "http://localhost:8000", "null"]}}, # Added "null" for file:// origin
                  methods=["GET", "HEAD", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"], # Allow standard methods including OPTIONS
                  allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-Client-ID"], # Allow required headers including X-Client-ID
                  supports_credentials=True) # Allow cookies/auth headers

    # Register blueprints
    from app.api import bp as api_bp
    from app.api.voice_routes import voice_bp # Import the new voice blueprint
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(voice_bp) # Register the voice blueprint (uses prefix defined in voice_routes.py)

    
    # Register additional routes
    from app.api import subscription_routes # Ensure auth_routes is not imported if not used
    app.register_blueprint(subscription_routes.bp, url_prefix='/api/subscriptions')
    # Ensure auth_routes blueprint is not registered
    from app.api import settings_routes # Import the new settings blueprint
    # from app.api import auth_routes 
    # app.register_blueprint(auth_routes.bp, url_prefix='/api/auth') 

    app.register_blueprint(settings_routes.bp, url_prefix='/api/settings') # Register settings blueprint
    # You can add other blueprints for web pages, auth, etc. later


    # Route to serve uploaded logos
    @app.route('/uploads/logos/<path:filename>')
    def serve_logo(filename):
        logo_dir = os.path.join(app.root_path, '..', 'uploads', 'logos')
        app.logger.debug(f"Attempting to serve logo: {filename} from {logo_dir}")
        try:
            return send_from_directory(logo_dir, filename)
        except FileNotFoundError:
            app.logger.error(f"Logo file not found: {filename} in {logo_dir}")
            return "File not found", 404

    @app.route('/uploads/audio/<path:filename>')
    def serve_audio(filename):
        audio_dir = os.path.join(app.root_path, 'uploads', 'audio')
        app.logger.debug(f"Attempting to serve audio: {filename} from {audio_dir}")
        try:
            return send_from_directory(audio_dir, filename)
        except FileNotFoundError:
            app.logger.error(f"Audio file not found: {filename} in {audio_dir}")
            return "File not found", 404

    @app.route('/test/')
    def test_page():
        return '<h1>Flask Backend is Running!</h1>'

    return app

# Import models here AFTER db is defined to avoid circular imports
# and so that Flask-Migrate can detect them
from app import models
