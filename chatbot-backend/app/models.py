# app/models.py
import uuid
from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    client_id = db.Column(db.String(36), index=True, unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Authentication fields
    password_hash = db.Column(db.String(128))
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(100))
    reset_password_token = db.Column(db.String(100))
    last_login = db.Column(db.DateTime)
    
    # OAuth fields
    auth_provider = db.Column(db.String(20))  # 'email', 'google', 'apple'
    oauth_id = db.Column(db.String(100))  # ID from OAuth provider
    
    # Add new fields for client information
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    company_name = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    
    # Add subscription relationship
    subscription = db.relationship('Subscription', backref='user', uselist=False)

    # FIX: Specify the foreign key for the relationship
    chatbots = db.relationship(
        'Chatbot',
        foreign_keys='Chatbot.user_id', # Tell SQLAlchemy to use Chatbot.user_id for the join
        backref='owner',
        lazy='dynamic'
    )

    def __repr__(self):
        return f'<User {self.email} client_id={self.client_id}>'
    
    def get_full_name(self):
        """Return the user's full name or email if not available"""
        if self.first_name or self.last_name:
            return f"{self.first_name or ''} {self.last_name or ''}".strip()
        return self.email
    
    def get_subscription_status(self):
        """Get the current subscription status"""
        if not self.subscription:
            return "No Subscription"
        return self.subscription.status
        
    def set_password(self, password):
        """Set the password hash"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Check the password hash"""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    def set_oauth_data(self, provider, oauth_id):
        """Set OAuth provider data"""
        self.auth_provider = provider
        self.oauth_id = oauth_id
        self.is_verified = True  # OAuth users are automatically verified

class Chatbot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # This is the primary FK link
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # This is a secondary link, useful for direct filtering perhaps
    client_id = db.Column(db.String(36), db.ForeignKey('user.client_id'), nullable=False, index=True)

    status = db.Column(db.String(50), default='Pending', nullable=False)
    source_type = db.Column(db.String(50))
    source_details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # New fields for index operation tracking
    index_operation_id = db.Column(db.String(100))  # Store the operation ID from Vertex AI
    index_operation_state = db.Column(db.String(50))  # Current state of the index operation
    index_operation_started_at = db.Column(db.DateTime)  # When the index operation started
    index_operation_completed_at = db.Column(db.DateTime)  # When the index operation completed
    index_operation_error = db.Column(db.Text)  # Store any error messages from the operation
    index_operation_progress = db.Column(db.Integer)  # Progress percentage (0-100)
    index_operation_metadata = db.Column(db.Text)  # Store additional operation metadata as JSON
    last_index_update = db.Column(db.DateTime)  # Last successful index update timestamp
    total_chunks_indexed = db.Column(db.Integer, default=0)  # Total number of chunks indexed
    index_version = db.Column(db.String(50))  # Version/timestamp of the current index
    
    # API Key for widget authentication
    api_key = db.Column(db.String(64), unique=True, index=True, nullable=True) # Nullable initially

    # Widget Customization Fields
    widget_primary_color = db.Column(db.String(7), nullable=True) # e.g., '#007bff'
    widget_text_color = db.Column(db.String(7), nullable=True) # e.g., '#ffffff'
    widget_welcome_message = db.Column(db.String(500), nullable=True)
    logo_path = db.Column(db.String(255), nullable=True) # Path to the custom logo file
    launcher_text = db.Column(db.String(100), nullable=True) # Text for the closed launcher button
    launcher_icon_path = db.Column(db.String(255), nullable=True) # Path to the custom launcher icon file

    widget_position = db.Column(db.String(50), nullable=True, default='bottom-right') # e.g., 'bottom-right', 'bottom-left', 'top-right', 'top-left'
    avatar_path = db.Column(db.String(255), nullable=True) # Path to the custom avatar file

    widget_background_color = db.Column(db.String(7), nullable=True) # e.g., '#ffffff'
    user_message_color = db.Column(db.String(7), nullable=True) # e.g., '#d1e7dd'
    bot_message_color = db.Column(db.String(7), nullable=True) # e.g., '#f8d7da'
    input_background_color = db.Column(db.String(7), nullable=True) # e.g., '#f0f0f0'
    # Removed voice_enabled, voice_input_language, voice_output_language, voice_profile, voice_speed, vad_enabled

    voice_enabled = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable voice input/output
    text_chat_enabled = db.Column(db.Boolean, default=True, nullable=False) # Enable/disable text chat input/output
    text_language = db.Column(db.String(10), default='en', nullable=False) # Primary language for text chat (e.g., 'en', 'es', 'fr')
    source_document_language = db.Column(db.String(10), nullable=True, default='en') # Language of the ingested RAG source documents (ISO 639-1)


    file_uploads_enabled = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable file uploads
    allowed_file_types = db.Column(db.String(500), nullable=True) # Comma-separated MIME types (e.g., 'image/png,application/pdf')
    max_file_size_mb = db.Column(db.Integer, nullable=True, default=10) # Max file size in MB
    save_history_enabled = db.Column(db.Boolean, default=True, nullable=False) # Enable/disable saving chat history
    history_retention_days = db.Column(db.Integer, nullable=True) # Days to retain history (null for indefinite)
    allow_user_history_clearing = db.Column(db.Boolean, default=False, nullable=False) # Allow end-users to clear their session history
    image_analysis_enabled = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable image analysis feature
    summarization_enabled = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable summarization feature
    allowed_scraping_domains = db.Column(db.Text, nullable=True) # Domains allowed for URL summarization (comma/newline separated)

    # User Feedback Settings
    feedback_thumbs_enabled = db.Column(db.Boolean, default=True, nullable=False) # Enable/disable thumbs up/down feedback
    detailed_feedback_enabled = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable detailed feedback form
    vad_enabled = db.Column(db.Boolean, default=True, nullable=False) # Enable/disable Voice Activity Detection

    # UI Element Visibility
    show_widget_header = db.Column(db.Boolean, default=True, nullable=False) # Toggle visibility of the widget header
    show_message_timestamps = db.Column(db.Boolean, default=True, nullable=False) # Toggle visibility of message timestamps

    start_open = db.Column(db.Boolean, default=False, nullable=False) # Widget starts open or closed
    show_typing_indicator = db.Column(db.Boolean, default=True, nullable=False) # Toggle visibility of the 'Thinking...' indicator
    default_error_message = db.Column(db.String(500), nullable=True, default='Sorry, an error occurred. Please try again.') # Default error message for the widget
    fallback_message = db.Column(db.String(500), nullable=True, default="Sorry, I don't have an answer for that right now.") # Fallback message
    # Simulated response delay
    response_delay_ms = db.Column(db.Integer, default=0, nullable=False) # Delay in milliseconds, non-negative validation in API
    enable_sound_notifications = db.Column(db.Boolean, default=False, nullable=False) # Enable/disable sound notifications for new messages

    base_prompt = db.Column(db.Text, nullable=True) # Custom base instructions/system prompt for the LLM
    # New field for knowledge adherence level
    knowledge_adherence_level = db.Column(db.String(20), nullable=False, default='strict') # Expected values: 'strict', 'moderate', 'flexible'
    # NEW FIELDS FOR CONSENT
    consent_message = db.Column(db.Text, nullable=True) # Allow null or provide default in API
    consent_required = db.Column(db.Boolean, default=False, nullable=False)

    # New field for Advanced RAG
    advanced_rag_enabled = db.Column(db.Boolean, default=False, nullable=False)
 
    def __repr__(self):
        return f'<Chatbot {self.name} id={self.id} client_id={self.client_id}>'

    def update_index_operation_status(self, operation_id, state, error=None, progress=None, metadata=None):
        """Update the status of an ongoing index operation."""
        self.index_operation_id = operation_id
        self.index_operation_state = state
        self.index_operation_error = error
        self.index_operation_progress = progress
        if metadata:
            self.index_operation_metadata = metadata

        if state == 'RUNNING' and not self.index_operation_started_at:
            self.index_operation_started_at = datetime.utcnow()
        elif state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            self.index_operation_completed_at = datetime.utcnow()
            if state == 'SUCCEEDED':
                self.last_index_update = datetime.utcnow()
                self.status = 'Ready'
            elif state == 'FAILED':
                self.status = 'Failed'
            elif state == 'CANCELLED':
                self.status = 'Cancelled'

        db.session.commit()

    def start_index_operation(self, operation_id):
        """Start tracking a new index operation."""
        self.index_operation_id = operation_id
        self.index_operation_state = 'RUNNING'
        self.index_operation_started_at = datetime.utcnow()
        self.index_operation_progress = 0
        self.status = 'Processing - Indexing Data'
        db.session.commit()

    def complete_index_operation(self, success=True, error=None, total_chunks=None):
        """Complete an index operation with success or failure."""
        self.index_operation_completed_at = datetime.utcnow()
        if success:
            self.status = 'Ready'
            self.last_index_update = datetime.utcnow()
            if total_chunks is not None:
                self.total_chunks_indexed = total_chunks
        else:
            self.status = 'Failed'
            self.index_operation_error = error
        
        final_status = self.status # Get the status that was just set
        client_id = self.client_id # Get the client_id
        chatbot_id = self.id # Get the id

        db.session.commit() # Commit the status change first

        # Now push the final status update via SSE
        try:
            from app.sse_utils import push_status_update
            push_status_update(chatbot_id, final_status, client_id)
        except Exception as e:
            # Log error if push fails, but don't rollback the commit
            # Use current_app logger if possible, or basic print
            try: 
                from flask import current_app
                current_app.logger.error(f"Failed SSE push after index completion for chatbot {chatbot_id}: {e}")
            except:
                print(f"ERROR: Failed SSE push after index completion for chatbot {chatbot_id}: {e}")


    def get_index_operation_status(self):
        """Get the current status of the index operation."""
        return {
            'operation_id': self.index_operation_id,
            'state': self.index_operation_state,
            'progress': self.index_operation_progress,
            'started_at': self.index_operation_started_at,
            'completed_at': self.index_operation_completed_at,
            'error': self.index_operation_error,
            'metadata': self.index_operation_metadata
        }

    def is_index_operation_in_progress(self):
        """Check if there's an ongoing index operation."""
        return (self.index_operation_state == 'RUNNING' and 
                self.index_operation_started_at and 
                not self.index_operation_completed_at)

    def get_index_stats(self):
        """Get statistics about the chatbot's index."""
        return {
            'total_chunks': self.total_chunks_indexed,
            'last_update': self.last_index_update,
            'index_version': self.index_version,
            'status': self.status
        }

class Plan(db.Model):
    """Subscription plan model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text)
    price = db.Column(db.Float, nullable=False)
    billing_cycle = db.Column(db.String(20), nullable=False, default='monthly')  # monthly, quarterly, yearly
    
    # Plan features and limits
    max_chatbots = db.Column(db.Integer, default=1)
    max_storage_mb = db.Column(db.Integer, default=100)
    max_queries_per_month = db.Column(db.Integer, default=1000)
    max_pages_per_chatbot = db.Column(db.Integer, default=50)
    max_files_per_chatbot = db.Column(db.Integer, default=5)
    
    # Additional features as boolean flags
    custom_branding = db.Column(db.Boolean, default=False)
    priority_support = db.Column(db.Boolean, default=False)
    analytics_dashboard = db.Column(db.Boolean, default=False)
    api_access = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    subscriptions = db.relationship('Subscription', backref='plan', lazy='dynamic')
    
    def __repr__(self):
        return f'<Plan {self.name} ${self.price}/{self.billing_cycle}>'


class Subscription(db.Model):
    """User subscription model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_id = db.Column(db.Integer, db.ForeignKey('plan.id'), nullable=False)
    
    # Subscription status
    status = db.Column(db.String(20), nullable=False, default='active')  # active, canceled, expired, trial
    
    # Subscription dates
    start_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_date = db.Column(db.DateTime)
    trial_end_date = db.Column(db.DateTime)
    
    # Payment tracking
    last_payment_date = db.Column(db.DateTime)
    next_payment_date = db.Column(db.DateTime)
    payment_method = db.Column(db.String(50))
    
    # Usage tracking
    current_period_queries = db.Column(db.Integer, default=0)
    total_queries = db.Column(db.Integer, default=0)
    storage_used_mb = db.Column(db.Float, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Subscription {self.id} User={self.user_id} Plan={self.plan_id} Status={self.status}>'
    
    def is_active(self):
        """Check if subscription is currently active"""
        return self.status == 'active' and (not self.end_date or self.end_date > datetime.utcnow())
    
    def is_in_trial(self):
        """Check if subscription is in trial period"""
        return self.status == 'trial' and self.trial_end_date and self.trial_end_date > datetime.utcnow()
    
    def days_remaining(self):
        """Calculate days remaining in current subscription period"""
        if not self.end_date:
            return None
        delta = self.end_date - datetime.utcnow()
        return max(0, delta.days)
    
    def record_query(self):
        """Record a query against this subscription"""
        self.current_period_queries += 1
        self.total_queries += 1
        db.session.commit()


class PaymentHistory(db.Model):
    """Payment history for subscriptions"""
    id = db.Column(db.Integer, primary_key=True)
    subscription_id = db.Column(db.Integer, db.ForeignKey('subscription.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(3), default='USD')
    payment_date = db.Column(db.DateTime, default=datetime.utcnow)
    payment_method = db.Column(db.String(50))
    transaction_id = db.Column(db.String(100))
    status = db.Column(db.String(20), default='completed')  # completed, failed, refunded
    
    # Relationship
    subscription = db.relationship('Subscription', backref='payments')
    
    def __repr__(self):
        return f'<Payment {self.id} Amount=${self.amount} Status={self.status}>'


class UsageLog(db.Model):
    """Log of subscription usage"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)  # query, index, storage
    action_details = db.Column(db.Text)
    resource_usage = db.Column(db.Float, default=0)  # Amount of resource used (queries, storage in MB, etc.)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='usage_logs')
    chatbot = db.relationship('Chatbot', backref='usage_logs')
    
    def __repr__(self):
        return f'<UsageLog {self.id} User={self.user_id} Action={self.action_type}>'

class VectorIdMapping(db.Model):
    """Maps chatbot IDs to their corresponding vector IDs in the index."""
    id = db.Column(db.Integer, primary_key=True)
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'), nullable=False, index=True)
    vector_id = db.Column(db.String(128), nullable=False, index=True) 
    source_identifier = db.Column(db.String(2048), nullable=True, index=True) # Store URL or filename
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Add relationship back to Chatbot if needed (optional)
    # chatbot = db.relationship('Chatbot', backref='vector_mappings')

    def __repr__(self):
        return f'<VectorIdMapping c:{self.chatbot_id} v:{self.vector_id}>'


class ChatMessage(db.Model):
    """Stores individual messages from a chat session."""
    id = db.Column(db.Integer, primary_key=True)
    chatbot_id = db.Column(db.Integer, db.ForeignKey('chatbot.id'), nullable=False, index=True)
    # Session ID to group messages within a single user interaction/session
    session_id = db.Column(db.String(64), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Feedback field
    thumb_feedback = db.Column(db.String(4), nullable=True) # 'up' or 'down'

    # Relationship back to Chatbot (optional, but can be useful)
    chatbot = db.relationship('Chatbot', backref=db.backref('chat_messages', lazy='dynamic'))

    def __repr__(self):
        return f'<ChatMessage {self.id} Chatbot={self.chatbot_id} Role={self.role} Time={self.timestamp}>'



# New DetailedFeedback model
class DetailedFeedback(db.Model):
    """Stores detailed feedback provided by users for specific messages."""
    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(db.Integer, db.ForeignKey('chat_message.id'), nullable=False, index=True)
    session_id = db.Column(db.String(64), nullable=False, index=True) # Denormalized for easier querying if needed
    feedback_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Relationship back to ChatMessage
    message = db.relationship('ChatMessage', backref=db.backref('detailed_feedback', uselist=False)) # Assuming one detailed feedback per message

    def __repr__(self):
        return f'<DetailedFeedback {self.id} for Message={self.message_id}>'

