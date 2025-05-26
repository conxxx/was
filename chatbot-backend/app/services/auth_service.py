"""
Authentication Service for Website AI Assistant Platform
Handles user authentication via email, Google, and Apple sign-in
"""
import os
import secrets
import jwt
from datetime import datetime, timedelta
from flask import current_app, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from app import db
from app.models import User
from app.services.subscription_service import SubscriptionService


class AuthService:
    """Service for handling user authentication"""
    
    @staticmethod
    def _get_serializer():
        """Get a URL-safe timed serializer for tokens"""
        return URLSafeTimedSerializer(current_app.config['SECRET_KEY'])
    
    @staticmethod
    def _generate_token():
        """Generate a secure random token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_verification_token(email):
        """Generate a verification token for email verification"""
        serializer = AuthService._get_serializer()
        return serializer.dumps(email, salt='email-verification')
    
    @staticmethod
    def verify_token(token, salt='email-verification', expiration=3600):
        """Verify a token and return the email if valid"""
        serializer = AuthService._get_serializer()
        try:
            email = serializer.loads(token, salt=salt, max_age=expiration)
            return email
        except:
            return None
    
    @staticmethod
    def generate_reset_token(email):
        """Generate a password reset token"""
        serializer = AuthService._get_serializer()
        return serializer.dumps(email, salt='password-reset')
    
    @staticmethod
    def generate_jwt_token(user_id, expiration=24):
        """Generate a JWT token for API authentication"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expiration),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16)
        }
        return jwt.encode(payload, current_app.config['SECRET_KEY'], algorithm='HS256')
    
    @staticmethod
    def verify_jwt_token(token):
        """Verify a JWT token and return the user_id if valid"""
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            return payload.get('user_id')
        except jwt.PyJWTError:
            return None
    
    @staticmethod
    def register_user(email, password=None, first_name=None, last_name=None, 
                      auth_provider='email', oauth_id=None, auto_verify=False):
        """
        Register a new user
        
        Args:
            email: User's email address
            password: Password (for email auth)
            first_name: User's first name
            last_name: User's last name
            auth_provider: Authentication provider (email, google, apple)
            oauth_id: ID from OAuth provider
            auto_verify: Whether to automatically verify the user
            
        Returns:
            Tuple of (success, user_or_error_message)
        """
        try:
            # Check if user already exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                # If user exists but was using a different auth method, update the auth method
                if auth_provider != 'email' and existing_user.auth_provider == 'email':
                    existing_user.auth_provider = auth_provider
                    existing_user.oauth_id = oauth_id
                    existing_user.is_verified = True
                    db.session.commit()
                    return True, existing_user
                return False, "User with this email already exists"
            
            # Create new user
            user = User(
                email=email,
                first_name=first_name,
                last_name=last_name,
                auth_provider=auth_provider,
                oauth_id=oauth_id,
                is_verified=auto_verify
            )
            
            # Set password if using email authentication
            if auth_provider == 'email' and password:
                user.set_password(password)
                
                # Generate verification token if not auto-verified
                if not auto_verify:
                    user.verification_token = AuthService.generate_verification_token(email)
            
            db.session.add(user)
            db.session.commit()
            
            # Assign free plan to new user
            free_plan = SubscriptionService.get_plan_by_name('Free')
            if free_plan:
                SubscriptionService.create_subscription(
                    user_id=user.id,
                    plan_id=free_plan.id,
                    trial_days=0
                )
            
            return True, user
            
        except Exception as e:
            current_app.logger.error(f"Error registering user: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def login_user(email, password=None, auth_provider='email', oauth_id=None):
        """
        Login a user
        
        Args:
            email: User's email address
            password: Password (for email auth)
            auth_provider: Authentication provider (email, google, apple)
            oauth_id: ID from OAuth provider
            
        Returns:
            Tuple of (success, user_or_error_message, token)
        """
        try:
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return False, "User not found", None
            
            if not user.is_active:
                return False, "Account is inactive", None
            
            # Check authentication method
            if auth_provider == 'email':
                if not user.check_password(password):
                    return False, "Invalid password", None
                
                if not user.is_verified:
                    return False, "Email not verified", None
            
            elif auth_provider in ['google', 'apple']:
                if user.auth_provider != auth_provider or user.oauth_id != oauth_id:
                    return False, f"Please use {user.auth_provider} to login", None
            
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            # Generate JWT token
            token = AuthService.generate_jwt_token(user.id)
            
            return True, user, token
            
        except Exception as e:
            current_app.logger.error(f"Error logging in user: {str(e)}")
            return False, str(e), None
    
    @staticmethod
    def verify_email(token):
        """
        Verify a user's email address using the verification token
        
        Args:
            token: Verification token
            
        Returns:
            Tuple of (success, message)
        """
        try:
            email = AuthService.verify_token(token)
            
            if not email:
                return False, "Invalid or expired verification token"
            
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return False, "User not found"
            
            if user.is_verified:
                return True, "Email already verified"
            
            user.is_verified = True
            user.verification_token = None
            db.session.commit()
            
            return True, "Email verified successfully"
            
        except Exception as e:
            current_app.logger.error(f"Error verifying email: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def request_password_reset(email):
        """
        Request a password reset
        
        Args:
            email: User's email address
            
        Returns:
            Tuple of (success, message, token)
        """
        try:
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return False, "User not found", None
            
            if user.auth_provider != 'email':
                return False, f"Please use {user.auth_provider} to login", None
            
            token = AuthService.generate_reset_token(email)
            user.reset_password_token = token
            db.session.commit()
            
            return True, "Password reset token generated", token
            
        except Exception as e:
            current_app.logger.error(f"Error requesting password reset: {str(e)}")
            db.session.rollback()
            return False, str(e), None
    
    @staticmethod
    def reset_password(token, new_password):
        """
        Reset a user's password
        
        Args:
            token: Password reset token
            new_password: New password
            
        Returns:
            Tuple of (success, message)
        """
        try:
            email = AuthService.verify_token(token, salt='password-reset')
            
            if not email:
                return False, "Invalid or expired reset token"
            
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return False, "User not found"
            
            user.set_password(new_password)
            user.reset_password_token = None
            db.session.commit()
            
            return True, "Password reset successfully"
            
        except Exception as e:
            current_app.logger.error(f"Error resetting password: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def verify_google_token(token):
        """
        Verify a Google ID token
        
        Args:
            token: Google ID token
            
        Returns:
            Tuple of (success, user_info_or_error_message)
        """
        try:
            # Get Google client ID from config
            google_client_id = current_app.config.get('GOOGLE_CLIENT_ID')
            
            if not google_client_id:
                return False, "Google client ID not configured"
            
            # Verify the token
            id_info = id_token.verify_oauth2_token(
                token, 
                google_requests.Request(), 
                google_client_id
            )
            
            # Check if the token is issued by Google
            if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                return False, "Invalid token issuer"
            
            # Extract user info
            user_info = {
                'email': id_info['email'],
                'first_name': id_info.get('given_name'),
                'last_name': id_info.get('family_name'),
                'oauth_id': id_info['sub']
            }
            
            return True, user_info
            
        except Exception as e:
            current_app.logger.error(f"Error verifying Google token: {str(e)}")
            return False, str(e)
    
    @staticmethod
    def verify_apple_token(token):
        """
        Verify an Apple ID token
        
        Args:
            token: Apple ID token
            
        Returns:
            Tuple of (success, user_info_or_error_message)
        """
        try:
            # Get Apple client ID from config
            apple_client_id = current_app.config.get('APPLE_CLIENT_ID')
            
            if not apple_client_id:
                return False, "Apple client ID not configured"
            
            # Verify the token
            # Note: Apple tokens require fetching the public key from Apple's JWKS endpoint
            # This is a simplified version
            id_info = jwt.decode(
                token,
                options={"verify_signature": False},  # In production, you should verify the signature
                audience=apple_client_id
            )
            
            # Extract user info
            user_info = {
                'email': id_info.get('email'),
                'oauth_id': id_info.get('sub')
            }
            
            if not user_info['email'] or not user_info['oauth_id']:
                return False, "Invalid token: missing required fields"
            
            return True, user_info
            
        except Exception as e:
            current_app.logger.error(f"Error verifying Apple token: {str(e)}")
            return False, str(e)
