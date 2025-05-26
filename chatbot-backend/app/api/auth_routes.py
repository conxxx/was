"""
API routes for authentication
"""
from flask import Blueprint, request, jsonify, current_app, url_for
from app import db
from app.models import User
from app.services.auth_service import AuthService
from app.services.email_service import EmailService
from flask_jwt_extended import jwt_required, get_jwt_identity
import json

# Create blueprint
bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['POST'])
def register():
    """Register a new user with email and password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        email = data.get('email')
        password = data.get('password')
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: email, password'
            }), 400
        
        success, result = AuthService.register_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            auth_provider='email'
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to register user',
                'error': result
            }), 400
        
        # Generate verification URL
        verification_token = result.verification_token
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        verification_url = f"{frontend_url}/verify-email/{verification_token}"
        
        # Send verification email
        EmailService.send_verification_email(result, verification_url)
        
        return jsonify({
            'success': True,
            'message': 'User registered successfully. Please check your email to verify your account.',
            'user': {
                'id': result.id,
                'email': result.email,
                'client_id': result.client_id,
                'first_name': result.first_name,
                'last_name': result.last_name,
                'is_verified': result.is_verified
            }
        }), 201
    except Exception as e:
        current_app.logger.error(f"Error registering user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to register user',
            'error': str(e)
        }), 500

@bp.route('/login', methods=['POST'])
def login():
    """Login a user with email and password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: email, password'
            }), 400
        
        success, result, token = AuthService.login_user(
            email=email,
            password=password,
            auth_provider='email'
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': result
            }), 401
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': result.id,
                'email': result.email,
                'client_id': result.client_id,
                'first_name': result.first_name,
                'last_name': result.last_name,
                'is_verified': result.is_verified
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error logging in user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to login',
            'error': str(e)
        }), 500

@bp.route('/verify-email/<token>', methods=['GET'])
def verify_email(token):
    """Verify a user's email address"""
    try:
        success, message = AuthService.verify_email(token)
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
        # Send welcome email after successful verification
        email = AuthService.verify_token(token)
        if email:
            user = User.query.filter_by(email=email).first()
            if user:
                EmailService.send_welcome_email(user)
        
        return jsonify({
            'success': True,
            'message': message
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error verifying email: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to verify email',
            'error': str(e)
        }), 500

@bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Request a password reset"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        email = data.get('email')
        
        if not email:
            return jsonify({
                'success': False,
                'message': 'Missing required field: email'
            }), 400
        
        success, message, token = AuthService.request_password_reset(email)
        
        if not success:
            # Don't reveal if the email exists or not for security
            return jsonify({
                'success': True,
                'message': 'If your email is registered, you will receive password reset instructions.'
            }), 200
        
        # Generate reset URL
        frontend_url = current_app.config.get('FRONTEND_URL', 'http://localhost:3000')
        reset_url = f"{frontend_url}/reset-password/{token}"
        
        # Send password reset email
        user = User.query.filter_by(email=email).first()
        if user:
            EmailService.send_password_reset_email(user, reset_url)
        
        return jsonify({
            'success': True,
            'message': 'If your email is registered, you will receive password reset instructions.'
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error requesting password reset: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to request password reset',
            'error': str(e)
        }), 500

@bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset a user's password"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        token = data.get('token')
        new_password = data.get('new_password')
        
        if not token or not new_password:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: token, new_password'
            }), 400
        
        success, message = AuthService.reset_password(token, new_password)
        
        if not success:
            return jsonify({
                'success': False,
                'message': message
            }), 400
        
        return jsonify({
            'success': True,
            'message': message
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error resetting password: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to reset password',
            'error': str(e)
        }), 500

@bp.route('/google-login', methods=['POST'])
def google_login():
    """Login or register a user with Google OAuth"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        token = data.get('token')
        
        if not token:
            return jsonify({
                'success': False,
                'message': 'Missing required field: token'
            }), 400
        
        # Verify Google token
        success, user_info = AuthService.verify_google_token(token)
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Invalid Google token',
                'error': user_info
            }), 400
        
        # Check if user exists
        user = User.query.filter_by(email=user_info['email']).first()
        
        if user:
            # Login existing user
            success, user, token = AuthService.login_user(
                email=user_info['email'],
                auth_provider='google',
                oauth_id=user_info['oauth_id']
            )
        else:
            # Register new user
            success, user = AuthService.register_user(
                email=user_info['email'],
                first_name=user_info['first_name'],
                last_name=user_info['last_name'],
                auth_provider='google',
                oauth_id=user_info['oauth_id'],
                auto_verify=True
            )
            
            if success:
                # Generate token for new user
                _, user, token = AuthService.login_user(
                    email=user_info['email'],
                    auth_provider='google',
                    oauth_id=user_info['oauth_id']
                )
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to authenticate with Google',
                'error': user if isinstance(user, str) else "Authentication failed"
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Google authentication successful',
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'client_id': user.client_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'is_verified': user.is_verified
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error with Google login: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to authenticate with Google',
            'error': str(e)
        }), 500

@bp.route('/apple-login', methods=['POST'])
def apple_login():
    """Login or register a user with Apple OAuth"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        token = data.get('token')
        user_data = data.get('user_data', {})  # Apple may provide additional user data on first login
        
        if not token:
            return jsonify({
                'success': False,
                'message': 'Missing required field: token'
            }), 400
        
        # Verify Apple token
        success, user_info = AuthService.verify_apple_token(token)
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Invalid Apple token',
                'error': user_info
            }), 400
        
        # Apple might not provide name in the token, so use from user_data if available
        first_name = user_data.get('first_name')
        last_name = user_data.get('last_name')
        
        # Check if user exists
        user = User.query.filter_by(email=user_info['email']).first()
        
        if user:
            # Login existing user
            success, user, token = AuthService.login_user(
                email=user_info['email'],
                auth_provider='apple',
                oauth_id=user_info['oauth_id']
            )
        else:
            # Register new user
            success, user = AuthService.register_user(
                email=user_info['email'],
                first_name=first_name,
                last_name=last_name,
                auth_provider='apple',
                oauth_id=user_info['oauth_id'],
                auto_verify=True
            )
            
            if success:
                # Generate token for new user
                _, user, token = AuthService.login_user(
                    email=user_info['email'],
                    auth_provider='apple',
                    oauth_id=user_info['oauth_id']
                )
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to authenticate with Apple',
                'error': user if isinstance(user, str) else "Authentication failed"
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Apple authentication successful',
            'token': token,
            'user': {
                'id': user.id,
                'email': user.email,
                'client_id': user.client_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'is_verified': user.is_verified
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error with Apple login: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to authenticate with Apple',
            'error': str(e)
        }), 500

@bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get the current authenticated user"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'email': user.email,
                'client_id': user.client_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'company_name': user.company_name,
                'phone': user.phone,
                'is_verified': user.is_verified,
                'auth_provider': user.auth_provider,
                'created_at': user.created_at.isoformat() if user.created_at else None,
                'last_login': user.last_login.isoformat() if user.last_login else None
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting current user: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to get user information',
            'error': str(e)
        }), 500

@bp.route('/update-profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update the current user's profile"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Update user fields
        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        if 'company_name' in data:
            user.company_name = data['company_name']
        if 'phone' in data:
            user.phone = data['phone']
        
        # Update password if provided and using email authentication
        if 'current_password' in data and 'new_password' in data and user.auth_provider == 'email':
            if not user.check_password(data['current_password']):
                return jsonify({
                    'success': False,
                    'message': 'Current password is incorrect'
                }), 400
            
            user.set_password(data['new_password'])
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'user': {
                'id': user.id,
                'email': user.email,
                'client_id': user.client_id,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'company_name': user.company_name,
                'phone': user.phone,
                'is_verified': user.is_verified,
                'auth_provider': user.auth_provider
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error updating profile: {str(e)}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': 'Failed to update profile',
            'error': str(e)
        }), 500
