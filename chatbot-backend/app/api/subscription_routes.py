"""
API routes for subscription management
"""
from flask import Blueprint, request, jsonify, current_app
from app import db
from app.models import User, Plan, Subscription, PaymentHistory, UsageLog
from app.services.subscription_service import SubscriptionService
from app.services.email_service import EmailService
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json

# Create blueprint
bp = Blueprint('subscription', __name__)

@bp.route('/plans', methods=['GET'])
def get_plans():
    """Get all available subscription plans"""
    try:
        plans = SubscriptionService.get_all_plans()
        
        # If no plans exist, create default plans
        if not plans:
            success = SubscriptionService.create_default_plans()
            if success:
                plans = SubscriptionService.get_all_plans()
        
        return jsonify({
            'success': True,
            'plans': [{
                'id': plan.id,
                'name': plan.name,
                'description': plan.description,
                'price': plan.price,
                'billing_cycle': plan.billing_cycle,
                'features': {
                    'max_chatbots': plan.max_chatbots,
                    'max_storage_mb': plan.max_storage_mb,
                    'max_queries_per_month': plan.max_queries_per_month,
                    'max_pages_per_chatbot': plan.max_pages_per_chatbot,
                    'max_files_per_chatbot': plan.max_files_per_chatbot,
                    'custom_branding': plan.custom_branding,
                    'priority_support': plan.priority_support,
                    'analytics_dashboard': plan.analytics_dashboard,
                    'api_access': plan.api_access
                }
            } for plan in plans]
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting plans: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to retrieve subscription plans',
            'error': str(e)
        }), 500

@bp.route('/plans/<int:plan_id>', methods=['GET'])
def get_plan(plan_id):
    """Get a specific subscription plan"""
    try:
        plan = SubscriptionService.get_plan_by_id(plan_id)
        
        if not plan:
            return jsonify({
                'success': False,
                'message': f'Plan with ID {plan_id} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'plan': {
                'id': plan.id,
                'name': plan.name,
                'description': plan.description,
                'price': plan.price,
                'billing_cycle': plan.billing_cycle,
                'features': {
                    'max_chatbots': plan.max_chatbots,
                    'max_storage_mb': plan.max_storage_mb,
                    'max_queries_per_month': plan.max_queries_per_month,
                    'max_pages_per_chatbot': plan.max_pages_per_chatbot,
                    'max_files_per_chatbot': plan.max_files_per_chatbot,
                    'custom_branding': plan.custom_branding,
                    'priority_support': plan.priority_support,
                    'analytics_dashboard': plan.analytics_dashboard,
                    'api_access': plan.api_access
                }
            }
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting plan: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to retrieve subscription plan',
            'error': str(e)
        }), 500

@bp.route('/subscribe', methods=['POST'])
@jwt_required()
def subscribe():
    """Subscribe a user to a plan"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        # Get user from JWT token
        user_id = get_jwt_identity()
        plan_id = data.get('plan_id')
        payment_method = data.get('payment_method')
        
        if not plan_id:
            return jsonify({
                'success': False,
                'message': 'Missing required field: plan_id'
            }), 400
        
        # Get user and plan
        user = User.query.get(user_id)
        plan = Plan.query.get(plan_id)
        
        if not user:
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
            
        if not plan:
            return jsonify({
                'success': False,
                'message': 'Plan not found'
            }), 404
        
        success, result = SubscriptionService.create_subscription(
            user_id=user_id,
            plan_id=plan_id,
            trial_days=0,  # No trial
            payment_method=payment_method
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to create subscription',
                'error': result
            }), 400
        
        # Get the subscription details
        subscription_details = SubscriptionService.get_user_subscription_details(user_id)
        
        # Send subscription confirmation email
        if plan.price > 0:  # Only send for paid plans
            EmailService.send_subscription_confirmation(user, result, plan)
        
        return jsonify({
            'success': True,
            'message': 'Subscription created successfully',
            'subscription': subscription_details
        }), 201
    except Exception as e:
        current_app.logger.error(f"Error creating subscription: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to create subscription',
            'error': str(e)
        }), 500

@bp.route('/subscriptions/<int:subscription_id>/cancel', methods=['POST'])
def cancel_subscription(subscription_id):
    """Cancel a subscription"""
    try:
        success, result = SubscriptionService.cancel_subscription(subscription_id)
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to cancel subscription',
                'error': result
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Subscription canceled successfully'
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error canceling subscription: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to cancel subscription',
            'error': str(e)
        }), 500

@bp.route('/users/<int:user_id>/subscription', methods=['GET'])
def get_user_subscription(user_id):
    """Get subscription details for a user"""
    try:
        subscription_details = SubscriptionService.get_user_subscription_details(user_id)
        
        if not subscription_details:
            return jsonify({
                'success': False,
                'message': f'No subscription found for user {user_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'subscription': subscription_details
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error getting user subscription: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to retrieve subscription details',
            'error': str(e)
        }), 500

@bp.route('/users/<int:user_id>/check-limits', methods=['GET'])
def check_limits(user_id):
    """Check if a user has reached their subscription limits"""
    try:
        action_type = request.args.get('action_type', 'query')
        
        allowed, message = SubscriptionService.check_subscription_limits(
            user_id=user_id,
            action_type=action_type
        )
        
        return jsonify({
            'success': True,
            'allowed': allowed,
            'message': message
        }), 200
    except Exception as e:
        current_app.logger.error(f"Error checking subscription limits: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to check subscription limits',
            'error': str(e)
        }), 500

@bp.route('/users/<int:user_id>/record-usage', methods=['POST'])
def record_usage(user_id):
    """Record usage of a resource"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        chatbot_id = data.get('chatbot_id')
        action_type = data.get('action_type')
        details = data.get('details')
        resource_amount = data.get('resource_amount', 1.0)
        
        if not chatbot_id or not action_type:
            return jsonify({
                'success': False,
                'message': 'Missing required fields: chatbot_id, action_type'
            }), 400
        
        success, result = SubscriptionService.record_usage(
            user_id=user_id,
            chatbot_id=chatbot_id,
            action_type=action_type,
            details=details,
            resource_amount=resource_amount
        )
        
        if not success:
            return jsonify({
                'success': False,
                'message': 'Failed to record usage',
                'error': result
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'Usage recorded successfully'
        }), 201
    except Exception as e:
        current_app.logger.error(f"Error recording usage: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Failed to record usage',
            'error': str(e)
        }), 500
