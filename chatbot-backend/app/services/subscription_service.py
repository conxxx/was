"""
Subscription Service for Website AI Assistant Platform
Manages subscription plans, user subscriptions, and usage tracking
"""
from datetime import datetime, timedelta
import json
from app import db
from app.models import User, Plan, Subscription, PaymentHistory, UsageLog, Chatbot
from sqlalchemy.exc import SQLAlchemyError
from flask import current_app


class SubscriptionService:
    """Service for managing subscription-related operations"""
    
    @staticmethod
    def create_default_plans():
        """Create default subscription plans if they don't exist"""
        try:
            # Check if plans already exist
            if Plan.query.count() > 0:
                current_app.logger.info("Default plans already exist")
                return True
                
            # Create default plans
            plans = [
                {
                    "name": "Free",
                    "description": "Basic plan with limited features",
                    "price": 0.00,
                    "billing_cycle": "monthly",
                    "max_chatbots": 1,
                    "max_storage_mb": 50,
                    "max_queries_per_month": 100,
                    "max_pages_per_chatbot": 20,
                    "max_files_per_chatbot": 2,
                    "custom_branding": False,
                    "priority_support": False,
                    "analytics_dashboard": False,
                    "api_access": False
                },
                {
                    "name": "Basic",
                    "description": "Standard plan for small businesses",
                    "price": 29.99,
                    "billing_cycle": "monthly",
                    "max_chatbots": 3,
                    "max_storage_mb": 500,
                    "max_queries_per_month": 1000,
                    "max_pages_per_chatbot": 100,
                    "max_files_per_chatbot": 10,
                    "custom_branding": True,
                    "priority_support": False,
                    "analytics_dashboard": True,
                    "api_access": False
                },
                {
                    "name": "Professional",
                    "description": "Advanced plan for growing businesses",
                    "price": 79.99,
                    "billing_cycle": "monthly",
                    "max_chatbots": 10,
                    "max_storage_mb": 2000,
                    "max_queries_per_month": 5000,
                    "max_pages_per_chatbot": 500,
                    "max_files_per_chatbot": 50,
                    "custom_branding": True,
                    "priority_support": True,
                    "analytics_dashboard": True,
                    "api_access": True
                },
                {
                    "name": "Enterprise",
                    "description": "Full-featured plan for large organizations",
                    "price": 199.99,
                    "billing_cycle": "monthly",
                    "max_chatbots": 50,
                    "max_storage_mb": 10000,
                    "max_queries_per_month": 25000,
                    "max_pages_per_chatbot": 2000,
                    "max_files_per_chatbot": 200,
                    "custom_branding": True,
                    "priority_support": True,
                    "analytics_dashboard": True,
                    "api_access": True
                }
            ]
            
            for plan_data in plans:
                plan = Plan(**plan_data)
                db.session.add(plan)
            
            db.session.commit()
            current_app.logger.info(f"Created {len(plans)} default subscription plans")
            return True
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error creating default plans: {str(e)}")
            db.session.rollback()
            return False
    
    @staticmethod
    def get_all_plans():
        """Get all available subscription plans"""
        return Plan.query.all()
    
    @staticmethod
    def get_plan_by_id(plan_id):
        """Get a specific plan by ID"""
        return Plan.query.get(plan_id)
    
    @staticmethod
    def get_plan_by_name(plan_name):
        """Get a specific plan by name"""
        return Plan.query.filter_by(name=plan_name).first()
    
    @staticmethod
    def create_subscription(user_id, plan_id, trial_days=0, payment_method=None):
        """
        Create a new subscription for a user
        
        Args:
            user_id: ID of the user
            plan_id: ID of the plan
            trial_days: Number of trial days (0 for no trial)
            payment_method: Payment method information
            
        Returns:
            Tuple of (success, subscription_or_error_message)
        """
        try:
            user = User.query.get(user_id)
            plan = Plan.query.get(plan_id)
            
            if not user:
                return False, "User not found"
            
            if not plan:
                return False, "Plan not found"
            
            # Check if user already has a subscription
            existing_subscription = Subscription.query.filter_by(user_id=user_id).first()
            if existing_subscription:
                return False, "User already has an active subscription"
            
            # Create new subscription
            subscription = Subscription(
                user_id=user_id,
                plan_id=plan_id,
                payment_method=payment_method
            )
            
            # Set up trial if applicable
            if trial_days > 0:
                subscription.status = 'trial'
                subscription.trial_end_date = datetime.utcnow() + timedelta(days=trial_days)
            
            # Set up billing cycle
            if plan.billing_cycle == 'monthly':
                subscription.end_date = datetime.utcnow() + timedelta(days=30)
                subscription.next_payment_date = datetime.utcnow() + timedelta(days=30)
            elif plan.billing_cycle == 'quarterly':
                subscription.end_date = datetime.utcnow() + timedelta(days=90)
                subscription.next_payment_date = datetime.utcnow() + timedelta(days=90)
            elif plan.billing_cycle == 'yearly':
                subscription.end_date = datetime.utcnow() + timedelta(days=365)
                subscription.next_payment_date = datetime.utcnow() + timedelta(days=365)
            
            db.session.add(subscription)
            db.session.commit()
            
            # Record initial payment if not a trial and not free
            if plan.price > 0 and trial_days == 0:
                payment = PaymentHistory(
                    subscription_id=subscription.id,
                    amount=plan.price,
                    payment_method=payment_method,
                    status='completed'
                )
                db.session.add(payment)
                subscription.last_payment_date = datetime.utcnow()
                db.session.commit()
            
            return True, subscription
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error creating subscription: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def cancel_subscription(subscription_id):
        """Cancel a subscription"""
        try:
            subscription = Subscription.query.get(subscription_id)
            
            if not subscription:
                return False, "Subscription not found"
            
            subscription.status = 'canceled'
            db.session.commit()
            
            return True, subscription
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error canceling subscription: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def check_subscription_limits(user_id, action_type='query'):
        """
        Check if a user has reached their subscription limits
        
        Args:
            user_id: ID of the user
            action_type: Type of action to check (query, chatbot, storage)
            
        Returns:
            Tuple of (allowed, message)
        """
        try:
            user = User.query.get(user_id)
            
            if not user:
                return False, "User not found"
            
            subscription = Subscription.query.filter_by(user_id=user_id).first()
            
            # If no subscription, check if free plan exists and use that for limits
            if not subscription:
                free_plan = Plan.query.filter_by(name='Free').first()
                if not free_plan:
                    return False, "No subscription and no free plan available"
                
                # Check limits based on action type
                if action_type == 'query':
                    # For queries, we need to track usage manually
                    query_count = UsageLog.query.filter_by(
                        user_id=user_id, 
                        action_type='query',
                        timestamp=datetime.utcnow() - timedelta(days=30)
                    ).count()
                    
                    if query_count >= free_plan.max_queries_per_month:
                        return False, "Monthly query limit reached on free plan"
                
                elif action_type == 'chatbot':
                    chatbot_count = Chatbot.query.filter_by(user_id=user_id).count()
                    if chatbot_count >= free_plan.max_chatbots:
                        return False, f"Maximum chatbot limit of {free_plan.max_chatbots} reached on free plan"
                
                # If we got here, the action is allowed on the free plan
                return True, "Action allowed on free plan"
            
            # Check if subscription is active
            if not subscription.is_active() and not subscription.is_in_trial():
                return False, "Subscription is not active"
            
            plan = Plan.query.get(subscription.plan_id)
            
            # Check limits based on action type
            if action_type == 'query':
                if subscription.current_period_queries >= plan.max_queries_per_month:
                    return False, f"Monthly query limit of {plan.max_queries_per_month} reached"
            
            elif action_type == 'chatbot':
                chatbot_count = Chatbot.query.filter_by(user_id=user_id).count()
                if chatbot_count >= plan.max_chatbots:
                    return False, f"Maximum chatbot limit of {plan.max_chatbots} reached"
            
            elif action_type == 'storage':
                if subscription.storage_used_mb >= plan.max_storage_mb:
                    return False, f"Storage limit of {plan.max_storage_mb}MB reached"
            
            # If we got here, the action is allowed
            return True, "Action allowed"
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error checking subscription limits: {str(e)}")
            return False, str(e)
    
    @staticmethod
    def record_usage(user_id, chatbot_id, action_type, details=None, resource_amount=1.0):
        """
        Record usage of a resource
        
        Args:
            user_id: ID of the user
            chatbot_id: ID of the chatbot
            action_type: Type of action (query, index, storage)
            details: Additional details about the action
            resource_amount: Amount of resource used
            
        Returns:
            Tuple of (success, message_or_log)
        """
        try:
            # Create usage log
            log = UsageLog(
                user_id=user_id,
                chatbot_id=chatbot_id,
                action_type=action_type,
                action_details=details,
                resource_usage=resource_amount
            )
            
            db.session.add(log)
            
            # Update subscription usage counters if applicable
            subscription = Subscription.query.filter_by(user_id=user_id).first()
            if subscription:
                if action_type == 'query':
                    subscription.record_query()
                elif action_type == 'storage':
                    subscription.storage_used_mb += resource_amount
                    db.session.commit()
            
            db.session.commit()
            return True, log
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error recording usage: {str(e)}")
            db.session.rollback()
            return False, str(e)
    
    @staticmethod
    def get_user_subscription_details(user_id):
        """Get detailed subscription information for a user"""
        try:
            user = User.query.get(user_id)
            
            if not user:
                return None
            
            subscription = Subscription.query.filter_by(user_id=user_id).first()
            
            if not subscription:
                # Return free plan details if available
                free_plan = Plan.query.filter_by(name='Free').first()
                if free_plan:
                    return {
                        'plan': {
                            'id': free_plan.id,
                            'name': free_plan.name,
                            'description': free_plan.description,
                            'price': free_plan.price,
                            'billing_cycle': free_plan.billing_cycle,
                            'features': {
                                'max_chatbots': free_plan.max_chatbots,
                                'max_storage_mb': free_plan.max_storage_mb,
                                'max_queries_per_month': free_plan.max_queries_per_month,
                                'max_pages_per_chatbot': free_plan.max_pages_per_chatbot,
                                'max_files_per_chatbot': free_plan.max_files_per_chatbot,
                                'custom_branding': free_plan.custom_branding,
                                'priority_support': free_plan.priority_support,
                                'analytics_dashboard': free_plan.analytics_dashboard,
                                'api_access': free_plan.api_access
                            }
                        },
                        'subscription': None,
                        'usage': {
                            'chatbots': Chatbot.query.filter_by(user_id=user_id).count(),
                            'queries_this_month': UsageLog.query.filter_by(
                                user_id=user_id, 
                                action_type='query',
                                timestamp=datetime.utcnow() - timedelta(days=30)
                            ).count(),
                            'storage_used_mb': 0
                        }
                    }
                return None
            
            plan = Plan.query.get(subscription.plan_id)
            
            # Get recent payments
            recent_payments = PaymentHistory.query.filter_by(
                subscription_id=subscription.id
            ).order_by(PaymentHistory.payment_date.desc()).limit(5).all()
            
            # Calculate usage statistics
            chatbot_count = Chatbot.query.filter_by(user_id=user_id).count()
            
            return {
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
                },
                'subscription': {
                    'id': subscription.id,
                    'status': subscription.status,
                    'start_date': subscription.start_date,
                    'end_date': subscription.end_date,
                    'trial_end_date': subscription.trial_end_date,
                    'is_active': subscription.is_active(),
                    'is_in_trial': subscription.is_in_trial(),
                    'days_remaining': subscription.days_remaining(),
                    'last_payment_date': subscription.last_payment_date,
                    'next_payment_date': subscription.next_payment_date,
                    'payment_method': subscription.payment_method,
                    'recent_payments': [{
                        'id': payment.id,
                        'amount': payment.amount,
                        'currency': payment.currency,
                        'date': payment.payment_date,
                        'status': payment.status
                    } for payment in recent_payments]
                },
                'usage': {
                    'chatbots': chatbot_count,
                    'chatbots_limit': plan.max_chatbots,
                    'chatbots_percentage': (chatbot_count / plan.max_chatbots * 100) if plan.max_chatbots > 0 else 0,
                    'queries_this_month': subscription.current_period_queries,
                    'queries_limit': plan.max_queries_per_month,
                    'queries_percentage': (subscription.current_period_queries / plan.max_queries_per_month * 100) 
                                         if plan.max_queries_per_month > 0 else 0,
                    'storage_used_mb': subscription.storage_used_mb,
                    'storage_limit_mb': plan.max_storage_mb,
                    'storage_percentage': (subscription.storage_used_mb / plan.max_storage_mb * 100) 
                                         if plan.max_storage_mb > 0 else 0
                }
            }
            
        except SQLAlchemyError as e:
            current_app.logger.error(f"Error getting subscription details: {str(e)}")
            return None
