"""
Email Service for Website AI Assistant Platform
Handles sending verification emails, password reset emails, and other notifications
"""
from flask import current_app, render_template_string
from flask_mail import Mail, Message
import os
import logging

# Initialize Mail instance
mail = Mail()

class EmailService:
    """Service for sending emails"""
    
    @staticmethod
    def init_app(app):
        """Initialize the email service with the Flask app"""
        mail.init_app(app)
    
    @staticmethod
    def send_email(subject, recipients, html_body, text_body=None, sender=None):
        """
        Send an email
        
        Args:
            subject: Email subject
            recipients: List of recipient email addresses
            html_body: HTML content of the email
            text_body: Plain text content of the email (optional)
            sender: Email sender (optional, uses default if not provided)
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            if not current_app.config.get('MAIL_SERVER'):
                current_app.logger.warning("Email service not configured. Email not sent.")
                return False
                
            msg = Message(
                subject=subject,
                recipients=recipients,
                html=html_body,
                body=text_body or html_body.replace('<br>', '\n'),
                sender=sender or current_app.config.get('MAIL_DEFAULT_SENDER')
            )
            
            mail.send(msg)
            current_app.logger.info(f"Email sent to {', '.join(recipients)}")
            return True
            
        except Exception as e:
            current_app.logger.error(f"Error sending email: {str(e)}")
            return False
    
    @staticmethod
    def send_verification_email(user, verification_url):
        """
        Send an email verification link
        
        Args:
            user: User model instance
            verification_url: URL for email verification
            
        Returns:
            Boolean indicating success or failure
        """
        subject = "Verify Your Email - Website AI Assistant"
        
        html_body = f"""
        <h1>Welcome to Website AI Assistant!</h1>
        <p>Hi {user.first_name or user.email},</p>
        <p>Thank you for signing up. Please verify your email address by clicking the link below:</p>
        <p><a href="{verification_url}">Verify Email</a></p>
        <p>This link will expire in 24 hours.</p>
        <p>If you did not sign up for an account, please ignore this email.</p>
        <br>
        <p>Best regards,</p>
        <p>The Website AI Assistant Team</p>
        """
        
        return EmailService.send_email(
            subject=subject,
            recipients=[user.email],
            html_body=html_body
        )
    
    @staticmethod
    def send_password_reset_email(user, reset_url):
        """
        Send a password reset link
        
        Args:
            user: User model instance
            reset_url: URL for password reset
            
        Returns:
            Boolean indicating success or failure
        """
        subject = "Password Reset - Website AI Assistant"
        
        html_body = f"""
        <h1>Password Reset</h1>
        <p>Hi {user.first_name or user.email},</p>
        <p>You requested a password reset. Please click the link below to reset your password:</p>
        <p><a href="{reset_url}">Reset Password</a></p>
        <p>This link will expire in 1 hour.</p>
        <p>If you did not request a password reset, please ignore this email.</p>
        <br>
        <p>Best regards,</p>
        <p>The Website AI Assistant Team</p>
        """
        
        return EmailService.send_email(
            subject=subject,
            recipients=[user.email],
            html_body=html_body
        )
    
    @staticmethod
    def send_welcome_email(user):
        """
        Send a welcome email after successful verification
        
        Args:
            user: User model instance
            
        Returns:
            Boolean indicating success or failure
        """
        subject = "Welcome to Website AI Assistant!"
        
        html_body = f"""
        <h1>Welcome to Website AI Assistant!</h1>
        <p>Hi {user.first_name or user.email},</p>
        <p>Thank you for verifying your email. Your account is now active!</p>
        <p>You can now create chatbots for your website and customize them to your needs.</p>
        <p>If you have any questions, please don't hesitate to contact our support team.</p>
        <br>
        <p>Best regards,</p>
        <p>The Website AI Assistant Team</p>
        """
        
        return EmailService.send_email(
            subject=subject,
            recipients=[user.email],
            html_body=html_body
        )
    
    @staticmethod
    def send_subscription_confirmation(user, subscription, plan):
        """
        Send a subscription confirmation email
        
        Args:
            user: User model instance
            subscription: Subscription model instance
            plan: Plan model instance
            
        Returns:
            Boolean indicating success or failure
        """
        subject = f"Subscription Confirmation - {plan.name} Plan"
        
        html_body = f"""
        <h1>Subscription Confirmation</h1>
        <p>Hi {user.first_name or user.email},</p>
        <p>Thank you for subscribing to the <strong>{plan.name}</strong> plan!</p>
        <p>Your subscription is now active and will renew on {subscription.next_payment_date.strftime('%B %d, %Y') if subscription.next_payment_date else 'N/A'}.</p>
        <h2>Plan Details:</h2>
        <ul>
            <li>Plan: {plan.name}</li>
            <li>Price: ${plan.price}/{plan.billing_cycle}</li>
            <li>Max Chatbots: {plan.max_chatbots}</li>
            <li>Storage: {plan.max_storage_mb} MB</li>
            <li>Queries per Month: {plan.max_queries_per_month}</li>
        </ul>
        <p>You can manage your subscription in your account settings.</p>
        <br>
        <p>Best regards,</p>
        <p>The Website AI Assistant Team</p>
        """
        
        return EmailService.send_email(
            subject=subject,
            recipients=[user.email],
            html_body=html_body
        )
