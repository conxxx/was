"""
Script to clear user data from the database
"""
import os
import sys
from flask import Flask
from app import create_app, db
from app.models import User, Chatbot, Subscription, Plan

def clear_database():
    """Clear all user data from the database"""
    try:
        # Create app context
        app = create_app()
        with app.app_context():
            # Delete all user-related data
           # ChatbotData.query.delete()
            Chatbot.query.delete()
            Subscription.query.delete()
            User.query.delete()
            
            # Commit changes
            db.session.commit()
            print("Database cleared successfully!")
            
            # Recreate default plans if needed
            create_default_plans(app)
            
    except Exception as e:
        print(f"Error clearing database: {str(e)}")
        return False
    
    return True

def create_default_plans(app):
    """Create default subscription plans"""
    with app.app_context():
        # Check if plans already exist
        if Plan.query.count() == 0:
            # Create free plan
            free_plan = Plan(
                name="Free",
                description="Basic plan with limited features",
                price=0,
                chatbot_limit=1,
                message_limit=100,
                file_upload_limit=5,
                max_file_size=5,  # MB
                features=["Basic RAG", "Website crawling (up to 50 pages)", "File upload (PDF, DOCX, TXT)"]
            )
            
            # Create standard plan
            standard_plan = Plan(
                name="Standard",
                description="Standard plan with more features",
                price=29,
                chatbot_limit=3,
                message_limit=1000,
                file_upload_limit=20,
                max_file_size=20,  # MB
                features=["Advanced RAG", "Website crawling (up to 200 pages)", "File upload (PDF, DOCX, TXT, CSV)", "Custom chatbot appearance"]
            )
            
            # Create premium plan
            premium_plan = Plan(
                name="Premium",
                description="Premium plan with all features",
                price=99,
                chatbot_limit=10,
                message_limit=10000,
                file_upload_limit=100,
                max_file_size=100,  # MB
                features=["Advanced RAG", "Website crawling (unlimited pages)", "File upload (all formats)", "Custom chatbot appearance", "Priority support", "Analytics dashboard"]
            )
            
            # Add plans to database
            db.session.add(free_plan)
            db.session.add(standard_plan)
            db.session.add(premium_plan)
            db.session.commit()
            
            print("Default plans created successfully!")

if __name__ == "__main__":
    clear_database()
