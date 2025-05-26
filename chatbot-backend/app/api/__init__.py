# app/api/__init__.py
from flask import Blueprint

bp = Blueprint('api', __name__)

# Import routes at the end to avoid circular dependencies
from app.api import routes