"""Gunicorn entry point for sharing initialized weights through fork."""
from server import app, initialize_detector

initialize_detector()
