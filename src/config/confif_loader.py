# src/config/config_loader.py
import os
from dotenv import load_dotenv
import streamlit as st

def load_api_key():
    """Load API key from environment or Streamlit secrets"""
    # First try to load from .env file (local development)
    load_dotenv()
    
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found, try Streamlit secrets (deployment)
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (FileNotFoundError, KeyError):
            pass
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please add it to Streamlit secrets.")
        st.stop()
    
    return api_key