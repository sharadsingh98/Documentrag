"""Configuration settings for RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .config_loader import load_api_key

# Load environment variables
load_dotenv()


def load_api_key():
    """
    Load API key from environment or Streamlit secrets
    
    Returns:
        str: OpenAI API key
    """
    # Try environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If not found, try Streamlit secrets (for deployment)
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("OPENAI_API_KEY")
        except (ImportError, FileNotFoundError, KeyError, AttributeError):
            pass
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found! "
            "Please set OPENAI_API_KEY in .env file or Streamlit secrets"
        )
    
    return api_key


class Config:
    """Configuration class for RAG system settings"""
    
    # LLM Settings
    LLM_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0
    
    # Embedding Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    
    # Vector Store
    VECTOR_STORE_K = 4  # Number of documents to retrieve
    
    # Default URLs for document ingestion
    DEFAULT_URLS = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
    ]
    
    @staticmethod
    def get_llm():
        """
        Get configured LLM instance
        
        Returns:
            ChatOpenAI: Configured LLM instance
        """
        api_key = load_api_key()
        return ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=api_key
        )
    
    @staticmethod
    def get_embeddings():
        """
        Get configured embeddings instance
        
        Returns:
            OpenAIEmbeddings: Configured embeddings instance
        """
        api_key = load_api_key()
        return OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=api_key
        )