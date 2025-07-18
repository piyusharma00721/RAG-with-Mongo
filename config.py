# config.py (for secure configuration management)
import os
import streamlit as st
from typing import Dict, Optional

class Config:
    """Configuration manager for secure handling of secrets"""
    
    def __init__(self):
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from various sources"""
        # Priority: Streamlit secrets > Environment variables > Default
        
        # MongoDB Configuration
        self.config['mongodb_uri'] = self._get_secret('MONGODB_URI')
        
        # API Keys
        self.config['gemini_api_key'] = self._get_secret('GEMINI_API_KEY')
        
        # Model paths
        self.config['gpt4all_model_path'] = self._get_secret('GPT4ALL_MODEL_PATH')
        
        # App settings
        self.config['app_title'] = self._get_secret('APP_TITLE', 'RAG System')
        self.config['debug'] = self._get_secret('DEBUG', 'False').lower() == 'true'
    
    def _get_secret(self, key: str, default: str = None) -> Optional[str]:
        """Get secret from Streamlit secrets, env vars, or default"""
        try:
            # Try Streamlit secrets first
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except:
            pass
        
        # Try environment variables
        value = os.getenv(key, default)
        return value if value != "" else default
    
    def get(self, key: str, default: str = None) -> Optional[str]:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def is_configured(self) -> bool:
        """Check if minimum configuration is available"""
        return bool(self.config.get('mongodb_uri'))

# secrets.toml (for Streamlit Cloud deployment)
"""
# Create this file in .streamlit/secrets.toml

[default]
MONGODB_URI = "mongodb+srv://sharmapiyush1106:N28xansYkZEb93Et@cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
GEMINI_API_KEY = "your_gemini_api_key_here"
GPT4ALL_MODEL_PATH = ""
APP_TITLE = "RAG System"
DEBUG = "False"
"""

# streamlit_app.py (modified main app with secure configuration)
import streamlit as st
import os
import tempfile
import shutil
from typing import List, Optional
import logging
from config import Config

# Initialize configuration
config = Config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your RAG system class here
# from rag_system import RAGSystem

def main():
    st.set_page_config(
        page_title=config.get('app_title', 'RAG System'),
        page_icon="🤖",
        layout="wide"
    )
    
    st.title(f"🤖 {config.get('app_title')}")
    st.write("Upload PDFs and ask questions using advanced AI models")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Check if MongoDB is pre-configured
        mongodb_uri = config.get('mongodb_uri')
        if mongodb_uri:
            st.success("✅ MongoDB connection configured")
        else:
            mongodb_uri = st.text_input(
                "MongoDB URI",
                type="password",
                help="Enter your MongoDB Atlas connection string"
            )
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["huggingface", "gemini"],
            help="Choose between HuggingFace (free) or Gemini (requires API key)"
        )
        
        # API Key for Gemini
        api_key = config.get('gemini_api_key')
        if model_type == "gemini":
            if api_key:
                st.success("✅ Gemini API key configured")
            else:
                api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    help="Enter your Google Gemini API key"
                )
        
        # GPT4All model path
        model_path = config.get('gpt4all_model_path')
        if model_type == "huggingface":
            if model_path and os.path.exists(model_path):
                st.success("✅ GPT4All model configured")
            else:
                st.warning("⚠️ GPT4All model not found. Please upload or specify path.")
                model_path = st.text_input(
                    "GPT4All Model Path",
                    help="Path to your local GPT4All model file"
                )
        
        # Initialize system
        if st.button("🚀 Initialize System"):
            if not mongodb_uri:
                st.error("Please provide MongoDB URI")
                return
            
            if model_type == "gemini" and not api_key:
                st.error("Please provide Gemini API key")
                return
            
            if model_type == "huggingface" and not model_path:
                st.error("Please provide GPT4All model path")
                return
            
            with st.spinner("Initializing system..."):
                # Initialize your RAG system here
                # ... (rest of initialization code)
                pass
    
    # Rest of your app code...

if __name__ == "__main__":
    main()