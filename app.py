
# RAG Streamlit App with MongoDB, LangChain, Gemini, and HuggingFace

import streamlit as st
import os
import tempfile
import shutil
import time
from typing import List, Optional, Tuple
import logging
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import the RAGSystem logic
from rag_system import RAGSystem  # You should put your class in rag_system.py for modularity

# Set up Streamlit page config
st.set_page_config(page_title="RAG System", layout="wide")

# Load config from env or secrets
@st.cache_resource(show_spinner=False)
def load_config():
    config = {
        'mongodb_uri': os.getenv("MONGODB_URI", st.secrets.get("MONGODB_URI", "")),
        'gemini_api_key': os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", "")),
        'gpt4all_model_path': os.getenv("GPT4ALL_MODEL_PATH", st.secrets.get("GPT4ALL_MODEL_PATH", ""))
    }
    return config

# UI Entry
config = load_config()

if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar: Config
with st.sidebar:
    st.header("🛠️ RAG System Configuration")

    mongodb_uri = config['mongodb_uri'] or st.text_input("MongoDB URI", type="password")
    gemini_api_key = config['gemini_api_key'] or st.text_input("Gemini API Key", type="password")
    model_path = config['gpt4all_model_path'] or st.text_input("GPT4All Model Path")

    embedding_model = st.selectbox("Embedding Model", ["huggingface", "gemini"], index=0)
    llm_model = st.selectbox("LLM", ["gemini", "gpt4all"], index=0)

    force_recreate = st.checkbox("Force Recreate Index", value=False)

    # Init button
    if st.button("🚀 Initialize System"):
        rag = st.session_state.rag
        if not mongodb_uri:
            st.error("MongoDB URI required")
        elif llm_model == "gemini" and not gemini_api_key:
            st.error("Gemini API key required")
        else:
            with st.spinner("Initializing system..."):
                if not rag.initialize_database(mongodb_uri): st.stop()
                if not rag.setup_embeddings(embedding_model, gemini_api_key): st.stop()
                if not rag.setup_vector_store(mongodb_uri): st.stop()
                if not rag.setup_llm(llm_model, gemini_api_key, model_path): st.stop()
                if not rag.create_vector_index(force_recreate): st.stop()
                st.session_state.system_ready = True
                st.success("✅ System Initialized")

    if st.session_state.system_ready:
        st.success("🟢 System Ready")
        if st.button("🔁 Reset System"):
            st.session_state.system_ready = False
            st.session_state.rag = RAGSystem()
            st.session_state.processed_files = []
            st.rerun()
    else:
        st.error("🔴 System Not Initialized")

# Document Upload Tab
if st.session_state.system_ready:
    tab1, tab2, tab3 = st.tabs(["📄 Upload", "🤖 Ask", "📊 Status"])

    with tab1:
        uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded and st.button("🔄 Process Files"):
            temp_dir = tempfile.mkdtemp()
            try:
                paths = []
                for f in uploaded:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, 'wb') as out:
                        out.write(f.read())
                    paths.append(path)
                with st.spinner("Processing..."):
                    total = st.session_state.rag.process_pdfs(paths)
                    st.session_state.processed_files.extend([f.name for f in uploaded])
                    st.success(f"✅ {total} document chunks added")
            finally:
                shutil.rmtree(temp_dir)

    with tab2:
        st.subheader("Ask Questions to AI")
        question = st.text_area("Your question")
        k = st.slider("Top K Results", 1, 10, 5)
        if st.button("🔍 Get Answer") and question:
            with st.spinner("Generating answer..."):
                answer, docs = st.session_state.rag.query_rag(question, k)
                if answer:
                    st.markdown("### 💬 Answer")
                    st.write(answer)
                    st.markdown("### 📄 Sources")
                    for doc in docs:
                        st.markdown(f"**{doc.metadata.get('source_file')}**\n\n{doc.page_content}")
                else:
                    st.error("❌ No answer returned")

    with tab3:
        st.header("📊 System Info")
        st.write(f"**Processed Files:** {st.session_state.processed_files}")
        dims = st.session_state.rag.get_embedding_dimensions()
        st.write(f"**Embedding Dimensions:** {dims}")
        st.write(f"**Mongo URI:** {'✔️' if mongodb_uri else '❌'}")
        st.write(f"**LLM:** {llm_model}, **Embedding:** {embedding_model}")
else:
    st.info("🛠️ Configure the system from the sidebar first.")
