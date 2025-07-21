import streamlit as st
import os
import tempfile
import shutil
import time
from typing import List, Optional, Tuple
import logging
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.embedding_model = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.BATCH_SIZE = 50
        self.client = None
        self.db = None
        self.collection = None

    def initialize_database(self, mongodb_uri: str) -> bool:
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client["langchain_db"]
            self.collection = self.db["local_rag"]
            logger.info("✅ Connected to MongoDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            st.error(f"Database connection failed: {e}")
            return False

    def get_embedding_dimensions(self) -> int:
        """Return fixed dimensions for Gemini embeddings."""
        return 768

    def check_vector_index(self) -> bool:
        """Check if vector_index exists using list_indexes."""
        try:
            if self.collection is None:
                logger.error("❌ Collection not initialized")
                return False
            indexes = self.collection.list_indexes()
            vector_index_exists = any(idx.get("name") == "vector_index" for idx in indexes)
            if vector_index_exists:
                logger.info("ℹ️ Found existing vector_index")
                return True
            logger.info("ℹ️ No vector_index found")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to check indexes: {e}")
            return False

    def setup_embeddings(self, api_key: str) -> bool:
        """Setup Gemini embedding model."""
        try:
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            logger.info("✅ Gemini embeddings initialized (768 dimensions)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup embeddings: {e}")
            st.error(f"Embedding setup failed: {e}")
            return False

    def setup_vector_store(self, mongodb_uri: str, namespace: str = "langchain_db.local_rag") -> bool:
        """Setup vector store."""
        try:
            if not self.embedding_model:
                raise ValueError("Embedding model not initialized")
            self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=mongodb_uri,
                namespace=namespace,
                embedding=self.embedding_model,
                index_name="vector_index"
            )
            logger.info("✅ Vector store initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup vector store: {e}")
            st.error(f"Vector store setup failed: {e}")
            return False

    def setup_llm(self, api_key: str) -> bool:
        """Setup Gemini LLM."""
        try:
            self.llm = GoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3
            )
            logger.info("✅ Gemini LLM initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup LLM: {e}")
            st.error(f"LLM setup failed: {e}")
            return False

    def create_vector_index(self) -> bool:
        """Create vector search index with proper dimensions if it doesn't exist."""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            if self.check_vector_index():
                logger.info("✅ Vector index already exists, skipping creation")
                return True
            dimensions = self.get_embedding_dimensions()
            logger.info(f"Attempting to create vector index with {dimensions} dimensions")
            self.vector_store.create_vector_search_index(dimensions=dimensions)
            logger.info(f"✅ Vector index creation command sent for {dimensions} dimensions")
            # Retry verification to account for Atlas delays
            for attempt in range(5):
                time.sleep(1)  # Wait for index to propagate
                if self.check_vector_index():
                    logger.info(f"✅ Vector index verified on attempt {attempt + 1}")
                    return True
                logger.warning(f"ℹ️ Vector index not found on attempt {attempt + 1}, retrying...")
            # Fallback: Assume creation succeeded if verification fails
            logger.warning("⚠️ Index verification failed after retries, assuming creation succeeded")
            st.warning("Index creation attempted but could not be verified. Try processing documents and querying. If errors persist, check MongoDB permissions or version.")
            return True  # Allow initialization to proceed
        except OperationFailure as e:
            if "IndexAlreadyExists" in str(e) or "already exists" in str(e).lower():
                logger.info("✅ Vector index already exists, skipping creation")
                return True
            logger.error(f"❌ Failed to create vector index: {e}")
            st.error(f"Vector index creation failed: {e}. Check MongoDB permissions or version (must be 6.0+).")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to create vector index: {e}")
            st.error(f"Vector index creation failed: {e}. Check MongoDB URI, connectivity, or cluster configuration.")
            return False

    def clear_collection(self) -> bool:
        """Manually clear the local_rag collection."""
        try:
            if self.collection:
                self.collection.drop()
                logger.info("✅ Successfully dropped local_rag collection")
                return True
            else:
                logger.error("❌ Collection not initialized")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to drop collection: {e}")
            st.error(f"Failed to clear collection: {e}")
            return False

    def process_pdfs(self, pdf_files: List[str], progress_callback=None) -> int:
        """Process PDF files and add to vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        total_docs = 0
        for idx, pdf_file in enumerate(pdf_files):
            try:
                if progress_callback:
                    progress_callback(f"Processing {os.path.basename(pdf_file)} ({idx+1}/{len(pdf_files)})")
                loader = PyPDFLoader(pdf_file)
                data = loader.load()
                split_docs = self.text_splitter.split_documents(data)
                for doc in split_docs:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)
                    doc.metadata['processed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                for i in range(0, len(split_docs), self.BATCH_SIZE):
                    batch = split_docs[i:i + self.BATCH_SIZE]
                    self.vector_store.add_documents(batch)
                    total_docs += len(batch)
                logger.info(f"✅ Processed {os.path.basename(pdf_file)} - {len(split_docs)} chunks")
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file}: {e}")
                if progress_callback:
                    progress_callback(f"Error processing {os.path.basename(pdf_file)}: {e}")
        return total_docs

    def query_rag(self, question: str, top_k: int = 5) -> Tuple[Optional[str], List[Document]]:
        """Query the RAG system."""
        if not self.vector_store or not self.llm:
            raise ValueError("System not fully initialized")
        try:
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            custom_prompt = PromptTemplate.from_template("""
You are an AI assistant designed to answer ONLY based on the provided document context or respond politely to greetings.

Instructions:
- If the user message is a greeting (like "hello", "hi", "good morning", etc.), respond with:
  "👋 Hello! I'm here to help you with questions related to your uploaded PDF documents. Please upload a PDF and ask your question."
- If the user asks a question that is NOT a greeting:
  - Use ONLY the following context to answer.
  - If the answer is not found in the context, reply with: "I don't have enough information to answer this question."
  - DO NOT make up answers or respond based on general knowledge.

Context:
{context}

User Input:
{question}

Your Response:
""")
            def format_docs(docs: List[Document]) -> str:
                if not docs:
                    return "No relevant documents found."
                formatted = []
                for i, doc in enumerate(docs[:top_k]):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    formatted.append(f"Document {i+1} (Source: {source}):\n{content}")
                return "\n\n".join(formatted)
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_prompt
                | self.llm
                | StrOutputParser()
            )
            start_time = time.time()
            answer = rag_chain.invoke(question)
            response_time = time.time() - start_time
            source_docs = retriever.invoke(question)
            logger.info(f"✅ Query processed in {response_time:.2f}s")
            return answer, source_docs[:top_k]
        except Exception as e:
            if "vector field is indexed with" in str(e).lower() and "dimensions but queried with" in str(e).lower():
                error_msg = (
                    f"❌ Dimension mismatch detected: {str(e)}. "
                    "Use the 'Clear Collection' button in the sidebar to reset the database, "
                    "or in MongoDB, drop the 'local_rag' collection. "
                    "Run: `db.local_rag.drop()` in mongosh."
                )
                logger.error(error_msg)
                st.error(error_msg)
                return None, []
            logger.error(f"❌ Error during RAG query: {e}")
            st.error(f"Query failed: {e}")
            return None, []

def load_config():
    """Load configuration from Streamlit secrets or environment variables."""
    config = {}
    try:
        config['mongodb_uri'] = st.secrets.get('MONGODB_URI', os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'))
        config['gemini_api_key'] = st.secrets.get('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY', ''))
    except:
        pass
    return config

def process_documents(uploaded_files):
    """Process uploaded documents."""
    if not uploaded_files:
        return
    temp_dir = tempfile.mkdtemp()
    pdf_paths = []
    try:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            pdf_paths.append(temp_path)
        progress_bar = st.progress(0)
        status_text = st.empty()
        def update_progress(message):
            status_text.text(message)
        with st.spinner("Processing documents..."):
            total_docs = st.session_state.rag_system.process_pdfs(pdf_paths, update_progress)
        for uploaded_file in uploaded_files:
            file_info = {
                'name': uploaded_file.name,
                'chunks': total_docs // len(uploaded_files) if uploaded_files else 0,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            st.session_state.processed_files.append(file_info)
        progress_bar.progress(100)
        st.success(f"✅ Successfully processed {len(uploaded_files)} files with {total_docs} document chunks!")
    except Exception as e:
        st.error(f"❌ Error processing documents: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    st.set_page_config(page_title="RAG Chat with Kanoon Bot ", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
    st.title("🤖 Multi PDF Document RAG Chat with Kanoon Bot")
    st.markdown("Chat with Kanoon Bot to understand your PDFs")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Set a fixed date and time for this session (05:38 PM IST, July 18, 2025)
    fixed_datetime = datetime(2025, 7, 18, 17, 38)  # 05:38 PM IST

    with st.sidebar:
        st.header("🔧 System Configuration")
        config = load_config()

        st.subheader("🚀 System Control")
        mongodb_uri = config.get('mongodb_uri', '')
        gemini_api_key = config.get('gemini_api_key', '')
        can_initialize = bool(mongodb_uri) and bool(gemini_api_key)
        if not can_initialize:
            st.warning("⚠️ Missing MongoDB URI or Gemini API Key in secrets or environment variables. Please configure them.")
        
        if st.button("🚀 Initialize System", disabled=not can_initialize):
            with st.spinner("Initializing system components..."):
                rag = st.session_state.rag_system
                progress_bar = st.progress(0)
                steps = [
                    (rag.initialize_database, [mongodb_uri], "Database"),
                    (rag.setup_embeddings, [gemini_api_key], "Embeddings"),
                    (rag.setup_vector_store, [mongodb_uri], "Vector Store"),
                    (rag.setup_llm, [gemini_api_key], "LLM"),
                    (rag.create_vector_index, [], "Vector Index")
                ]
                for i, (func, args, name) in enumerate(steps):
                    progress_bar.progress((i + 1) * 20)
                    if not func(*args):
                        st.error(f"❌ {name} initialization failed. Check MongoDB URI, Gemini API key, or database configuration.")
                        st.stop()
                st.session_state.system_ready = True
                st.success("✅ System initialized successfully!")
                st.rerun()

        if st.session_state.system_ready:
            st.success("🟢 System Ready")
            st.info("📏 Embeddings: 768 dimensions (Gemini)")
            if st.button("🔄 Reset System"):
                st.session_state.system_ready = False
                st.session_state.rag_system = RAGSystem()
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.rerun()
            if st.button("🗑️ Clear Collection", help="Drops the local_rag collection to resolve dimension mismatches. Reprocess documents after clearing."):
                with st.spinner("Clearing collection..."):
                    if st.session_state.rag_system.clear_collection():
                        st.session_state.system_ready = False
                        st.session_state.processed_files = []
                        st.session_state.chat_history = []
                        st.success("✅ Collection cleared. Please reinitialize the system and reprocess documents.")
                        st.rerun()

        st.subheader("📄 Document Upload & Processing")
        st.info("ℹ️ If you encounter a dimension mismatch error, use the 'Clear Collection' button above, or in MongoDB, drop the 'local_rag' collection. Run: `db.local_rag.drop()` in mongosh.")
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True, help="Upload PDFs to build your knowledge base")
        if uploaded_files and st.button("🔄 Process Documents", type="primary"):
            process_documents(uploaded_files)

        if st.session_state.processed_files:
            st.subheader("Processed Files")
            for file_info in st.session_state.processed_files:
                st.write(f"✅ {file_info['name']} ({file_info['chunks']} chunks)")

    if st.session_state.system_ready:
        # Apply soft background to chat UI
        st.markdown(
            """
            <style>
            .stChatContainer {
                background-color: #f5f5f5; /* Soft gray background */
                padding: 10px;
                border-radius: 5px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.subheader("💬 Chat with Kanoon Bot")
        # Display current date and time
        st.caption(f"Date and Time: {fixed_datetime.strftime('%I:%M %p IST on %B %d, %Y (%A)')}")

        # Chat container with auto-scroll to bottom
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(f"**👤 You:** {message['content']}")
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(f"**🤖 Kanoon:** {message['content']}")

        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(f"**👤 You:** {prompt}")

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = st.session_state.rag_system.query_rag(prompt, top_k=5)
                if answer:
                    st.markdown(f"**🤖 Kanoon:** {answer}")
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(sources):
                            st.write(f"**Source {i+1}: {doc.metadata.get('source_file', 'Unknown')}**")
                            st.write(doc.page_content)
                else:
                    st.markdown("I don't have enough information to answer this question.")
                    st.session_state.chat_history.append({"role": "assistant", "content": "I don't have enough information to answer this question."})

        # Add at the end to help simulate scroll-to-bottom
        # st.markdown("<div id='scroll-to-bottom'></div>", unsafe_allow_html=True)
        # st.components.v1.html("<script>document.getElementById('scroll-to-bottom').scrollIntoView();</script>", height=0)

    else:
        st.info("👈 Configure and initialize the system using the sidebar")
        st.markdown("""
        ### Getting Started:
        1. **Configure MongoDB**: Set MONGODB_URI in secrets or environment variables (e.g., mongodb://localhost:27017/)
        2. **If URI is not handy with you, try my own mongo URI
        3. **Initialize System**: Click the initialize button
        4. **Upload PDFs**: Use the document upload section in the sidebar
        5. **Ask Questions**: Start chatting in the main area
        ### Troubleshooting:
        - Ensure MongoDB is accessible and the URI is correct
        - If you see a dimension mismatch error, use the 'Clear Collection' button in the sidebar, or drop the 'local_rag' collection in MongoDB. Run: `db.local_rag.drop()` in mongosh
        - If index creation fails, ensure MongoDB version is 6.0+ and supports vector search. Check user permissions and network access.
        """)

if __name__ == "__main__":
    main()