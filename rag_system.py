
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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.embedding_model = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        self.BATCH_SIZE = 50
        self.client = None
        self.db = None
        self.collection = None
        
    def initialize_database(self, mongodb_uri: str) -> bool:
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(mongodb_uri)
            # Test connection
            self.client.admin.command('ping')
            
            # Get database and collection references
            self.db = self.client.get_database("langchain_db")
            self.collection = self.db.get_collection("local_rag")
            
            logger.info("✅ Connected to MongoDB")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            st.error(f"Database connection failed: {e}")
            return False
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensions of the current embedding model"""
        if isinstance(self.embedding_model, GoogleGenerativeAIEmbeddings):
            return 768
        elif isinstance(self.embedding_model, HuggingFaceEmbeddings):
            # Different HuggingFace models have different dimensions
            model_name = self.embedding_model.model_name
            if "all-MiniLM-L6-v2" in model_name:
                return 384
            elif "mxbai-embed-large-v1" in model_name:
                return 1024
            elif "all-mpnet-base-v2" in model_name:
                return 768
            else:
                return 384  # Default fallback
        else:
            return 384  # Default fallback
    
    def drop_vector_index(self) -> bool:
        """Drop existing vector index"""
        try:
            if self.collection is not None:
                # List all indexes
                indexes = list(self.collection.list_indexes())
                vector_index_exists = any(idx.get('name') == 'vector_index' for idx in indexes)
                
                if vector_index_exists:
                    self.collection.drop_index('vector_index')
                    logger.info("✅ Dropped existing vector index")
                    return True
                else:
                    logger.info("ℹ️ No vector index to drop")
                    return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to drop vector index: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            if self.collection is not None:
                result = self.collection.delete_many({})
                logger.info(f"✅ Cleared {result.deleted_count} documents from collection")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False
    
    def setup_embeddings(self, model_type: str = "huggingface", api_key: str = "") -> bool:
        """Setup embedding model"""
        try:
            if model_type == "gemini" and api_key:
                try:
                    # Try initializing Gemini embeddings directly
                    self.embedding_model = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key = api_key
                    )
                    logger.info("✅ Gemini embeddings initialized (768 dimensions)")
                    return True
                except Exception as e:
                    logger.error(f"❌ Gemini Embeddings initialization failed: {e}")
                    st.error("⚠️ Gemini API key invalid or failed to initialize embeddings. Falling back to HuggingFace.")
            
            # Fallback to HuggingFace (384d)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✅ HuggingFace embeddings initialized (384 dimensions)")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup embeddings: {e}")
            st.error(f"Embedding setup failed: {e}")
            return False

                    
    
    def setup_vector_store(self, mongodb_uri: str, namespace: str = "langchain_db.local_rag") -> bool:
        """Setup vector store"""
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
    
    def setup_llm(self, model_type: str = "gemini", api_key: str = "", model_path: str = "") -> bool:
        """Setup LLM"""
        try:
            if model_type == "gemini" and api_key:
                try:
                    self.llm = GoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key,
                        temperature=0.3
                    )
                    logger.info("✅ Gemini LLM initialized")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to initialize Gemini LLM: {e}")
                    st.error("Gemini API key may be invalid or failed to initialize LLM.")
                    return False

            elif model_type == "gpt4all" and model_path and os.path.exists(model_path):
                self.llm = GPT4All(
                    model=model_path, 
                    verbose=False,
                    streaming=True
                )
                logger.info("✅ GPT4All LLM initialized")
                return True
            else:
                raise ValueError("Invalid model type or missing requirements")

        except Exception as e:
            logger.error(f"❌ Failed to setup LLM: {e}")
            st.error(f"LLM setup failed: {e}")
            return False

    
    def create_vector_index(self, force_recreate: bool = False) -> bool:
        """Create vector search index"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            dimensions = self.get_embedding_dimensions()
            
            if force_recreate:
                logger.info("🔄 Force recreating vector index...")
                self.drop_vector_index()
                self.clear_collection()
                time.sleep(2)  # Give MongoDB time to process
            
            try:
                self.vector_store.create_vector_search_index(dimensions=dimensions)
                logger.info(f"✅ Vector index created with {dimensions} dimensions")
            except Exception as e:
                if "IndexAlreadyExists" in str(e):
                    logger.info(f"ℹ️ Vector index already exists")
                    # Check if dimensions match
                    if "vector field is indexed with" in str(e):
                        logger.warning("⚠️ Dimension mismatch detected, recreating index...")
                        self.drop_vector_index()
                        self.clear_collection()
                        time.sleep(2)
                        self.vector_store.create_vector_search_index(dimensions=dimensions)
                        logger.info(f"✅ Vector index recreated with {dimensions} dimensions")
                else:
                    raise e
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create vector index: {e}")
            st.error(f"Vector index creation failed: {e}")
            return False
    
    def process_pdfs(self, pdf_files: List[str], progress_callback=None) -> int:
        """Process PDF files and add to vector store"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        total_docs = 0
        
        for idx, pdf_file in enumerate(pdf_files):
            try:
                if progress_callback:
                    progress_callback(f"Processing {os.path.basename(pdf_file)} ({idx+1}/{len(pdf_files)})")
                
                # Load and split PDF
                loader = PyPDFLoader(pdf_file)
                data = loader.load()
                split_docs = self.text_splitter.split_documents(data)
                
                # Add metadata
                for doc in split_docs:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)
                    doc.metadata['processed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Batch insert to avoid memory issues
                for i in range(0, len(split_docs), self.BATCH_SIZE):
                    batch = split_docs[i:i + self.BATCH_SIZE]
                    self.vector_store.add_documents(batch)
                    total_docs += len(batch)
                
                logger.info(f"✅ Processed {os.path.basename(pdf_file)} - {len(split_docs)} chunks")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_file}: {e}")
                if progress_callback:
                    progress_callback(f"Error processing {os.path.basename(pdf_file)}: {e}")
                continue
        
        return total_docs
    
    def query_rag(self, question: str, top_k: int = 5) -> Tuple[Optional[str], List[Document]]:
        """Query the RAG system"""
        if not self.vector_store or not self.llm:
            raise ValueError("System not fully initialized")
        
        try:
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            # Create prompt template
            custom_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Instructions:
- Answer based on the provided context
- If the information is not in the context, say "I don't have enough information to answer this question"
- Be specific and cite relevant information from the context
- Keep your answer clear and well-structured

Answer:
""")
            
            def format_docs(docs: List[Document]) -> str:
                """Format documents for context"""
                if not docs:
                    return "No relevant documents found."
                
                formatted = []
                for i, doc in enumerate(docs[:top_k]):
                    source = doc.metadata.get('source_file', 'Unknown')
                    content = doc.page_content.strip()
                    formatted.append(f"Document {i+1} (Source: {source}):\n{content}")
                
                return "\n\n".join(formatted)
            
            # Create RAG chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Execute query
            start_time = time.time()
            answer = rag_chain.invoke(question)
            response_time = time.time() - start_time
            
            # Get source documents
            source_docs = retriever.invoke(question)
            
            logger.info(f"✅ Query processed in {response_time:.2f}s")
            
            return answer, source_docs[:top_k]
            
        except Exception as e:
            # Check if it's a dimension mismatch error
            if "vector field is indexed with" in str(e) and "dimensions but queried with" in str(e):
                logger.error("❌ Dimension mismatch detected - please recreate the vector index")
                st.error("❌ Dimension mismatch detected. Please reset the system and reinitialize with 'Force Recreate Index' option.")
                return None, []
            else:
                logger.error(f"❌ Error during RAG query: {e}")
                st.error(f"Query failed: {e}")
                return None, []

