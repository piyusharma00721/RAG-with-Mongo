
# !pip3 install --quiet --upgrade pymongo langchain langchain-community langchain-huggingface gpt4all pypdf
# !pip install langchain-mongodb
# !pip install langchain huggingface-hub langchain-huggingface langchain-mongodb sentence-transformers
# !pip install gpt4all



import pypdf



MONGODB_URI = ("mongodb+srv://sharmapiyush1106:N28xansYkZEb93Et@cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")


#Run this only when the llm model is not locally present


# !huggingface-cli download TheBloke/Mistral-7B-OpenOrca-GGUF mistral-7b-openorca.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False



from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings

# Load the embedding model (https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

# Instantiate vector store
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
   connection_string = MONGODB_URI,
   namespace = "langchain_db.local_rag",
   embedding=embedding_model,
   index_name="vector_index"
)



from pymongo import MongoClient

# uri = "mongodb+srv://sharmapiyush1106:<ZJeTUUjTBHRfwwOP>@cluster0.ljxdkle.mongodb.net/"
uri = "mongodb+srv://sharmapiyush1106:N28xansYkZEb93Et@cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongodb+srv://sharmapiyush1106:ZJeTUUjTBHRfwwOP @cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0

client = MongoClient(uri)
try:
    dbs = client.list_database_names()
    print("✅ Connected! Databases:", dbs)
except Exception as e:
    print("❌ Failed to connect:", e)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time

# Define the folder containing PDFs
pdf_folder_path = "./Data_PDF"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith(".pdf")]

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Batch size (documents per insert)
BATCH_SIZE = 50

def batch_insert(docs, batch_size=BATCH_SIZE):
    """Insert documents in small batches to avoid memory spikes."""
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        try:
            vector_store.add_documents(batch)
            print(f"✅ Inserted batch {i // batch_size + 1} ({len(batch)} docs)")
        except Exception as e:
            print(f"❌ Error inserting batch {i // batch_size + 1}: {e}")

# Process PDFs one by one
for idx, pdf_file in enumerate(pdf_files, 1):
    print(f"\n📄 Processing {os.path.basename(pdf_file)} ({idx}/{len(pdf_files)})")
    try:
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        split_docs = text_splitter.split_documents(data)
        batch_insert(split_docs)
    except Exception as e:
        print(f"❌ Failed to process {pdf_file}: {e}")


from pymongo import MongoClient

# uri = "mongodb+srv://sharmapiyush1106:<ZJeTUUjTBHRfwwOP>@cluster0.ljxdkle.mongodb.net/"
uri = "mongodb+srv://sharmapiyush1106:N28xansYkZEb93Et@cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# mongodb+srv://sharmapiyush1106:ZJeTUUjTBHRfwwOP @cluster0.ljxdkle.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0

client = MongoClient(uri)
try:
    dbs = client.list_database_names()
    print("✅ Connected! Databases:", dbs)
except Exception as e:
    print("❌ Failed to connect:", e)


# Use helper method to create the vector search index
vector_store.create_vector_search_index(
   dimensions = 1024 # The dimensions of the vector embeddings to be indexed
)


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import GPT4All

# Configure the LLM
local_path = "/home/piyush/RAG-with-Mongo/mistral-7b-openorca.Q4_K_M.gguf"

# Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]

# Verbose is required to pass to the callback manager
llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)



from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import pprint

# Step 1: Define Prompt Template
custom_prompt = PromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end concisely and clearly:
{context}

Question: {question}
""")

# Step 2: Format context documents
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs[:5])  # Limit to top 5 docs for speed & clarity

# Step 3: Create RAG Chain (Retriever + Prompt + LLM + Parser)
retriever = vector_store.as_retriever()  # Define retriever from vector_store
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

# Step 4: Query RAG
def query_rag(question: str, top_k: int = 5):
    try:
        # Set top-k documents from vector store
        retriever.search_kwargs["k"] = top_k

        # Invoke chain
        answer = rag_chain.invoke(question)

        # Fetch source docs (optional)
        documents = retriever.invoke(question)

        print("\n🧠 Question:", question)
      #   print("\n✅ Answer:\n", answer)
        print("\n📄 Source Documents (top {}):".format(top_k))
        pprint.pprint(documents[:top_k])

        return answer
    except Exception as e:
        print(f"❌ Error during RAG query: {e}")
        return None





# Run example
query_rag("What punishment anyone can get when - Whoever intentionally gives false evidence in any stage of a judicial proceeding, or fabricates false evidence for the purpose of being used in any stage of a judicial proceeding")
