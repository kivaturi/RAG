import streamlit as st
import os
import tempfile
import time

# Core LangChain & RAG Libraries
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Document Assistant")

# Sidebar for Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key Input
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # 2. File Uploader
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True
    )

# ==========================================
# 2. CACHED RESOURCE FUNCTIONS
# ==========================================
# We use @st.cache_resource so these heavy operations run only once
# or when the input (uploaded_files) changes.

@st.cache_resource
def load_and_process_documents(files):
    """
    Saves uploaded files to temp dir, loads them, chunks them,
    and builds the FAISS vector store.
    """
    if not files:
        return None

    with st.spinner("Processing documents... This might take a minute."):
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = []
            
            # Save uploaded files to temp disk so PyPDFLoader can read them
            for file in files:
                temp_filepath = os.path.join(temp_dir, file.name)
                with open(temp_filepath, "wb") as f:
                    f.write(file.getbuffer())
                
                # Load the PDF
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())
            
            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            
            # Embeddings
            # Using cpu-friendly HuggingFace embeddings to save API costs
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Vector Store
            vector_store = FAISS.from_documents(splits, embeddings)
            
            return vector_store

# ==========================================
# 3. MAIN APP LOGIC
# ==========================================

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main Interaction Loop
if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue.")

elif not uploaded_files:
    st.info("Please upload PDF documents in the sidebar to begin.")

else:
    # 1. Build/Load the Vector Store
    vector_store = load_and_process_documents(uploaded_files)
    
    if vector_store:
        # 2. Setup the Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 3. Setup the LLM Chain
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # 4. Handle User Query
        if prompt := st.chat_input("Ask a question about your documents..."):
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                    source_docs = response["source_documents"]

                    # Display Answer
                    st.markdown(answer)
                    
                    # Display Sources (Expandable)
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(source_docs):
                            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            page_num = doc.metadata.get('page', 'Unknown')
                            st.markdown(f"**Source {i+1}:** {source_name} (Page {page_num})")
                            st.text(doc.page_content[:300] + "...") # Show preview

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
