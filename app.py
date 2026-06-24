import streamlit as st
import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain_ollama import ChatOllama


st.set_page_config(
    page_title="Local RAG Chatbot",
    layout="wide"
)

st.title("📄 Local RAG Chatbot (Llama 3.2)")


def load_document(uploaded_file):
    extension = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        file_path = temp_file.name

    if extension == "pdf":
        loader = PyPDFLoader(file_path)
    elif extension == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


@st.cache_resource
def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0
    )


def create_vector_store(chunks):
    embeddings = get_embedding_model()
    
    # Using an ephemeral or memory-based setup prevents accumulating duplicates on local disk across sessions
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return db


uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file:
    # Check if this file has already been processed into the session state
    if "db" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        
        with st.spinner("Processing document and creating vector database..."):
            documents = load_document(uploaded_file)
            chunks = chunk_documents(documents)
            
            # Save stats and DB to session state so they persist across reruns
            st.session_state["pages_loaded"] = len(documents)
            st.session_state["total_chunks"] = len(chunks)
            st.session_state["db"] = create_vector_store(chunks)
            st.session_state["file_name"] = uploaded_file.name
            
        st.success("Vector Database Ready!")

    # Display stats from session state
    st.write(f"Pages Loaded: {st.session_state['pages_loaded']}")
    st.write(f"Total Chunks: {st.session_state['total_chunks']}")

    # Question Input
    question = st.text_input(
        "Ask a question about the document"
    )

    if question:
        with st.spinner("Searching document..."):
            # Pull the database instance from session state
            db = st.session_state["db"]
            
            results = db.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                doc.page_content
                for doc in results
            )

            prompt = f"""
You are a helpful and precise document assistant.

Instructions:
- Answer the question thoroughly based on the provided context.
- If the context contains formatting anomalies (like spaces between letters in names or words), ignore the spacing and interpret the words normally.
- If the answer absolutely cannot be answered using the text below, say "Information not found in document."

Context:
{context}

Question:
{question}
Answer:
"""

            llm = get_llm()
            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("Retrieved Context"):
                st.write(context)