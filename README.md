# AI-Powered Document Q&A System (RAG Chatbot)

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF, DOCX, and TXT documents and ask questions about their content.

The system performs semantic search using vector embeddings and generates context-aware answers using a local Llama 3.2 model through Ollama.

## Features

* PDF, DOCX, and TXT document support
* Document chunking
* Semantic search
* ChromaDB vector database
* HuggingFace embeddings
* Local LLM inference using Ollama
* Streamlit web interface
* Context-aware question answering

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Ollama
* Llama 3.2

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

Document Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Retrieval
↓
Llama 3.2
↓
Answer Generation
