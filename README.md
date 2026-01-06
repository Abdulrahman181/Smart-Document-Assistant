# Chat with Your Documents

A Retrieval-Augmented Generation (RAG) application that allows users to chat with their own documents using Large Language Models (LLMs) and semantic search.

---

## Overview

This project implements a complete **RAG pipeline** that combines document retrieval with local language model inference.  
Users can ask questions, and the system retrieves relevant document chunks using a vector database, then generates answers **strictly based on the retrieved context**.

The application runs fully locally and does not rely on external APIs.

---

## Features

- Chat with custom documents
- Semantic search using vector embeddings
- FAISS vector store for fast retrieval
- Local LLM inference (LLaMA via llama-cpp / CTransformers)
- Context-aware answers (no hallucinations)
- Source document references
- Streamlit web interface
- Fully offline and privacy-friendly

---

## Architecture

Documents → Chunking → Embeddings → FAISS
                                      ↓
User Question → Retriever → LLM → Answer

---

## Tech Stack

- Python
- LangChain
- FAISS
- Sentence-Transformers
- LLaMA (GGUF format)
- Streamlit

---

## Project Structure

chat-with-your-documents/
├── app.py

├── ingest.py

├── requirements.txt

├── README.md

├── .gitignore

└── data/

---
# 💻 Authors
  - Abdul Rahman Ahmed 

  - abdulrahmannassar202@gmail.com

# 📌 Project link
  - https://github.com/Abdulrahman181/Smart-Document-Assistant

