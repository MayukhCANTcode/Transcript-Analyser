# YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to chat with YouTube videos using AI.

The application extracts the transcript from a YouTube video, converts the content into embeddings, stores them in a vector database, and uses Gemini to answer questions based on the video content.

---

## Features

- Ask questions about any YouTube video
- AI-powered question answering using Gemini
- Video summarization
- Study notes generation
- Quiz generation
- Flashcard generation
- Semantic search using vector embeddings
- Streamlit web interface

---

## Architecture

YouTube URL
↓
Transcript Extraction
↓
Text Chunking
↓
Embeddings (all-MiniLM-L6-v2)
↓
Chroma Vector Database
↓
Retriever
↓
Gemini 2.5 Flash
↓
Answer Generation

---

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers
- YouTube Transcript API

---

## Project Structure

youtube-chatbot/

├── app.py

├── streamlit_app.py

├── requirements.txt

├── .env

└── README.md

---

## Installation

Clone the repository:

git clone <repository-url>

cd youtube-chatbot

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GOOGLE_API_KEY=YOUR_API_KEY

Run the application:

streamlit run streamlit_app.py

---

## How It Works

1. User enters a YouTube URL.
2. Transcript is extracted using YouTube Transcript API.
3. Transcript is split into chunks.
4. Embeddings are generated using all-MiniLM-L6-v2.
5. Embeddings are stored in ChromaDB.
6. Relevant chunks are retrieved using semantic search.
7. Gemini generates answers using retrieved context.

---

## Future Improvements

- Conversational memory
- Multi-video support
- PDF export for notes
- Agent-based workflow
- LangGraph integration
- YouTube playlist support

---

## Author

Mayukh Das

NIT Silchar

Electronics & Instrumentation Engineering
