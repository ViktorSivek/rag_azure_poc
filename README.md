# RAG Azure PoC with LangGraph Multiagent Workflow

A Retrieval-Augmented Generation (RAG) application built with Azure AI Search, OpenAI, and LangGraph for multiagent workflows. The application provides intelligent document search with page citations and automatic email notifications.

## Features

- **RAG System**: Retrieval-augmented generation using Azure AI Search and OpenAI
- **Page Citations**: Precise source citations with page numbers (e.g., "Document (p. 12, 15)")
- **LangGraph Multiagent**: Simple 2-node workflow (RAG → Email notification)
- **Email Integration**: Automatic Gmail notifications with questions and answers
- **Clean UI**: Modern web interface for asking questions and viewing responses

## Architecture

```
User Question → LangGraph Workflow:
                ├── RAG Node (retrieval + synthesis)
                └── Email Node (Gmail notification)
                → Response + Email Status
```

## Setup

### 1. Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Azure AI Search
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_ADMIN_KEY=your-search-admin-key
AZURE_SEARCH_INDEX_NAME=telco-rag-v2

# Gmail SMTP (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
EMAIL_TO=recipient@gmail.com
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: http://localhost:8000

## Usage

1. **Ask Questions**: Type questions about your indexed documents
2. **Get Answers**: Receive AI-generated responses with source citations
3. **Email Notifications**: Automatically receive Gmail with question + answer
4. **View Status**: See email delivery status in the UI (✓ success / ✗ failed)

## LangGraph Workflow

The application uses a simple 2-node LangGraph workflow:

- **RAG Node**: Performs document retrieval and answer synthesis
- **Email Node**: Sends Gmail notification with the Q&A

Email format:

- **Subject**: "RAG message"
- **Body**: Question + RAG answer + sources

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── email_service.py     # Gmail SMTP service
│   └── langgraph_workflow.py # LangGraph multiagent workflow
├── frontend/
│   └── index.html           # Web interface
├── scripts/
│   ├── create_index.py      # Azure Search index creation
│   └── ingest_data.py       # Document ingestion
└── requirements.txt         # Python dependencies
```

## API Endpoints

- `GET /healthz` - Health check
- `POST /ask` - Ask questions (returns answer + sources + email_status)

## Technologies

- **Backend**: FastAPI, LangGraph, Azure AI Search, OpenAI
- **Frontend**: HTML/CSS/JavaScript
- **Email**: SMTP (Gmail)
- **Deployment**: Docker support included
