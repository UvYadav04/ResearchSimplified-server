# ⚙️ Research Simplified — Server

Research Simplified (server) is a **high-performance FastAPI backend** that powers real-time research paper simplification and contextual AI interactions.

It handles **document ingestion, intelligent section processing, vector-based retrieval, and streaming LLM responses**, enabling a fast and interactive experience for understanding academic content.

---

## ✨ Key Features

### ⚡ Asynchronous Streaming Pipeline
- Fully **async architecture** using FastAPI
- Non-blocking request handling for:
  - Document processing
  - Chat queries
- Streams responses token-by-token for:
  - Section-wise simplification
  - Conversational responses

---

### 📄 Intelligent Document Processing
- Parses uploaded research papers
- Splits content into structured sections
- Filters out **irrelevant or low-value sections**
- Prepares optimized chunks for downstream processing

---

### 🧠 LLM Integration (LangChain)
- Uses **LangChain** for orchestrating LLM workflows
- Supports:
  - Section-level simplification
  - Context-aware Q&A
- Structured prompt pipelines for consistent outputs

---

### 🔍 Vector-Based Retrieval (Redis)
- Stores embeddings in **Redis (vector store)**
- Enables:
  - Fast similarity search
  - Context retrieval for queries
- Optimized for low-latency lookups

---

### 💬 Context-Aware Query System
- Retrieves relevant chunks based on user queries
- Supports:
  - Section-specific queries
  - Whole-document understanding
- Enhances accuracy of generated responses

---

### 🧵 Concurrent Task Handling
- Uses **async task execution** for parallel workflows
- Handles:
  - Chunk processing
  - Embedding generation
  - Streaming responses
- Improves throughput and responsiveness

---

### 🧱 Structured Architecture
- Modular design using **class-based components**
- Implements **Singleton pattern** for:
  - Shared resources (LLM, vector store, etc.)
- Ensures efficient resource utilization

---

### 🔄 Robust Error Handling
- Graceful failure handling across pipeline stages
- Prevents system-wide blocking on partial failures
- Provides consistent API responses

---

---

## 🛠️ Tech Stack

- **FastAPI**
- **Uvicorn**
- **LangChain**
- **Redis (Vector Store)**
- **Python (Async / asyncio)**

---

## 🔁 API Workflow

### Document Processing
- Upload PDF
- Extract and process sections
- Generate embeddings
- Store in vector database

### Query Handling
- Accept user query
- Retrieve relevant context from Redis
- Pass context to LLM via LangChain
- Stream response back to client

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/UvYadav04/ResearchSimplified-server
cd ResearchSimplified-server
python -m venv venv
source venv/bin/activate   # On Linux / Mac
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
