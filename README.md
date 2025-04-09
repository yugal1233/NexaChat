# NexaChat – Lightweight LLM Chatbot

## 📌 Overview

This project allows you to upload PDFs, extract text, convert it into embeddings using Ollama's Embedding Model, and store them in a Milvus vector database. You can then perform semantic searches and AI Chat on the stored text.

🔗 [Full System Design Notion Write-up](https://desert-burst-2c1.notion.site/NexaChat-Lightweight-LLM-Chatbot-1d03ae4e7994808b8482e52ed88b375b?pvs=4)

## 🏗️ Project Structure

```
app/
│-- models/               # Pretrained embedding model and Large Language Models
│-- uploads/              # Directory for uploaded PDF files
│-- main.py               # Main Streamlit app
│-- requirements.txt      # Dependencies
│-- Dockerfile            # Docker container setup
│-- .dockerignore         # Files to exclude from Docker image
│-- README.md             # Project documentation
```

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- Ollama application
- Milvus installed and running (Preferable in Docker)
- Docker (if using containerization)

### Setup

1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd app
   ```
2. Install Ollama
3. Pull Ollama LLM Model:
   ```sh
   ollama pull gemma3
   ```
4. Pull Ollama Embedding Model:
   ```sh
   ollama pull nomic-embed-text
   ```
5. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
6. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Run Locally

1. Start Milvus:
   ```sh
   docker run -d --name milvus-standalone -p 19530:19530 milvusdb/milvus:latest
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run src/app_v3.py -- --host "replace_with_milvus_host_ip" --port "19530" --ollama_model "gemma3"
   ```

### Run with Docker

1. Build the Docker image:
   ```sh
   docker build -t pdf-milvus-app .
   ```
2. Run the container:
   ```sh
   docker run --network my-network -p 8501:8501 pdf-milvus-app-new -- --host "replace_with_milvus_host_ip" --port "19530" --ollama_model "gemma3"
   ```

## 📂 Features

- Upload PDFs and extract text
- Chunk text using LangChain
- Convert text into embeddings with Sentence Transformers
- Store embeddings in Milvus
- Perform semantic search on stored PDF content

## 🤖 Model Used

- `all-MiniLM-L6-v2` from `sentence-transformers`
- `gemma3` from `ollama`

## 🛠️ Troubleshooting

- Ensure Milvus is running before starting the app.
- If running in a restricted environment, download the model manually and place it in `models/`.

## 📜 License

This project is licensed under the MIT License.