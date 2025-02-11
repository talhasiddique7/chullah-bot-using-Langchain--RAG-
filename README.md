# Chullah RAG Chatbot

A lightweight **Retrieval-Augmented Generation (RAG) chatbot** built using **LangChain**, **FAISS**, and **Llama 2** to answer queries based on a CSV knowledge base. Designed for **efficient local use within 8GB RAM**.

## 🚀 Features
- **RAG-based chatbot** using FAISS for fast retrieval.
- **Uses a CSV file** as a knowledge base.
- **Runs locally on CPU** with optimized embeddings.
- **Lightweight Llama 2 model** (GGML version).
- **Streamlit UI** for an interactive experience.

## 🛠 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/talhasiddique7/chullah-bot-using-Langchain--RAG-.git
   cd chullah bot using Langchain (RAG)
   ```

2. Install Python 3.10 (if not installed):
   ```bash
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev
   ```

3. Create a virtual environment and activate it:
   ```bash
   python3.10 -m venv rag_env
   source rag_env/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📥 Preparing the Data

Place your **CSV file** (e.g., `data.csv`) in the root directory.

Run the following command to **embed the data and create a FAISS index**:
   ```bash
   python ingest.py
   ```

## 🏃 Running the Chatbot

Start the chatbot with:
   ```bash
   streamlit run app.py
   ```

## 📌 Example Queries
- *"How do I place an order?"*
- *"What payment methods are accepted?"*
- *"Can I track my order?"*

## 💡 Technologies Used
- **LangChain** for RAG processing.
- **FAISS** for efficient vector search.
- **Sentence-Transformers** for embeddings.
- **CTransformers** for running Llama 2 locally.
- **Streamlit** for an interactive UI.

## 👤 Author
**Talha Sidduqe**

---

💬 *Feel free to contribute and improve this project!*

