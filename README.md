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

## ⚡ Performance Optimization Notes  

For **faster responses** (within 3 seconds), consider the following:  

1. **Use a smaller model**:  
   - Instead of large models like `Mistral-7B`, use **TinyLlama-1.1B** or `GPT-2` variations.  
   - Recommended model:  
     ```python
     model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
     model_type="llama"
     ```
   
2. **Reduce max tokens**:  
   - Set `max_new_tokens=50` or lower to minimize generation time.  

3. **Adjust temperature and top-p**:  
   - Use `temperature=0.3` for deterministic outputs.  
   - Set `top_p=0.9` to prioritize high-probability responses.  

4. **Use a faster embedding model**:  
   - Instead of `all-MiniLM-L12-v2`, use **`all-MiniLM-L6-v2`** for better speed.  

5. **Limit retrieval results**:  
   - Adjust FAISS retriever parameters:  
     ```python
     retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 2, 'fetch_k': 5})
     ```
   - This reduces the number of retrieved documents, speeding up processing.  

6. **Optimize local execution**:  
   - Run on a machine with **at least 8GB RAM** for smooth performance.  
   - Use **GPU acceleration** if available (via `llama-cpp-python` or `CTransformers`).  

By applying these optimizations, your **Chullah RAG chatbot** will respond in under **3 seconds**. 🚀  


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

