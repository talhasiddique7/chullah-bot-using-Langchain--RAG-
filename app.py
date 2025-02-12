import streamlit as st
import time
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Ensure this is set in .env

if not HF_API_KEY:
    st.error("❌ Hugging Face API key is missing! Set HF_API_KEY in .env.")

# Initialize Hugging Face Inference Client
client = InferenceClient(provider="sambanova", api_key=HF_API_KEY)

# Streamlit UI
st.title("🔥 Chullah RAG Chatbot")
st.write("Ask me anything about Chullah!")

# Define Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["query"],
    template="You are an AI assistant for Chullah. "
             "Respond in a clear and **direct** sentence format, avoiding Q: and A: structures. "
             "If the question is unrelated to Chullah, simply say: "
             "'I can only answer questions related to Chullah.' \n\n"
             "{query}"
)

@st.cache_resource
def load_faiss():
    """ Load FAISS database with embeddings """
    st.write("🔄 Loading FAISS vector database...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
        db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 5})
        st.success("✅ FAISS database loaded successfully!")
        return retriever
    except Exception as e:
        st.error(f"❌ Error loading FAISS database: {e}")
        return None

def query_huggingface(prompt):
    """ Query Hugging Face Together API """
    messages = [
        {"role": "system", "content": "you are an AI assistant for Chullah. Respond in a clear and direct sentence format, avoiding Q: and A: structures. If the question is unrelated to Chullah, simply say: 'I can only answer questions related to Chullah.'"},
        {"role": "user", "content": prompt}
        
        ]
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=100,
            temperature=0.3,
        )
        return completion.choices[0].message.content.strip() if completion.choices else "No response."
    
    except Exception as e:
        return f"Error: {e}"

# Load FAISS retriever
retriever = load_faiss()

if retriever and HF_API_KEY:
    query = st.text_input("Enter your query:")

    if st.button("Ask"):
        if query.strip():
            with st.spinner("⏳ Thinking..."):
                try:
                    start_time = time.time()  # Start timing
                    
                    # Retrieve relevant documents
                    docs = retriever.get_relevant_documents(query)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Apply prompt template
                    formatted_query = custom_prompt.format(query=query)
                    
                    # Query Hugging Face API via Together provider
                    response_text = query_huggingface(formatted_query)
                    
                    end_time = time.time()  # End timing
                    response_time = round(end_time - start_time, 2)  # Calculate response time
                    
                    # Display response
                    if "I don't know" in response_text or response_text.strip() == "":
                        st.warning("⚠️ I can only answer questions related to Chullah.")
                    else:
                        st.subheader("**Answer:**")
                        st.write(response_text)
                        st.write(f"⏱ Response Time: {response_time} seconds")
                        
                        # Display sources
                        if docs:
                            st.subheader("📌 Sources:")
                            for i, doc in enumerate(docs, 1):
                                st.write(f"{i}. {doc.metadata.get('source', 'Unknown Source')}")
                        else:
                            st.info("No relevant sources found.")
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        else:
            st.warning("⚠️ Please enter a valid query.")