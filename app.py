import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Load FAISS database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embeddings)
retriever = db.as_retriever(search_kwargs={'k': 5})

# Load lightweight model for efficiency
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_type="llama",
    max_new_tokens=100,
    temperature=0.3
)

# Setup QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Streamlit UI
st.title("Chullah RAG Chatbot")
st.write("Ask me anything about Chullah!")

query = st.text_input("Enter your query:")

if st.button("Ask"):
    if query:
        response = qa_chain.invoke({"query": query})
        st.write("**Answer:**", response["result"])
        sources = response.get("source_documents", [])
        
        if sources:
            st.write("\n**Sources:**")
            for i, source in enumerate(sources, 1):
                st.write(f"{i}. {source.metadata}")
        else:
            st.write("\nNo relevant sources found.")
