import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
        retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 2, 'fetch_k': 5})
        st.success("✅ Database loaded successfully!")
        return retriever
    except Exception as e:
        st.error(f"❌ Error loading FAISS database: {e}")
        return None

@st.cache_resource
def load_llm():
    """ Load lightweight LLM for efficiency """
    st.write("🔄 Loading Language Model...")
    try:
        llm = CTransformers(
            model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  # Adjust if needed
            model_type="mistral",
            max_new_tokens=80,
            temperature=0.3,
            verbose=False
            
        )
        st.success("✅ LLM loaded successfully!")
        return llm
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

retriever = load_faiss()
llm = load_llm()

if retriever and llm:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Enter your query:")

    if st.button("Ask"):
        if query.strip():
            with st.spinner("⏳ Thinking..."):
                try:
                    # Apply prompt template
                    formatted_query = custom_prompt.format(query=query)
                    response = qa_chain.invoke({"query": formatted_query})

                    # Check for irrelevant responses
                    if "I don't know" in response["result"] or response["result"].strip() == "":
                        st.warning("⚠️ I can only answer questions related to Chullah.")
                    else:
                        st.subheader("**Answer:**")
                        response_text = response["result"]
                        response_text = response_text.replace("Q:", "").replace("A:", "").strip()
                        st.write(response_text)


                        # Display sources
                        sources = response.get("source_documents", [])
                        if sources:
                            st.subheader("📌 Sources:")
                            for i, source in enumerate(sources, 1):
                                st.write(f"{i}. {source.metadata.get('source', 'Unknown Source')}")
                        else:
                            st.info("No relevant sources found.")
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        else:
            st.warning("⚠️ Please enter a valid query.")
