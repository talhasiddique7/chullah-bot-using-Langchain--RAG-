import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def clean_csv_data(csv_path):
    df = pd.read_csv(csv_path, delimiter=",", quotechar='"')
    
    # Drop duplicates based on the 'Question' column
    df = df.drop_duplicates(subset=['User'])

    # Create a unified text format (Question + Answer)
    df["text"] = df.apply(lambda row: f"Q: {row['User']} A: {row['Bot']}", axis=1)

    return df

# Load and clean CSV data
csv_path = "data/Chullah_Chatbot_Data.csv"
df = clean_csv_data(csv_path)

# Load data into LangChain
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Load optimized embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Create FAISS vector store
db = FAISS.from_documents(documents, embeddings)

# Save FAISS index locally
db.save_local("vectorstore/db_faiss")

print("✅ Embedding and FAISS index creation complete.")
