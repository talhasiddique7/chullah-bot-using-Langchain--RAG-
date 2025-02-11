import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load CSV
csv_path = "Chullah_Chatbot_Data.csv"
df = pd.read_csv(csv_path, delimiter=",", quotechar='"')

# Combine all columns into a single text field
df["text"] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)

# Load data into LangChain
loader = DataFrameLoader(df, page_content_column="text")
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
documents = text_splitter.split_documents(docs)

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store
db = FAISS.from_documents(documents, embeddings)

# Save FAISS index locally
db.save_local("vectorstore/db_faiss")

print("Embedding and FAISS index creation complete.")
