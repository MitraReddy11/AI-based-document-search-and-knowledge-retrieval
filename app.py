import os
from dotenv import load_dotenv
from startup import startup_cleanup

from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


startup_cleanup()

load_dotenv()
GROQ_API_KEY = os.getenv("Groq_API_KEY")
pdf_loader = DirectoryLoader(
    path="./data",
    glob="**/*.pdf",
    loader_cls=UnstructuredPDFLoader
)

text_loader = DirectoryLoader(
    path="./data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = pdf_loader.load() + text_loader.load()
print(f"Loaded {len(documents)} documents")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")


def clean_metadata(docs):
    for doc in docs:
        doc.metadata = {k: str(v) for k, v in doc.metadata.items()}
    return docs

cleaned_docs = clean_metadata(chunks)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=cleaned_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="fresh_session"
)

vector_store.persist()

print("Vector database created successfully!")


model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

print("Model initialized successfully!")
