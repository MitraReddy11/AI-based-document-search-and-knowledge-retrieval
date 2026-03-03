📌 Project Overview
This project implements the data ingestion and indexing layer of a Retrieval-Augmented Generation (RAG) system using a Corrective RAG approach.
The system loads documents (PDF, TXT, CSV), processes them into chunks, generates embeddings, and stores them in a vector database (Chroma) for retrieval-based question answering.

Corrective RAG improves traditional RAG by ensuring:
Cleaned metadata
Controlled chunking
Consistent indexing
Reduced context leakage

Documents (PDF/TXT/CSV)
        ↓
DirectoryLoader
        ↓
Text Chunking
        ↓
Embedding Generation (MiniLM)
        ↓
Chroma Vector Store
        ↓
Ready for Retrieval

📂 Supported File Types
.pdf → Loaded using UnstructuredPDFLoader
.txt → Loaded using TextLoader
.csv → Loaded using CSVLoader
All supported files inside the ./data folder are automatically detected.

⚙️ Technologies Used
LangChain
ChromaDB
HuggingFace Embeddings
Sentence Transformers
Python 3.10+

🧠 What is Corrective RAG?
Traditional RAG may suffer from:
Noisy metadata
Context leakage
Poor chunking strategy
Mixed session indexing
Corrective RAG ensures:
Clean document ingestion
Controlled chunk sizes
Proper embedding initialization
Persistent and isolated vector collections
Structured retrieval-ready storage
This ensures higher accuracy during answer generation.

Project Structure:
rag-copilot/
│
├── data/
│   ├── sample.pdf
│   ├── sample.txt
│   └── sample.csv
│
├── chroma_db/
│
├── eg.py
├── requirements.txt
└── README.md
