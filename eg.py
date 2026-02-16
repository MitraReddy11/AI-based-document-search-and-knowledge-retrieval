from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

pdf_loader = DirectoryLoader(path="./data", glob="**/*.pdf",
                             loader_cls=UnstructuredPDFLoader)

text_loader = DirectoryLoader(path="./data", glob="**/*.txt",
                              loader_cls=TextLoader)

csv_loader = DirectoryLoader(path="./data", glob="**/*.csv",
                             loader_cls=CSVLoader)

pdf_docs = pdf_loader.load()
text_docs = text_loader.load()
csv_docs = csv_loader.load()

docs = pdf_docs + text_docs + csv_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

chunks = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents=chunks,
                                     embedding=embedding_model,
                                     persist_directory="./chroma_db",
                                     collection_name="fresh_session")

vector_store.persist()

print(docs[0].page_content)
