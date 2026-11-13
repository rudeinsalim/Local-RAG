import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "chroma_db"
DOC_DIRECTORY = "docs"

def ingest_documents():
    if not os.path.exists(DOC_DIRECTORY):
        os.makedirs(DOC_DIRECTORY)
        print(F"Created directory: {DOC_DIRECTORY}/ directory. please add documents there.")


    #1. Load documents from the DOC_DIRECTORY
    raw_documents = []
    for filename in os.listdir(DOC_DIRECTORY):
        file_path = os.path.join(DOC_DIRECTORY, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            raw_documents.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            raw_documents.extend(loader.load())
        
    if not raw_documents:
        print("No documents found to ingest. Add a .txt or .pdf file to the docs folder.")
        return
    
    print*(f"Loaded {len(raw_documents)} raw document page(s).")

    #2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks of text.")


    #3 Initialize embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    #4. Create vector store
    print(f"Persisting vector embeddings to '{PERSIST_DIRECTORY}' directory...")
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    print("Ingestion complete! Knowledge base is ready for use.")

    if __name__ == "__main__":
        ingest_documents()