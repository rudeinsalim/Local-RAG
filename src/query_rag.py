import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "chroma_db"

def query_knowledge_base(question: str):
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Error: Vector database not found at '{PERSIST_DIRECTORY}'. Please run the ingestion script first.")
        return

    #1. Load the local embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #2. Load persistent vector database
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    #3. Perform similarity search to retrieve relevant documents
    k_results = 3
    print(f"\nQuerying the knowledge base for: '{question}'")
    retrieved_docs = vector_store.similarity_search(question, k=k_results)

    #4. Display the retrieved documents
    print("--- Retrieved Documents ---")
    for idx, (doc, score) in enumerate(retrieved_docs, 1):
        print(f"\n[Result {idx} | Similarity score: {round(score, 4)}]")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("---------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your question: ")

    query_knowledge_base(user_query)