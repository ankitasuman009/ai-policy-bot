from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Step 1: Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Connect to ChromaDB
vector_store = Chroma(
    collection_name="company_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


print("RAG components initialized successfully!")

queries = [
    "sick leave",
    "paid time off PTO",
    "parental maternity paternity leave",
    "bereavement leave",
    "holidays"
]

for query in queries:
    print(f"\n\nQUERY: {query}")

    docs = vector_store.similarity_search(query, k=3)

    for doc in docs:
        print(
            f"Page {doc.metadata.get('page_label')}: "
            f"{doc.page_content[:200]}"
        )
 