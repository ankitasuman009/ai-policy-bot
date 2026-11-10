from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# Step 1: Initialize the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Connect to the existing ChromaDB collection
vector_store = Chroma(
    collection_name="company_policy",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Step 3: Initialize our local LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("RAG application is ready!")

question = "What is the company's maternity leave policy?"

# Retrieve relevant chunks
retrieved_docs = vector_store.similarity_search(
    question,
    k=3
)

print("\nRetrieved context:")

for doc in retrieved_docs:
    print(doc.page_content)
    print("Page:", doc.metadata.get("page_label"))
    print("-------------------")

# Combine retrieved chunks into one context
context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

# Create the prompt
prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the provided context.

Follow these rules strictly:

1. Do not use your general knowledge to answer questions.
2. Do not invent or assume any company policy.
3. If the answer is not available in the context,
   say that the information could not be found.
4. If the document contains placeholders such as
   [Company name], [two months], or [Insert this if...],
   do not treat them as confirmed company policies.
5. If a policy is conditional or incomplete,
   clearly explain that it is conditional or incomplete.
6. Never present template content as an officially
   confirmed company policy.
7. Never present text inside square brackets as a confirmed value.
   If a relevant policy contains bracketed placeholders,
   explicitly identify them as placeholders.

Context:
{context}

Question:
{question}

Answer:
"""

# Generate the answer
response = llm.invoke(prompt)

print("\nFinal Answer:")
print(response.content)

print("\nSources:")

unique_sources = set()

for doc in retrieved_docs:
    source = doc.metadata.get("source")
    page = doc.metadata.get("page_label")

    unique_sources.add((source, page))

for source, page in sorted(unique_sources):
    print(f"- {source} (Page {page})")