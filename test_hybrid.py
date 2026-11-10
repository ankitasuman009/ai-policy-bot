from collections import defaultdict

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import re

def label_conditional_sections(text):

    pattern = r'(\[Insert\b.*?\])'

    return re.sub(
        pattern,
        r'\n[CONDITIONAL TEMPLATE SECTION - NOT CONFIRMED COMPANY POLICY]\n\1\n',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
def label_unresolved_placeholders(text):

    pattern = r'\[(?!Insert\b|CONDITIONAL TEMPLATE SECTION\b)([^\]]+)\]'

    def replace_placeholder(match):
        value = match.group(1).strip()

        return f'[UNRESOLVED TEMPLATE VALUE: {value}]'

    return re.sub(
        pattern,
        replace_placeholder,
        text,
        flags=re.IGNORECASE
    )

def remove_conditional_sections(text):
    pattern = (
        r'\[CONDITIONAL TEMPLATE SECTION - NOT CONFIRMED COMPANY POLICY\]'
        r'\s*\[Insert\b.*?\]'
    )

    return re.sub(
        pattern,
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
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

# Step 3: Initialize Ollama
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# Step 3: Load all documents from ChromaDB
all_data = vector_store.get(include=["documents", "metadatas"])

print("Total documents:", len(all_data["documents"]))
# Step 4: Convert documents and metadata into Document objects since the BM25Retriever expects a list of Document objects
documents = [
    Document(
        page_content=content,
        metadata=metadata
    )
    for content, metadata in zip(
        all_data["documents"],
        all_data["metadatas"]
    )
]

print("Documents converted:", len(documents))

# Step 4: Create BM25 retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Step 5: Create semantic retriever
semantic_retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)

# Step 6: Combine both retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

query_prompt = ChatPromptTemplate.from_template("""
    You are a search query generation assistant for a company policy document.

    Given the user's question, generate 6 distinct search queries.

    Rules:
    - Each query must explore a different aspect of the original question.
    - Avoid generating queries that mean almost the same thing.
    - Cover different relevant policy categories.
    - Do not introduce topics unrelated to the original question.
    - Return only the queries, one per line.
    - Do not include numbering or explanations.

    Original question:
    {question}
    """)

query_generation_chain = query_prompt | llm | StrOutputParser()


query = "What is the company's leave policy?"

generated_queries = query_generation_chain.invoke({
    "question": query
})

queries = [
    q.strip()
    for q in generated_queries.split("\n")
    if q.strip()
]

print("\nGenerated queries:")
for q in queries:
    print("-", q)


# Store RRF scores for each unique document
doc_scores = defaultdict(float)
doc_objects = {}

# Retrieve documents for each generated query
for q in queries:
    docs = ensemble_retriever.invoke(q)
    # Calculate RRF score based on document position
    for rank, doc in enumerate(docs, 1):
        doc_id = doc.page_content
        rrf_score = 1 / (60 + rank)
        doc_scores[doc_id] += rrf_score
        doc_objects[doc_id] = doc


# Sort documents by their combined RRF scores
ranked_docs = sorted(
    doc_scores.items(),
    key=lambda item: item[1],
    reverse=True
)

print("\nTotal unique documents:", len(ranked_docs))

# Select the top 10 documents
top_k = 10

final_docs = [
    doc_objects[doc_id]
    for doc_id, score in ranked_docs[:top_k]
]

# print("\nFinal documents selected:", len(final_docs))

# print("\n========== DOCUMENTS SENT TO LLM ==========")

# for i, doc in enumerate(final_docs, 1):
#     print(f"\nDocument {i} | Page: {doc.metadata.get('page_label')}")
#     print(doc.page_content)
#     print("-" * 80)

# # Step 7: Prepare the retrieved documents as context
context = "\n\n".join(
    remove_conditional_sections(
        label_unresolved_placeholders(
            label_conditional_sections(doc.page_content)
        )
    )
    for doc in final_docs
)

print("\nConditional sections:",
      context.count("[CONDITIONAL TEMPLATE SECTION"))

print("\nUnresolved placeholders:",
      context.count("[UNRESOLVED TEMPLATE VALUE:"))

# print("\n========== LABELED CONTEXT ==========")
# print(context)

# Step 8: Create a prompt for the LLM
prompt = ChatPromptTemplate.from_template("""
    You are a company policy assistant.

    The provided context comes from an employee handbook TEMPLATE.
    It may contain optional sections and unresolved placeholders.
    It must not automatically be treated as a finalized company policy.

    Rules:

    1. Use only the information explicitly stated in the provided context.
    2. Never invent, assume, or complete missing policy details.
    3. Content marked [CONDITIONAL TEMPLATE SECTION - NOT CONFIRMED COMPANY POLICY]
        is optional template content. Never present it as an established company policy.
    4. Content marked [UNRESOLVED TEMPLATE VALUE: ...] represents
        information that has not been finalized. Never replace it with a specific value.
    5. If a policy contains unresolved values, clearly mention that the details
        are not finalized in the handbook template.
    6. Do not combine separate sections to create new benefits or entitlements.
    7. Do not change the meaning of the source text.
    8. If the context does not provide enough information, explicitly say so.
    9. Distinguish between information described in the template and
        information confirmed as an actual company policy.

    Policy context:
    {context}

    User question:
    {question}

    Answer:
    """)

# Step 9: Generate the answer
answer_chain = prompt | llm | StrOutputParser()

answer = answer_chain.invoke({
    "context": context,
    "question": query
})

print("\n========== FINAL ANSWER ==========")
print(answer)

# print("\n========== RANKED DOCUMENTS ==========")

# for rank, (doc_id, score) in enumerate(ranked_docs, 1):
#     doc = doc_objects[doc_id]
#     print(
#         f"\nRank: {rank} | "
#         f"Page: {doc.metadata.get('page_label')} | "
#         f"RRF Score: {score:.4f}"
#     )
#     print(doc.page_content[:300])
#     print("-" * 60)
    
    
# Run the same query through all three retrievers
# semantic_docs = semantic_retriever.invoke(query)
# bm25_docs = bm25_retriever.invoke(query)
# hybrid_docs = ensemble_retriever.invoke(query)

# # Limit hybrid results to the top 10 after RRF ranking
# hybrid_docs = hybrid_docs[:10]


# def print_results(name, docs):
#     print(f"\n\n========== {name} ==========")

#     for i, doc in enumerate(docs, 1):
#         print(f"\nRank {i} | Page {doc.metadata.get('page_label')}")
#         print(doc.page_content[:250])
#         print("-" * 60)


# print_results("SEMANTIC SEARCH", semantic_docs)
# print_results("BM25 SEARCH", bm25_docs)
# print_results("HYBRID SEARCH", hybrid_docs)