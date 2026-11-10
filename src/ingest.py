import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re
from langchain_core.documents import Document

PDF_PATH = "documents/company_policy.pdf"

# load the PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# # split the documents into smaller chunks
# Split documents while keeping track of section headings

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

section_pattern = re.compile(
    r"^[ \t]*(Paid time off \(PTO\)|Holidays|Sick leave|Long-term illness|"
    r"Bereavement leave|Jury duty and voting|Parental leave|"
    r"Paternity and maternity leave|Returning to work after parental leave|"
    r"Leaving our company)[ \t]*$",
    re.IGNORECASE | re.MULTILINE
)

chunks = []
current_section = "Unknown"

for page in documents:
    page_text = page.page_content

    # Find all recognized section headings on this page
    matches = list(section_pattern.finditer(page_text))
    # if matches:
    #     print(
    #         f"Page {page.metadata.get('page_label')}:",
    #         [match.group(1).strip() for match in matches]
    #     )
    # Include text before the first heading in the previous section
    sections = []
    start = 0

    for match in matches:
        section_text = page_text[start:match.start()].strip()

        if section_text:
            sections.append((current_section, section_text))

        current_section = match.group(0).strip()
        start = match.end()

    # Include the remaining text after the last heading
    remaining_text = page_text[start:].strip()

    if remaining_text:
        sections.append((current_section, remaining_text))

    # Split each section independently
    for section_name, section_text in sections:
        section_doc = Document(
            page_content=section_text,
            metadata={
                **page.metadata,
                "section": section_name
            }
        )

        section_chunks = text_splitter.split_documents([section_doc])
        chunks.extend(section_chunks)

print(f"Total chunks generated: {len(chunks)}")

for i, chunk in enumerate(chunks):
    if 25 <= chunk.metadata.get("page", -1) <= 30:
        print(
            f"\nChunk {i} | "
            f"Page: {chunk.metadata.get('page_label')} | "
            f"Section: {chunk.metadata.get('section')}"
        )
        print(chunk.page_content)

# load environment variables from .env file
load_dotenv()

# Initialize the OpenAI embeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# store chunks and their embedding in chroma vector database
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="company_policy")

print("Embedded generated and stored in Chroma vector database successfully.")

# Step 7: Verify stored documents
print("\nVerifying ChromaDB...")

print("Total stored chunks:", vector_store._collection.count())

# Retrieve one sample document
results = vector_store.similarity_search(
    "How many days of sick leave do employees receive?",
    k=2
)

for i, doc in enumerate(results):
    print(f"\nDocument {i + 1}:")
    print("Section:", doc.metadata.get("section"))
    print("Page:", doc.metadata.get("page_label"))
    print(doc.page_content)