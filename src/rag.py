from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import re

from typer import prompt

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

# Load documents from ChromaDB safely
stored_data = vector_store.get(include=["documents", "metadatas"])

all_docs = [
    Document(page_content=content, metadata=metadata or {})
    for content, metadata in zip(stored_data["documents"], stored_data["metadatas"])
    if content.strip()
]

# Initialize Retrievers
bm25_retriever = BM25Retriever.from_documents(all_docs, k=5)
semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]
)

# Initialize LLM
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Dynamic RAG components initialized successfully!")

def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    if not history:
        return question
    recent_history = history[-6:]
    conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent_history)
    
    prompt = f"""You are a search query rewriting assistant.
        Rewrite the user's latest question into a concise, standalone search query for searching policy documents.
        Conversation history:
        {conversation}
        Latest question:
        {question}
        Rewritten search query:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()


def retrieve_relevant_documents(search_query: str, top_k: int = 10):
    # Keep focused questions focused. LLM-generated query variants often pull
    # in loosely related sections, which then pollute both the answer context
    # and the source list. The hybrid retriever already combines keyword and
    # semantic matching for the original query.
    retrieved_docs = hybrid_retriever.invoke(search_query)

    # Hybrid ranking can miss a section whose heading directly matches a
    # specific question. Add those indexed section chunks before weaker hits.
    query_terms = _topic_terms(search_query)
    section_matches = [
        doc for doc in all_docs
        if query_terms.intersection(
            _topic_terms(str(doc.metadata.get("section", "")))
        )
    ]

    combined = []
    seen_content = set()
    for doc in section_matches + retrieved_docs:
        content_key = re.sub(r"\s+", " ", doc.page_content).casefold()
        if content_key not in seen_content:
            seen_content.add(content_key)
            combined.append(doc)
        if len(combined) >= top_k:
            break
    return combined

def label_unresolved_placeholders(text: str) -> str:
    # Mark complete optional sections before handling single-line placeholders.
    text = re.sub(
        r"\[Insert\b.*?\]",
        lambda match: (
            "[CONDITIONAL TEMPLATE CONTENT, not confirmed policy: "
            + match.group(0)[1:-1].strip()
            + "]"
        ),
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Mark bracketed values while leaving the conditional label intact.
    pattern = r'\[([^\]\n]+)\]'
    def replace_placeholder(match):
        val = match.group(1).strip()
        if val.lower().startswith("insert"):
            return f"[CONDITIONAL TEMPLATE CONTENT, not confirmed policy: {val}]"
        if "conditional" in val.lower():
            return match.group(0)
        return f'[UNRESOLVED TEMPLATE VALUE: {val}]'
    return re.sub(pattern, replace_placeholder, text, flags=re.IGNORECASE)

def clean_placeholder_markers(answer: str, context: str = "") -> str:
    pattern = r'\[UNRESOLVED TEMPLATE VALUE:\s*(.*?)\]'
    marker_values = re.findall(pattern, answer, flags=re.IGNORECASE)
    answer = re.sub(
        pattern,
        lambda match: f"[{match.group(1).strip().rstrip(' .')}]",
        answer,
        flags=re.IGNORECASE,
    )

    # If the model removed the marker itself but repeated a value from a
    # marked context value, restore brackets around it without discarding its answer.
    context_values = re.findall(pattern, context, flags=re.IGNORECASE)
    contains_unmarked_value = False
    for raw_value in context_values:
        value = raw_value.strip().rstrip(" .")
        if not value:
            continue
        value_pattern = rf"(?<![\w\[])({re.escape(value)})(?![\w\]])"
        exact_matches = list(re.finditer(value_pattern, answer, flags=re.IGNORECASE))
        if exact_matches:
            contains_unmarked_value = True
            answer = re.sub(
                value_pattern,
                lambda match: f"[{match.group(1)}]",
                answer,
                flags=re.IGNORECASE,
            )
            continue

        # For paraphrases such as “through the HRIS” vs “[through our HRIS]”,
        # use the distinctive content word as an anchor and bracket its short
        # local phrase. This is value-agnostic and does not alter the answer.
        anchors = [
            token for token in re.findall(r"[a-z0-9]+", value.casefold())
            if len(token) >= 4 and token not in _STOP_WORDS
        ]
        if not anchors:
            continue
        anchor_pattern = rf"(?<![\w\[])({re.escape(anchors[-1])})(?![\w\]])"
        anchor_matches = list(re.finditer(anchor_pattern, answer, flags=re.IGNORECASE))
        for match in reversed(anchor_matches):
            # Don't rewrap an already bracketed or quoted template value.
            if answer.rfind("[", 0, match.start()) > answer.rfind("]", 0, match.start()):
                continue
            contains_unmarked_value = True
            sentence_start = max(
                answer.rfind(".", 0, match.start()),
                answer.rfind("!", 0, match.start()),
                answer.rfind("?", 0, match.start()),
                answer.rfind("\n", 0, match.start()),
            ) + 1
            previous_words = list(re.finditer(r"\b[\w’'-]+\b", answer[sentence_start:match.start()]))
            phrase_start = (
                sentence_start + previous_words[-2].start()
                if len(previous_words) >= 2
                else sentence_start + previous_words[-1].start()
                if previous_words
                else match.start()
            )
            answer = answer[:phrase_start] + "[" + answer[phrase_start:match.end()] + "]" + answer[match.end():]
    if marker_values or contains_unmarked_value:
        answer = answer.rstrip() + (
            "\n\nBracketed values are unfilled template placeholders and are "
            "not confirmed company policy."
        )

    # Small models may abstain even when a relevant optional section is in the
    # context. Surface the retrieved section and its condition instead of
    # inventing a policy summary.
    abstains = re.search(
        r"(?:does|do) not (?:provide|specify|describe)|"
        r"doesn't (?:provide|specify|describe)|no (?:specific )?(?:guidance|information)",
        answer,
        flags=re.IGNORECASE,
    )
    if abstains and not re.search(r"conditional template text", answer, flags=re.IGNORECASE):
        blocks = re.findall(
            r"\[CONDITIONAL TEMPLATE CONTENT, not confirmed policy:\s*(.*?)\]",
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if blocks:
            block = re.sub(r"\s+", " ", blocks[0]).strip()
            instruction = re.match(r"(Insert\b.*?:)\s*(.*)", block, flags=re.IGNORECASE)
            if instruction:
                condition, body = instruction.group(1).rstrip(":. "), instruction.group(2).strip()
            else:
                condition, body = "The excerpt is marked as conditional", block
            body = re.sub(
                r"\[UNRESOLVED TEMPLATE VALUE:\s*(.*?)\]",
                lambda match: f"[{match.group(1).strip()}]",
                body,
                flags=re.IGNORECASE,
            )
            sentences = re.split(r"(?<=[.!?])\s+", answer, maxsplit=1)
            if sentences and re.search(
                r"(?:does|do) not (?:provide|specify|describe)|"
                r"doesn't (?:provide|specify|describe)|no (?:specific )?(?:guidance|information)",
                sentences[0],
                flags=re.IGNORECASE,
            ):
                remainder = f" {sentences[1]}" if len(sentences) > 1 else ""
                answer = (
                    "The retrieved excerpt does not confirm that the conditional text "
                    "applies as active policy."
                    + remainder
                )
            answer = answer.rstrip() + (
                "\n\nThe handbook also contains conditional template text "
                f"(applicability unconfirmed): {condition}. It states: {body}"
            )
    return answer


_STOP_WORDS = {
    "what", "when", "where", "which", "does", "about", "tell", "explain",
    "policy", "handbook", "company", "employee", "employees", "their", "this",
    "that", "with", "from", "have", "the", "and", "for", "are", "can", "how",
    "leav", "does", "say", "says", "handle", "handles", "template", "unspecified",
    "detail", "details", "point", "out", "any", "specific", "summarize",
}


def _topic_terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z0-9]+", text.casefold()):
        if token in {"leave", "leaves", "leaving"}:
            token = "leav"
        elif token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if (len(token) < 4 and token != "pto") or token in _STOP_WORDS:
            continue
        terms.add(token)
    return terms


def format_sources(docs, limit: int = 3, question: str = "") -> list[dict]:
    """Prefer source sections whose names match meaningful question terms."""
    question_terms = _topic_terms(question)
    scored_docs = [
        (
            len(question_terms.intersection(
                _topic_terms(str(doc.metadata.get("section", "")))
            )),
            doc,
        )
        for doc in docs
    ]
    best_score = max((score for score, _ in scored_docs), default=0)
    ranked_docs = (
        [doc for score, doc in scored_docs if score == best_score]
        if best_score
        else docs
    )

    sources = []
    seen = set()
    for doc in ranked_docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page_label")
        key = (source, page)
        if not source or key in seen:
            continue
        seen.add(key)
        sources.append({"document": source, "page": page})
        if len(sources) >= limit:
            break
    return sources


def prepare_context(docs, question: str = "") -> str:
    # Give excerpts from sections matching the question's topic first, while
    # preserving the retriever's ranking among equally relevant excerpts.
    question_terms = _topic_terms(question)
    docs = sorted(
        enumerate(docs),
        key=lambda item: (
            -len(question_terms.intersection(
                _topic_terms(str(item[1].metadata.get("section", "")))
            )),
            item[0],
        ),
    )
    docs = [doc for _, doc in docs]
    section_relevant_docs = [
        doc for doc in docs
        if question_terms.intersection(
            _topic_terms(str(doc.metadata.get("section", "")))
        )
    ]
    # When the retriever found the section named in the question, exclude
    # unrelated sections that can tempt a small model to mix policies.
    if section_relevant_docs:
        docs = section_relevant_docs

    unique_chunks = []
    seen_content = set()
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        
        numbered_toc_rows = re.findall(r"(?m)^\s*.+?\s+\d{1,2}\s*$", content)
        if "table of contents" in content.lower() or len(numbered_toc_rows) >= 8:
            continue
            
        # Skip exact duplicates and chunks already fully covered by a stronger
        # retrieved chunk. Do not reorder the remaining results by page/length.
        normalized = re.sub(r"\s+", " ", content).casefold()
        if normalized in seen_content or any(
            normalized in re.sub(r"\s+", " ", existing.page_content).casefold()
            for existing in unique_chunks
        ):
            continue
        unique_chunks.append(doc)
        seen_content.add(normalized)
        if len(unique_chunks) >= 5:
            break

    processed_texts = []
    for idx, doc in enumerate(unique_chunks):
        cleaned = label_unresolved_placeholders(doc.page_content)
        page_label = doc.metadata.get("page_label", "Unknown Page")
        
        # We explicitly wrap every fragment in rigid, unambiguous text boundaries
        isolated_chunk = f"""
            ==================================================
            DOCUMENT EXCERPT #{idx + 1} (Page: {page_label})
            --------------------------------------------------
            {cleaned}
            ==================================================
        """
        processed_texts.append(isolated_chunk)

    return "\n\n".join(processed_texts)



def build_answer_prompt(question: str, context: str, history: list[dict] | None = None) -> str:
    conversation_history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in (history or [])[-6:]
    )

    return f"""Answer the latest question using only the handbook excerpts below.

Instructions:
- Answer directly and include every relevant, explicit policy detail. Do not say a topic is unspecified if the excerpts state a rule, condition, or exception.
- Preserve conditions and exceptions exactly. Do not add assumptions or outside facts.
- Text marked [UNRESOLVED TEMPLATE VALUE: ...] is an unfilled value. Never present it as confirmed policy.
- Explain confirmed parts of a sentence even when another value is unfilled; identify only that value as unspecified.
- If a relevant excerpt is marked CONDITIONAL TEMPLATE CONTENT, do not omit it or say no guidance exists. State its condition, summarize what it says, and clarify that the handbook does not confirm it applies.
- For lists of template values, report the full relevant list as unconfirmed template options.
- Do not invent a list of missing policy details. Mention only unresolved values that appear in the excerpts and matter to the question.
- Use conversation history only to understand follow-up references. Treat the retrieved excerpts as the only evidence about policy.
- Do not add related policies from other sections unless they answer the question.
- Do not cite internal excerpt numbers; source pages are shown separately.
- If the excerpts truly do not answer the question, say: "The provided handbook excerpts do not specify this."
- Never expose the internal UNRESOLVED TEMPLATE VALUE marker in your answer.
- Keep the answer concise and distinguish template language from confirmed policy.

Conversation history:
{conversation_history}

Retrieved policy excerpts:
{context}

Latest question: {question}

Answer:"""

def get_answer(question: str, history: list[dict] | None = None):
    search_query = rewrite_query(question, history)
    retrieved_docs = retrieve_relevant_documents(search_query)

    context = prepare_context(retrieved_docs, question)
    prompt = build_answer_prompt(question, context, history)

    response = llm.invoke(prompt)
    answer = clean_placeholder_markers(response.content, context)

    return {
        "answer": answer,
        "sources": format_sources(retrieved_docs, question=question)
    }

# get answers in chunks using streaming
def get_answer_stream(question: str, history: list[dict] | None = None):

    search_query = rewrite_query(question, history)

    retrieved_docs = retrieve_relevant_documents(search_query)

    context = prepare_context(retrieved_docs, question)
    prompt = build_answer_prompt(question, context, history)

    response_text = ""

    for chunk in llm.stream(prompt):
        if chunk.content:
            response_text += chunk.content

    cleaned_response = clean_placeholder_markers(response_text, context)

    yield {
        "type": "token",
        "content": cleaned_response
    }

    yield {
        "type": "sources",
        "sources": format_sources(retrieved_docs, question=question)
    }

    yield {"type": "done"}

# if __name__ == "__main__":
#     get_answer("What does the handbook say about workplace harassment?")

if __name__ == "__main__":
    test_questions = [
    "How many days of sick leave do employees receive?",
    "Can employees take time off when they are unwell?",
    "What happens if an employee has a long-term medical condition?",
    "How much paid time off can an employee take?",
    "What leave is available after the birth of a child?",
    "What happens to unused leave when an employee leaves?"
    ]

    for question in test_questions:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        docs = retrieve_relevant_documents(question, top_k=5)

        for i, doc in enumerate(docs, start=1):
            print(f"\nDocument {i}")
            print("Section:", doc.metadata.get("section"))
            print("Page:", doc.metadata.get("page_label"))
            print("Content:")
            print(doc.page_content[:600])
            print("-" * 60)
