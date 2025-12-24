import os
import uuid
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE


# -----------------------------
# Step 2: Load PDFs + Metadata
# -----------------------------
def load_pdfs_with_metadata(pdf_paths: List[str]) -> List[Document]:
    """
    Loads multiple PDFs and returns page-wise Documents.

    Metadata added/ensured:
      - source: file name (e.g., "doc1.pdf")
      - page: page number (1-based for readability)
      - doc_id: unique id per PDF file (stable per run)
    """
    all_docs: List[Document] = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = str(uuid.uuid4())
        file_name = os.path.basename(pdf_path)

        loader = PyPDFLoader(pdf_path)
        pages: List[Document] = loader.load()  # one Document per page

        for d in pages:
            d.metadata = d.metadata or {}
            d.metadata["source"] = file_name
            d.metadata["doc_id"] = doc_id

            # PyPDFLoader sets page as 0-based; convert to 1-based for users
            page_0_based = d.metadata.get("page", None)
            d.metadata["page"] = (page_0_based + 1) if isinstance(page_0_based, int) else None

        all_docs.extend(pages)

    return all_docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits Documents into chunks while preserving metadata.
    Adds chunk_id for traceability.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    for i, ch in enumerate(chunks):
        ch.metadata = ch.metadata or {}
        ch.metadata["chunk_id"] = i

    return chunks


# -----------------------------
# OpenAI key handling
# -----------------------------
def _get_openai_api_key() -> str:
    """
    Loads .env and validates OPENAI_API_KEY.
    NOTE: OS env vars can override .env. Ensure OPENAI_API_KEY is correct.
    """
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY not found.\n"
            "Create a .env file in project root with:\n"
            "OPENAI_API_KEY=your_key_here"
        )
    return key


# -----------------------------
# Step 3: Embeddings + FAISS
# -----------------------------
def build_faiss_index(chunks: List[Document]) -> FAISS:
    """
    Creates embeddings and builds a FAISS vector store.
    """
    _ = _get_openai_api_key()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_faiss_index(vectorstore: FAISS, index_dir: str = "indexes/faiss_index") -> None:
    """
    Saves FAISS index to disk so you don't have to re-embed every run.
    """
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)


def load_faiss_index(index_dir: str = "indexes/faiss_index") -> Optional[FAISS]:
    """
    Loads FAISS index from disk if present.
    """
    if not os.path.exists(index_dir):
        return None

    _ = _get_openai_api_key()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Required for FAISS local load
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def retrieval_test(vectorstore: FAISS, query: str, k: int = 4) -> List[Document]:
    """
    Runs a similarity search and prints retrieved chunks with sources.
    Returns the retrieved docs.
    """
    results = vectorstore.similarity_search(query, k=k)

    print("\n====================")
    print("Query:", query)
    print("====================\n")

    for idx, doc in enumerate(results, start=1):
        meta = doc.metadata or {}
        preview = doc.page_content[:250].replace("\n", " ")

        print(f"--- Result {idx} ---")
        print(
            f"Source: {meta.get('source')} | "
            f"Page: {meta.get('page')} | "
            f"Chunk: {meta.get('chunk_id')} | "
            f"DocID: {meta.get('doc_id')}"
        )
        print("Preview:", preview)
        print()

    return results


# -----------------------------
# Step 4: RAG Answer + Citations
# -----------------------------
def format_context_with_citations(docs: List[Document]) -> str:
    """
    Builds a context string that includes source/page markers.
    The LLM will use these markers to ground answers.
    """
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")

        text = d.page_content.strip()
        blocks.append(
            f"[{i}] Source: {source} | Page: {page} | Chunk: {chunk_id}\n{text}"
        )
    return "\n\n".join(blocks)


def collect_citations(docs: List[Document]) -> List[str]:
    """
    Creates a clean, de-duplicated citation list like: "file.pdf (p. 2)".
    """
    seen = set()
    citations = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            citations.append(f"{src} (p. {page})")
    return citations


def answer_with_rag(vectorstore: FAISS, question: str, k: int = 5) -> Dict[str, Any]:
    """
    Retrieves top-k chunks and uses an LLM to answer grounded in those chunks.
    Returns answer + citations + retrieved docs.
    """
    retrieved_docs = vectorstore.similarity_search(question, k=k)

    context = format_context_with_citations(retrieved_docs)
    citations = collect_citations(retrieved_docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", RAG_USER_PROMPT_TEMPLATE),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {
        "answer": response.content,
        "citations": citations,
        "retrieved_docs": retrieved_docs,
    }


# -----------------------------
# Debug helpers
# -----------------------------
def debug_print_samples(docs: List[Document], n: int = 3) -> None:
    """
    Prints a few sample docs/chunks to verify metadata + content.
    """
    print(f"\nTotal documents/chunks: {len(docs)}\n")
    for i, d in enumerate(docs[:n]):
        meta = d.metadata or {}
        text_preview = d.page_content[:250].replace("\n", " ")
        print(f"--- Sample {i+1} ---")
        print("Metadata:", meta)
        print("Text preview:", text_preview)
        print()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("✅ rag_pipeline.py started running...")

    # ---------- Step 2: Load PDFs ----------
    sample_dir = "sample_pdfs"
    print("📂 Looking for PDFs inside:", os.path.abspath(sample_dir))

    if not os.path.exists(sample_dir):
        raise RuntimeError(f"Folder '{sample_dir}' not found. Create it and add PDFs.")

    pdf_files = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith(".pdf")
    ]
    print("📄 PDFs found:", pdf_files)

    if not pdf_files:
        raise RuntimeError("No PDFs found in sample_pdfs/. Add 1–2 PDFs and re-run.")

    page_docs = load_pdfs_with_metadata(pdf_files)
    debug_print_samples(page_docs, n=2)

    chunks = chunk_documents(page_docs, chunk_size=1000, chunk_overlap=150)
    debug_print_samples(chunks, n=3)

    # ---------- Step 3: Build/Load FAISS ----------
    index_dir = "indexes/faiss_index"
    print(f"\n🔎 Checking for existing FAISS index at: {index_dir}")

    vectorstore = load_faiss_index(index_dir=index_dir)

    if vectorstore is None:
        print("🚀 No index found. Building FAISS index (creating embeddings)...")
        vectorstore = build_faiss_index(chunks)
        print("✅ FAISS index ready. Saving to disk...")
        save_faiss_index(vectorstore, index_dir=index_dir)
        print("💾 Saved FAISS index.")
    else:
        print("✅ Loaded existing FAISS index from disk.")

    # ---------- Quick Retrieval Test ----------
    test_query = "What is this document about?"
    _ = retrieval_test(vectorstore, test_query, k=4)

    # ---------- Step 4: RAG Answer Test ----------
    print("\n🧠 RAG Answer Test\n")
    rag_question = "Summarize the key terms and conditions of the policy."
    rag_out = answer_with_rag(vectorstore, rag_question, k=5)

    print("Answer:\n")
    print(rag_out["answer"])
    print("\nCitations (from retrieved chunks):")
    for c in rag_out["citations"]:
        print("-", c)