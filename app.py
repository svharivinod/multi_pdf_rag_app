import os
import tempfile
import streamlit as st

from rag_pipeline import (
    load_pdfs_with_metadata,
    chunk_documents,
    build_faiss_index,
    load_faiss_index,
    save_faiss_index,
    answer_with_rag,
)

st.set_page_config(page_title="Multi-PDF RAG Chat", layout="wide")

st.markdown(
    """
    <style>
    /* Main page text */
    html, body, [class*="css"]  {
        font-size: 20px;
    }

    /* Title */
    h1 {
        font-size: 2.2rem !important;
    }

    /* Section headers */
    h2, h3 {
        font-size: 1.6rem !important;
    }

    /* Captions / helper text */
    .stCaption {
        font-size: 1.1rem !important;
    }

    /* Input labels */
    label {
        font-size: 1.1rem !important;
    }

    /* Text input */
    .stTextInput input {
        font-size: 1.1rem !important;
        padding: 10px;
    }

    /* Buttons */
    .stButton button {
        font-size: 1.1rem !important;
        padding: 0.6em 1.2em;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] {
        font-size: 1.05rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.title("📄 CiteRAG — Document Q&A with Citations")
st.write("Upload multiple PDFs and ask questions. Answers are grounded in the documents, with clear page-level citations.")

# -----------------------------
# Session state
# -----------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []

# -----------------------------
# Sidebar: Upload PDFs
# -----------------------------
st.sidebar.header("📂 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} PDF(s) uploaded")

    if st.sidebar.button("🚀 Build / Rebuild Index"):
        with st.spinner("Processing PDFs and building index..."):
            temp_dir = tempfile.mkdtemp()
            pdf_paths = []

            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.read())
                pdf_paths.append(file_path)

            # Step 1: Load PDFs
            docs = load_pdfs_with_metadata(pdf_paths)

            # Step 2: Chunk
            chunks = chunk_documents(docs)

            # Step 3: Build FAISS
            vectorstore = build_faiss_index(chunks)

            # Save for reuse
            save_faiss_index(vectorstore)

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_names = [f.name for f in uploaded_files]

        st.sidebar.success("✅ Index built successfully")

# -----------------------------
# Load existing index if available
# -----------------------------
if st.session_state.vectorstore is None:
    existing_vs = load_faiss_index()
    if existing_vs:
        st.session_state.vectorstore = existing_vs
        st.sidebar.info("ℹ️ Loaded existing FAISS index from disk")

# -----------------------------
# Main Chat Interface
# -----------------------------
st.divider()
st.header("💬 Ask a Question")

question = st.text_input(
    "Type your question about the uploaded PDFs:",
    placeholder="e.g. What are the key terms and conditions?",
)

if st.button("Ask") and question:
    if st.session_state.vectorstore is None:
        st.error("Please upload PDFs and build the index first.")
    else:
        with st.spinner("Thinking..."):
            result = answer_with_rag(
                st.session_state.vectorstore,
                question,
                k=5,
            )

        st.subheader("✅ Answer")
        st.write(result["answer"])

        st.subheader("📌 Citations")
        for c in result["citations"]:
            st.write(f"- {c}")
