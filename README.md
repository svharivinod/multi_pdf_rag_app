# 📄 CiteRAG — Document Q&A with Citations

CiteRAG is a **Multi-PDF Retrieval-Augmented Generation (RAG) application** that allows users to upload multiple PDF documents, ask natural-language questions, and receive **grounded answers with clear page-level citations**.

The system ensures that answers are generated **only from the uploaded documents**, making it suitable for use cases that require **accuracy, traceability, and trust**.

<img width="1860" height="825" alt="CiteRAG" src="https://github.com/user-attachments/assets/b51c3dba-4cb0-4a53-b067-833eebe447f5" />


---

## 🚀 Key Features

- 📚 Upload **multiple PDFs simultaneously**
- 🔍 Semantic search using **vector embeddings**
- 🧠 LLM answers grounded strictly in document context
- 📌 **Page-level citations** for every answer
- ⚡ FAISS vector index for fast retrieval
- 💾 Persistent index (no re-embedding on every run)
- 🖥️ Simple and clean **Streamlit UI**

---

## 🧠 Why Retrieval-Augmented Generation (RAG)?

Large Language Models are powerful but can hallucinate.

CiteRAG uses a **RAG pipeline**, where:
1. Relevant content is first retrieved from documents
2. The LLM then answers **only using that retrieved content**

This ensures:
- No fabricated answers
- Full traceability
- Verifiable outputs

---

## 🏗️ System Architecture
User Question
│
▼
Streamlit UI
│
▼
FAISS Vector Store (Embeddings)
│
▼
Top-K Relevant Chunks
│
▼
LLM (ChatOpenAI)
│
▼
Answer + Page-Level Citations

Flowchart TD
    
    U[User] -->|Upload PDFs| UI[Streamlit UI]

    UI -->|PDF Files| L[PDF Loader<br/>(PyPDFLoader)]
    L --> M[Page-wise Documents<br/>+ Metadata]

    M --> C[Text Chunking<br/>(RecursiveCharacterTextSplitter)]
    C --> CH[Text Chunks<br/>+ chunk_id]

    CH --> E[Embeddings<br/>(OpenAI text-embedding-3-small)]
    E --> V[FAISS Vector Store]

    V -->|Top-K Similarity Search| R[Relevant Chunks]

    UI -->|User Question| QE[Query Embedding]
    QE --> V

    R --> P[Prompt Assembly<br/>(Context + Question)]
    P --> LLM[ChatOpenAI]

    LLM --> A[Answer]
    A --> CIT[Citations<br/>(PDF + Page No.)]

    CIT --> UI



## 🧩 Core RAG Pipeline (`rag_pipeline.py`)

The project follows a clean, modular RAG pipeline:

<img width="4439" height="2574" alt="Untitled-2025-12-24-2147_1 excalidraw" src="https://github.com/user-attachments/assets/04fb7788-4512-4482-84a6-1f56a6d458aa" />


### 1️⃣ PDF Loading
- Each PDF is loaded page-by-page
- Metadata added:
  - `source` (file name)
  - `page` (1-based)
  - `doc_id` (unique per document)

### 2️⃣ Text Chunking
- Pages are split into overlapping chunks
- Metadata is preserved across chunks
- Each chunk gets a `chunk_id`

### 3️⃣ Embeddings
- Chunks are converted into embeddings using:
  - `text-embedding-3-small`

### 4️⃣ Vector Storage
- Embeddings are stored in **FAISS**
- Index is saved to disk for reuse

### 5️⃣ Retrieval
- Top-K relevant chunks are retrieved based on semantic similarity

### 6️⃣ Answer Generation
- LLM answers **only from retrieved chunks**
- Citations are generated from chunk metadata

---

## 🖥️ Application UI

### 🔹 Upload PDFs
Users can upload one or more PDF files directly from the UI.

### 🔹 Ask Questions
Users can ask free-form questions such as:
- *“What are these documents about?”*
- *“What are the key terms and conditions?”*
- *“What is the policy maturity date?”*

### 🔹 Grounded Answers with Citations
Every answer includes:
- Clear explanation
- Exact PDF name(s)
- Page number(s)

---

## 📸 Screenshots

> 📌 Add these screenshots to a `/screenshots` folder in your repo

- `screenshots/home.png`
- `screenshots/answer_with_citations.png`
- `screenshots/multi_pdf_upload.png`

(You can add the images later and update the links.)

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **OpenAI API**
- **FAISS**
- **Streamlit**
- **PyPDF**
- **dotenv**

---

## 📂 Project Structure
multi_pdf_rag_app/
│
├── app.py # Streamlit UI
├── rag_pipeline.py # Core RAG pipeline
├── prompts.py # Prompt templates
├── requirements.txt
├── README.md
├── sample_pdfs/
├── indexes/ # FAISS index (ignored in git)
├── .env.example
└── .gitignore

## 🔐 Environment Setup

Create a `.env` file in the project root:
OPENAI_API_KEY=your_api_key_here

Install dependencies: pip install -r requirements.txt

Run the app: streamlit run app.py

## ⚠️ Limitations

Works best with text-based PDFs (not scanned images)

No document-specific filtering yet (future enhancement)

Retrieval quality depends on chunking strategy

## 🚀 Future Improvements

Document-level filters

Highlighted evidence snippets

Confidence scoring per answer

Streaming responses

Authentication & deployment hardening

## 🎯 Learning Outcomes

This project demonstrates:

End-to-end RAG system design

Correct use of embeddings and vector stores

Prompt grounding and hallucination control

Practical LLM application architecture

Production-ready ML engineering practices

## 🙌 Acknowledgements

Built as part of a hands-on learning exercise to deeply understand Retrieval-Augmented Generation using Langchain.

## ⭐ If you like this project

Feel free to ⭐ the repository or fork it for experimentation.







