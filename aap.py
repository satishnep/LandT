
import os
import re
import hashlib

import fitz
import numpy as np
import streamlit as st
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="College Notes AI Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_secret(name, default=""):
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def split_into_chunks(text):
    words = re.sub(r"\s+", " ", text).strip().split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(words), step):
        part = words[start:start + CHUNK_SIZE]
        if part:
            chunks.append(" ".join(part))
        if start + CHUNK_SIZE >= len(words):
            break

    return chunks


def extract_pdf(pdf_bytes, filename):
    documents = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            for chunk_number, chunk in enumerate(
                split_into_chunks(text), start=1
            ):
                documents.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_number,
                    "chunk": chunk_number,
                })

    return documents


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9_+-]+\b", text.lower())


def build_index(documents, model):
    texts = [doc["text"] for doc in documents]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    bm25 = BM25Okapi([tokenize(text) for text in texts])
    return vectors, bm25


def retrieve(question, documents, vectors, bm25, model, top_k=5):
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")

    # Dense semantic similarity
    dense_scores = (vectors @ query_vector + 1.0) / 2.0

    # BM25 keyword relevance
    keyword_scores = np.asarray(
        bm25.get_scores(tokenize(question)), dtype="float32"
    )

    if keyword_scores.max() > 0:
        keyword_scores /= keyword_scores.max()

    # Hybrid retrieval: 65% semantic + 35% keyword relevance
    scores = 0.65 * dense_scores + 0.35 * keyword_scores
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, index in enumerate(best_indices, start=1):
        item = dict(documents[int(index)])
        item["score"] = float(scores[int(index)])
        item["source_id"] = f"S{rank}"
        results.append(item)

    return results


def generate_answer(question, sources, api_key, model_name):
    context = "\n\n".join(
        f"[{item['source_id']}] File: {item['source']} | "
        f"Page: {item['page']}\n{item['text']}"
        for item in sources
    )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a college notes and research assistant. "
                    "Answer using only the provided context. Do not invent "
                    "facts. If the context is insufficient, say so. Cite "
                    "claims using source labels such as [S1] and [S2]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Give a clear, student-friendly answer with citations."
                ),
            },
        ],
    )

    return response.choices[0].message.content or "No answer returned."


def reset_store():
    st.session_state.documents = []
    st.session_state.vectors = None
    st.session_state.bm25 = None
    st.session_state.file_hashes = set()


if "documents" not in st.session_state:
    reset_store()

st.title("📚 AI-Powered College Notes & Research Assistant")
st.write(
    "Upload notes or research papers, ask questions, and inspect the "
    "source passages used to answer them."
)

with st.sidebar:
    st.header("Settings")

    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        help="Optional if configured in environment variables or Streamlit Secrets.",
    )
    api_key = api_key_input.strip() or get_secret("GROQ_API_KEY")
    llm_model = get_secret("GROQ_MODEL", DEFAULT_LLM_MODEL)

    top_k = st.slider(
        "Number of retrieved passages",
        min_value=3,
        max_value=8,
        value=5,
    )

    st.metric("Indexed text chunks", len(st.session_state.documents))

    if st.button("Clear knowledge base"):
        reset_store()
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload college notes or research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
    documents = list(st.session_state.documents)
    hashes = set(st.session_state.file_hashes)
    errors = []

    progress = st.progress(0)

    for i, uploaded in enumerate(uploaded_files):
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()

        if digest in hashes:
            progress.progress((i + 1) / len(uploaded_files))
            continue

        try:
            extracted = extract_pdf(data, uploaded.name)

            if extracted:
                documents.extend(extracted)
                hashes.add(digest)
            else:
                errors.append(
                    f"{uploaded.name}: no selectable text found. "
                    "Scanned PDFs require OCR."
                )
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")

        progress.progress((i + 1) / len(uploaded_files))

    if documents:
        with st.spinner("Creating embeddings and search index..."):
            model = load_embedding_model()
            vectors, bm25 = build_index(documents, model)

        st.session_state.documents = documents
        st.session_state.vectors = vectors
        st.session_state.bm25 = bm25
        st.session_state.file_hashes = hashes

        st.success(
            f"Knowledge base ready: {len(documents)} text chunks indexed."
        )

    for error in errors:
        st.warning(error)

if st.session_state.documents:
    st.divider()
    st.subheader("Ask a question")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Explain overfitting and its prevention methods.",
    )

    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching your documents..."):
                model = load_embedding_model()
                sources = retrieve(
                    question.strip(),
                    st.session_state.documents,
                    st.session_state.vectors,
                    st.session_state.bm25,
                    model,
                    top_k,
                )

            st.subheader("Answer")

            if api_key:
                try:
                    with st.spinner("Generating answer..."):
                        answer = generate_answer(
                            question.strip(), sources, api_key, llm_model
                        )
                    st.markdown(answer)
                except Exception as exc:
                    st.error(f"LLM request failed: {exc}")
                    st.info(
                        "Check your API key and model name. "
                        "The retrieved passages are still available below."
                    )
            else:
                st.info(
                    "Add a Groq API key in the sidebar to generate an answer. "
                    "The relevant passages are displayed below."
                )

            st.subheader("Retrieved source passages")

            for item in sources:
                title = (
                    f"[{item['source_id']}] {item['source']} | "
                    f"Page {item['page']} | Score {item['score']:.3f}"
                )
                with st.expander(title):
                    st.write(item["text"])

else:
    st.info("Upload PDF files and click Process PDFs to begin.")

with st.expander("About this project and research evaluation"):
    st.markdown("""
    **Pipeline:** PDF extraction → overlapping chunks → embeddings →
    hybrid retrieval → LLM answer generation → source references.

    **Research experiments:**
    - Compare BM25, dense retrieval, and hybrid retrieval.
    - Measure Recall@k, Precision@k, and Mean Reciprocal Rank.
    - Evaluate answer correctness, faithfulness, and citation accuracy.
    - Measure response latency and API cost.

    **Limitations:** Text-based PDFs work best. Scanned PDFs require OCR.
    The knowledge base is session-only and is not permanently stored.
    Always verify important answers against the original PDF.
    """)
     