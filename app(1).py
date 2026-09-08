import os
import tempfile

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="JHULLANS PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Upload a PDF, build a local FAISS knowledge base, and ask questions using Groq.")


# -----------------------------
# Configuration
# -----------------------------
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    # Streamlit Cloud: add GROQ_API_KEY in App Settings -> Secrets.
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def extract_pdf_text(uploaded_file):
    """Extract text page-by-page from the uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping character-based chunks."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(pages):
    """Create chunks while preserving source page numbers."""
    all_chunks = []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])

        for chunk in chunks:
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_data["page"],
                }
            )

    return all_chunks


def build_faiss_index(chunks, embedding_model):
    texts = [item["text"] for item in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def retrieve(question, index, chunks, embedding_model, top_k=TOP_K):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            }
        )

    return results


def generate_answer(client, question, retrieved_chunks):
    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the supplied PDF context.
If the answer is not present in the context, clearly say:
"I could not find that information in the uploaded PDF."

Do not invent facts.
Keep the answer clear and easy to understand.
When useful, mention the PDF page number from the provided sources.
"""

    user_prompt = f"""PDF CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# Session state
# -----------------------------
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}` characters")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP}` characters")
    st.write(f"**Retrieved chunks:** `{TOP_K}`")

    st.divider()

    if st.session_state.index is not None:
        st.success("FAISS index is ready.")
        st.metric("Indexed chunks", len(st.session_state.chunks))
    else:
        st.info("Upload a PDF to build the index.")


# -----------------------------
# API key check
# -----------------------------
client = get_groq_client()

if client is None:
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets "
        "or set it as an environment variable when running locally."
    )
    st.code('GROQ_API_KEY = "your_groq_api_key"')
    st.stop()


# -----------------------------
# PDF upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="For best results, upload a text-based PDF. Scanned/image-only PDFs need OCR.",
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Reading PDF, creating chunks, embeddings, and FAISS index..."):
            try:
                pages = extract_pdf_text(uploaded_file)

                if not pages:
                    st.error(
                        "No selectable text was found. This may be a scanned PDF. "
                        "Use an OCR-enabled PDF or add OCR support."
                    )
                    st.stop()

                chunks = build_chunks(pages)

                if not chunks:
                    st.error("No usable text chunks were created from this PDF.")
                    st.stop()

                embedding_model = load_embedding_model()
                index = build_faiss_index(chunks, embedding_model)

                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully: {uploaded_file.name} "
                    f"({len(pages)} pages, {len(chunks)} chunks)."
                )

            except Exception as e:
                st.error(f"Could not process the PDF: {e}")

# -----------------------------
# Chat area
# -----------------------------
if st.session_state.index is not None:
    st.subheader("💬 Ask questions about your PDF")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the uploaded PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF and generating an answer..."):
                try:
                    embedding_model = load_embedding_model()

                    retrieved = retrieve(
                        question,
                        st.session_state.index,
                        st.session_state.chunks,
                        embedding_model,
                    )

                    answer = generate_answer(
                        client,
                        question,
                        retrieved,
                    )

                    st.markdown(answer)

                    with st.expander("🔎 Retrieved sources"):
                        for i, item in enumerate(retrieved, start=1):
                            st.markdown(
                                f"**Source {i} — PDF page {item['page']} "
                                f"(similarity: {item['score']:.3f})**"
                            )
                            st.write(item["text"])

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    st.error(f"Error while generating the answer: {e}")
else:
    st.info("👆 Upload a PDF to start.")

st.divider()
st.caption("RAG pipeline: PDF → text extraction → chunks → embeddings → FAISS → retrieval → Groq LLM")
