import hashlib
import re
import time
from pathlib import Path

import chromadb
import pymupdf
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag import generate_answer


PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 100
RELEVANCE_THRESHOLD = 0.35
MAX_HISTORY_TURNS = 4

st.set_page_config(page_title="ManualMind", page_icon="🔧")


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def load_database():
    return chromadb.PersistentClient(
        path=str(PROJECT_FOLDER / "chroma_db")
    )


def extract_chunks(pdf_bytes):
    chunks = []
    empty_pages = []

    model = load_model()
    tokenizer = model.tokenizer

    # Reserve room for tokens the model adds automatically.
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_budget = min(
        CHUNK_SIZE,
        model.max_seq_length - special_tokens
    )

    if not 0 <= OVERLAP < token_budget:
        raise ValueError("OVERLAP must be smaller than the token budget.")

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Please upload a PDF without a password.")

        page_count = len(document)

        for page_index, page in enumerate(document):
            text = re.sub(
                r"\s+", " ", page.get_text("text", sort=True)
            ).strip()

            if not text:
                empty_pages.append(page_index + 1)
                continue

            # Find each token's position in the original page text.
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False
            )

            offsets = encoded["offset_mapping"]
            start = 0
            chunk_number = 1

            while start < len(offsets):
                end = min(start + token_budget, len(offsets))

                # Extract original text and verify its actual token count.
                while end > start:
                    char_start = offsets[start][0]
                    char_end = offsets[end - 1][1]
                    chunk_text = text[char_start:char_end].strip()

                    actual_tokens = tokenizer(
                        chunk_text,
                        add_special_tokens=True,
                        truncation=False
                    )["input_ids"]

                    if len(actual_tokens) <= model.max_seq_length:
                        break

                    end -= 1

                if end == start:
                    raise ValueError(
                        f"Could not split text on PDF page {page_index + 1}."
                    )

                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_index + 1,
                        "chunk_number": chunk_number
                    })
                    chunk_number += 1

                if end == len(offsets):
                    break

                start = max(start + 1, end - OVERLAP)

    return chunks, page_count, empty_pages


def process_manual(pdf_bytes, manual_key):
    processing_start = time.perf_counter()
    chunks, page_count, empty_pages = extract_chunks(pdf_bytes)

    if not chunks:
        raise ValueError(
            "No selectable text was found. "
            "Please try a text-based PDF. OCR is not included yet."
        )

    model = load_model()
    texts = [chunk["text"] for chunk in chunks]

    token_lists = model.tokenizer(
        texts, truncation=False
    )["input_ids"]

    if any(len(tokens) > model.max_seq_length for tokens in token_lists):
        raise ValueError(
            "Some chunks exceed the embedding model's input limit. "
            "Reduce CHUNK_SIZE and OVERLAP, then process again."
        )

    collection = load_database().get_or_create_collection(
        name=f"ui_{manual_key}",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}
    )

    # Reuse a complete index for this exact PDF and settings.
    indexed_new = collection.count() != len(chunks)

    if indexed_new:
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        for start in range(0, len(chunks), 100):
            batch = chunks[start:start + 100]

            collection.upsert(
                ids=[
                    f"chunk_{start + index}"
                    for index in range(len(batch))
                ],
                documents=[chunk["text"] for chunk in batch],
                embeddings=embeddings[
                    start:start + len(batch)
                ].tolist(),
                metadatas=[
                    {
                        "page": chunk["page"],
                        "chunk_number": chunk["chunk_number"]
                    }
                    for chunk in batch
                ]
            )

    return {
        "collection_name": collection.name,
        "pages": page_count,
        "chunks": len(chunks),
        "empty_pages": empty_pages,
        "processing_time": time.perf_counter() - processing_start,
        "indexed_new": indexed_new
    }

def render_pdf_page(pdf_bytes, page_number):
    """Convert one PDF page into an image for display."""

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page = document.load_page(page_number - 1)

        image = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False
        )

        return image.tobytes("png")


def show_evidence(message, pdf_bytes, message_index):
    """Display retrieved passages and an optional PDF page preview."""

    relevance_scores = message.get("relevance_scores", [])

    with st.expander("View retrieved passages"):
        for index, source in enumerate(message["sources"]):
            if index < len(relevance_scores):
                st.caption(
                    f"S{index + 1} — retrieval relevance: "
                    f"{relevance_scores[index] * 100:.0f}%"
                )
            st.text(source)

    pages = message.get("pages", [])

    if pages:
        with st.expander("View original PDF page"):
            selected_page = st.selectbox(
                "Choose a source page",
                options=pages,
                format_func=lambda page: f"PDF page {page}",
                key=f"source_page_{message_index}"
            )

            if st.checkbox(
                "Show page preview",
                key=f"show_preview_{message_index}"
            ):
                page_image = render_pdf_page(
                    pdf_bytes,
                    selected_page
                )

                st.image(
                    page_image,
                    caption=f"Original manual — PDF page {selected_page}"
                )

                
def build_conversation_context(messages):
    """Return a short recent conversation history for follow-up questions."""
    turns = []

    for message in messages[-(MAX_HISTORY_TURNS * 2):]:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            turns.append(f"User: {content}")
        elif role == "assistant":
            turns.append(f"ManualMind: {content}")

    return "\n".join(turns)


def answer_question(
    question,
    collection_name,
    manual_name,
    conversation_history=""
):
    model = load_model()

    tokens = model.tokenizer(
        question, truncation=False
    )["input_ids"]

    if len(tokens) > model.max_seq_length:
        raise ValueError("Please shorten your question.")

    collection = load_database().get_collection(
        name=collection_name,
        embedding_function=None
    )

    embedding = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=min(4, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    # Chroma's cosine distance is converted to a 0-1 similarity estimate.
    relevance_scores = [
        max(0.0, min(1.0, 1.0 - float(distance)))
        for distance in distances
    ]
    top_relevance = relevance_scores[0] if relevance_scores else 0.0

    sources = []
    for index, passage in enumerate(documents, start=1):
        page = metadatas[index - 1]["page"]
        sources.append(
            f"[S{index}] Manual: {manual_name}; PDF page: {page}\n"
            f"{passage}"
        )

    pages = sorted({
        int(metadata["page"])
        for metadata in metadatas
    })

    # Prevent unsupported answers when retrieval is too weak.
    if top_relevance < RELEVANCE_THRESHOLD:
        answer = (
            "I could not find sufficiently relevant information in the "
            "uploaded manual to answer this question reliably. "
            "Try rephrasing the question or asking about a topic covered "
            "in the manual."
        )
        return answer, sources, pages, relevance_scores, top_relevance

    generation_question = question
    if conversation_history:
        generation_question = (
            "Use the recent conversation below to understand references "
            "such as 'it', 'this', or 'that'. Answer the current question "
            "using the retrieved manual passages as the factual source.\n\n"
            f"Recent conversation:\n{conversation_history}\n\n"
            f"Current question:\n{question}"
        )

    context = "\n\n".join(sources)
    answer = generate_answer(generation_question, context)

    return answer, sources, pages, relevance_scores, top_relevance


st.title("🔧 ManualMind")
st.write("Upload an equipment manual and ask questions about it.")
st.caption(
    "Ask questions naturally. Recent conversation context is used "
    "to understand follow-up questions while answers remain grounded "
    "in the uploaded manual."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your manual", type=["pdf"]
)

if uploaded_file is None:
    st.session_state.clear()
    st.info("Upload a PDF using the panel on the left.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()

# The key includes processing settings to keep indexes separate.
settings = f"{MODEL_NAME}|{CHUNK_SIZE}|{OVERLAP}|extract_tokens_v2"
manual_key = hashlib.sha256(
    pdf_bytes + settings.encode()
).hexdigest()[:24]

# Clear the old conversation when the uploaded manual changes.
if st.session_state.get("manual_key") != manual_key:
    st.session_state.manual_key = manual_key
    st.session_state.messages = []
    st.session_state.ready = False
    st.session_state.info = None
    st.session_state.quick_question = None

if st.sidebar.button("Process manual", type="primary"):
    st.session_state.ready = False

    try:
        with st.spinner("Reading and indexing your manual..."):
            info = process_manual(pdf_bytes, manual_key)

        st.session_state.info = info
        st.session_state.ready = True
        st.session_state.messages = []

    except Exception as error:
        st.error(f"Could not process the manual: {error}")

if not st.session_state.get("ready"):
    st.info("Click Process manual to prepare this PDF for questions.")
    st.stop()

info = st.session_state.info

st.sidebar.success("Manual ready")
st.sidebar.subheader("Manual statistics")
st.sidebar.write(f"Pages: {info['pages']}")
st.sidebar.write(f"Chunks: {info['chunks']}")
st.sidebar.write(f"Embedding model: {MODEL_NAME}")
st.sidebar.write(f"Processing time: {info['processing_time']:.2f} s")
st.sidebar.write(
    "Index status: " + ("Created" if info["indexed_new"] else "Reused")
)

if info["empty_pages"]:
    st.sidebar.warning(
        "No text was extracted from PDF pages: "
        + ", ".join(map(str, info["empty_pages"]))
    )

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []

st.subheader("Quick Actions")
quick_columns = st.columns(5)
quick_actions = [
    ("Summarize", "Give a concise summary of the most important information covered in the manual."),
    ("Explain", "Explain the main technical concepts covered in this manual in simple terms."),
    ("Specifications", "Find and list the important equipment specifications and technical values mentioned in the manual."),
    ("Maintenance", "Find the important maintenance procedures, intervals, and recommendations mentioned in the manual."),
    ("Safety", "Find the important safety precautions, warnings, and operating instructions mentioned in the manual.")
]

for column, (label, prompt) in zip(quick_columns, quick_actions):
    with column:
        if st.button(label, use_container_width=True):
            st.session_state.quick_question = prompt

for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("role") == "assistant":
            if message.get("top_relevance") is not None:
                st.caption(
                    f"Retrieval relevance: "
                    f"{message['top_relevance'] * 100:.0f}%"
                )

            if message.get("computing_time") is not None:
                st.caption(
                    f"⏱ Computing time: {message['computing_time']:.2f} seconds"
                )

            feedback = message.get("feedback")
            if feedback:
                st.caption(f"Feedback recorded: {feedback}")
            else:
                st.caption("Was this answer useful?")
                feedback_columns = st.columns(2)
                with feedback_columns[0]:
                    if st.button("👍 Yes", key=f"feedback_yes_{message_index}"):
                        message["feedback"] = "Helpful"
                        st.rerun()
                with feedback_columns[1]:
                    if st.button("👎 No", key=f"feedback_no_{message_index}"):
                        message["feedback"] = "Not helpful"
                        st.rerun()

        if message.get("sources"):
            show_evidence(message, pdf_bytes, message_index)

chat_question = st.chat_input("Ask a question about your manual")
quick_question = st.session_state.pop("quick_question", None)
question = (
    chat_question
    if chat_question and chat_question.strip()
    else quick_question
)

if question and question.strip():
    question = question.strip()

    previous_messages = list(st.session_state.messages)
    conversation_history = build_conversation_context(previous_messages)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            start_time = time.perf_counter()

            with st.spinner(
                "🔎 Searching the manual and generating your answer..."
            ):
                (
                    answer,
                    sources,
                    pages,
                    relevance_scores,
                    top_relevance
                ) = answer_question(
                    question,
                    info["collection_name"],
                    uploaded_file.name,
                    conversation_history
                )

            computing_time = time.perf_counter() - start_time

            message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "pages": pages,
                "relevance_scores": relevance_scores,
                "top_relevance": top_relevance,
                "computing_time": computing_time,
                "feedback": None
            }

            st.session_state.messages.append(message)
            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Open Ollama from the Windows Start menu, then try again.")

        except requests.exceptions.Timeout:
            st.error("The model took too long. Check that Ollama is responding.")

        except requests.exceptions.HTTPError as error:
            st.error(f"Ollama error: {error.response.text}")

        except Exception as error:
            st.error(f"Could not answer the question: {error}")
