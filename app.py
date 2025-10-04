import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_utils import (
    get_chroma,
    get_collection,
    retrieve,
    build_context_snippets,
    answer_with_context,
    ingest_files_streaming,
)

load_dotenv()
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY. Add it to .env or Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="CA-GPT", page_icon="🏫")
st.title("🏫 CA-GPT")

st.subheader("Learn about Cary Academy without digging through documents, just through CA-GPT!")

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    model = st.selectbox("Model", ["gpt-5-mini", "gpt-5"], index=0)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.")
    st.markdown("---")
    st.markdown("### RAG")
    embedding_model = st.selectbox(
        "Embedding model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="Small = cheaper; Large = highest quality",
    )
    persist_dir = st.text_input("Vector store dir", value=".chroma")

# Init vector store once
if "chroma_ready" not in st.session_state:
    st.session_state.chroma = get_chroma(persist_dir=persist_dir)
    st.session_state.col = get_collection(
        st.session_state.chroma,
        name="docs",
        openai_api_key=api_key,
        embedding_model=embedding_model,
    )
    st.session_state.chroma_ready = True

# Upload & index
st.subheader("Knowledge Base (RAG)")
uploads = st.file_uploader(
    "Add PDFs/TXT/MD", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if uploads and st.button("Index uploaded files"):
    # Save to a temp folder and ingest
    os.makedirs("tmp_uploads", exist_ok=True)
    saved = []
    for uf in uploads:
        path = os.path.join("tmp_uploads", uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(path)

    n_chunks = ingest_files_streaming(
        saved,
        st.session_state.col,
        chunk_size=900,
        chunk_overlap=140,
        batch_size=64,
        precompute_embeddings=True,   # batch to OpenAI (faster)
        openai_client=client,
        embedding_model=embedding_model,  # ← use the same model as the collection
        max_file_mb=50,
        progress_cb=lambda n: st.write(f"Indexed chunks: {n}"),
    )
    st.success(f"Ingested {len(saved)} file(s), {n_chunks} chunk(s).")

st.caption("Tip: index once, then your documents persist in `.chroma/` between app restarts.")

# Message history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Render prior messages (skip system)
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Ask something…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    # --- RAG step: retrieve & answer ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking with your documents…"):
            top = retrieve(user_msg, st.session_state.col, k=5)
            ctx = build_context_snippets(top, max_chars=3000)
            answer = answer_with_context(client, model, system_prompt, user_msg, ctx)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()