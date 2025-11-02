import os, re, uuid, math
from typing import List, Dict, Iterable, Tuple, Optional
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ---------- PDF loaders: streaming ----------
def iter_pdf_text(path: str) -> Iterable[str]:
    """
    Yields plain text per page. Tries pymupdf first, falls back to pypdf.
    """
    try:
        import fitz  # pymupdf
        with fitz.open(path) as doc:
            for page in doc:
                yield page.get_text("text") or ""
        return
    except Exception:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        for p in reader.pages:
            try:
                yield p.extract_text() or ""
            except Exception:
                yield ""
    except Exception as e:
        # last resort: treat as empty
        yield ""

def load_text(path: str) -> Iterable[str]:
    """
    Yields one big chunk of text for .txt/.md files (still streamable).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        yield f.read()

def iter_any(path: str) -> Iterable[str]:
    lower = path.lower()
    if lower.endswith(".pdf"):
        yield from iter_pdf_text(path)
    elif lower.endswith((".txt", ".md")):
        yield from load_text(path)
    else:
        yield from load_text(path)  # naive fallback

# ---------- Chunking (generator) ----------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def iter_chunks_from_pages(pages: Iterable[str], chunk_size=1200, chunk_overlap=180) -> Iterable[str]:
    """
    Build chunks across page boundaries without loading the whole doc into memory.
    """
    buf = ""
    for page in pages:
        buf += " " + normalize_ws(page)
        while len(buf) >= chunk_size + chunk_overlap:
            yield buf[:chunk_size]
            buf = buf[chunk_size - chunk_overlap:]
    # flush remainder
    if buf:
        yield buf[:chunk_size]

# ---------- Chroma setup ----------
def get_chroma(persist_dir: str = ".chroma"):
    # auto-creates the directory on first use
    client = chromadb.PersistentClient(path=persist_dir)
    return client

def get_collection(client, name="docs", openai_api_key: str = None, embedding_model: str = "text-embedding-3-small"):
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model
    )
    return client.get_or_create_collection(name=name, embedding_function=ef)

# ---------- Streaming ingest with batching ----------
def ingest_files_streaming(
    file_paths: List[str],
    collection,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 180,
    batch_size: int = 64,
    precompute_embeddings: bool = False,
    openai_client: Optional[OpenAI] = None,
    embedding_model: str = "text-embedding-3-small",
    max_file_mb: int = 200
) -> int:
    """
    Stream pages -> stream chunks -> add to Chroma in batches.
    Optionally precompute embeddings (lets you tightly control batch size).
    """
    if precompute_embeddings and openai_client is None:
        raise ValueError("openai_client is required when precompute_embeddings=True")

    total = 0
    ids, docs, metas = [], [], []

    def flush():
        nonlocal ids, docs, metas, total
        if not docs:
            return
        if precompute_embeddings:
            # embed in the client to control batch size precisely
            emb = openai_client.embeddings.create(model=embedding_model, input=docs).data
            vectors = [e.embedding for e in emb]
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        else:
            # let Chroma's embedding function handle it
            collection.add(ids=ids, documents=docs, metadatas=metas)
        total += len(docs)
        ids, docs, metas = [], [], []

    for path in file_paths:
        # skip huge files defensively
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > max_file_mb:
                # you could show a warning in Streamlit instead
                print(f"Skipping {os.path.basename(path)} ({size_mb:.1f} MB > {max_file_mb} MB).")
                continue
        except Exception:
            pass

        page_iter = iter_any(path)
        chunk_iter = iter_chunks_from_pages(page_iter, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for i, chunk in enumerate(chunk_iter):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metas.append({"source": os.path.basename(path), "chunk": i})
            if len(docs) >= batch_size:
                flush()

        # flush end-of-file
        flush()

    return total

# ---------- Retrieval & answering ----------
def retrieve(query: str, collection, k: int = 5):
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))

def build_context_snippets(pairs: List[Tuple[str, Dict]], max_chars=3000):
    blobs, used = [], 0
    for doc, meta in pairs:
        tag = f"[{meta.get('source','unknown')}#chunk{meta.get('chunk',0)}]"
        piece = f"{tag}\n{doc}\n"
        if used + len(piece) > max_chars:
            break
        blobs.append(piece)
        used += len(piece)
    return "\n---\n".join(blobs)

def answer_with_context(client: OpenAI, model: str, system_prompt: str, user_query: str, context_blob: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
         f"Use the following context to answer. If the answer isn't in the context, say so briefly.\n\n"
         f"### Context\n{context_blob}\n\n### Question\n{user_query}"}
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()
