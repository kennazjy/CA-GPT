# rag_utils.py
"""
rag_utils.py — lightweight RAG utilities for Streamlit + OpenAI + Chroma
"""

from __future__ import annotations
import os
import re
import uuid
import time
from typing import Iterable, List, Dict, Tuple, Optional

# ----------------------------
# Parsing & loaders
# ----------------------------

def _normalize_ws(s: str) -> str:
    """Collapse whitespace to keep chunks compact."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def iter_pdf_text(path: str) -> Iterable[str]:
    """
    Yield text per page. Tries PyMuPDF (fast) first, then pypdf.
    Never loads the entire PDF into memory.
    """
    try:
        import fitz  # type: ignore
        with fitz.open(path) as doc:
            for page in doc:
                yield page.get_text("text") or ""
        return
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(path)
        for p in reader.pages:
            try:
                yield p.extract_text() or ""
            except Exception:
                yield ""
    except Exception:
        return


def load_text_file(path: str) -> Iterable[str]:
    """Yield the entire text file as one string (still streamed as an iterator)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        yield f.read()


def fetch_url_text(url: str) -> Iterable[str]:
    """Yield cleaned visible text from a web page."""
    try:
        import requests
        from bs4 import BeautifulSoup
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        yield text
    except Exception:
        return


def iter_any(path_or_url: str) -> Iterable[str]:
    """Dispatch to the appropriate loader and yield text blocks."""
    lower = path_or_url.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        yield from fetch_url_text(path_or_url)
    elif lower.endswith(".pdf"):
        yield from iter_pdf_text(path_or_url)
    elif lower.endswith((".txt", ".md")):
        yield from load_text_file(path_or_url)
    else:
        yield from load_text_file(path_or_url)

# ----------------------------
# Chunking
# ----------------------------

def iter_chunks_from_pages(
    pages: Iterable[str],
    chunk_size: int = 900,
    chunk_overlap: int = 140,
) -> Iterable[str]:
    """Stream chunks across page boundaries without building huge lists."""
    buf = ""
    for page in pages:
        buf += " " + _normalize_ws(page)
        while len(buf) >= chunk_size + chunk_overlap:
            yield buf[:chunk_size]
            buf = buf[chunk_size - chunk_overlap:]
    if buf:
        yield buf[:chunk_size]


def estimate_chunk_count_from_path(
    path_or_url: str,
    chunk_size: int = 900,
    chunk_overlap: int = 140,
) -> int:
    """Rough estimate of chunk count."""
    return sum(
        1 for _ in iter_chunks_from_pages(iter_any(path_or_url), chunk_size, chunk_overlap)
    )

# ----------------------------
# Vector store (Chroma) setup
# ----------------------------

def get_chroma(persist_dir: str = ".chroma"):
    """Return a PersistentClient (creates the folder on first use)."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    return client


def get_collection(
    client,
    name: str = "docs",
    openai_api_key: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
):
    """
    Return (or create) a Chroma collection that uses OpenAI embeddings on the server side.
    """
    from chromadb.utils import embedding_functions
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name=embedding_model,
    )
    return client.get_or_create_collection(name=name, embedding_function=ef)

# ----------------------------
# Ingestion (streaming + batching)
# ----------------------------

def ingest_files_streaming(
    file_paths: List[str],
    collection,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 140,
    batch_size: int = 64,
    precompute_embeddings: bool = True,
    openai_client=None,                       # required if precompute_embeddings=True
    embedding_model: str = "text-embedding-3-small",
    max_file_mb: int = 50,
    progress_cb=None                          # optional: progress_cb(total_chunks_int)
) -> int:
    """
    Stream pages -> stream chunks -> write to Chroma in batches.
    """
    if precompute_embeddings and openai_client is None:
        raise ValueError("openai_client is required when precompute_embeddings=True")

    total = 0
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    def _flush():
        nonlocal ids, docs, metas, total
        if not docs:
            return
        t0 = time.time()
        if precompute_embeddings:
            emb = openai_client.embeddings.create(model=embedding_model, input=docs).data
            vectors = [e.embedding for e in emb]
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
        else:
            collection.add(ids=ids, documents=docs, metadatas=metas)
        dt = time.time() - t0
        total += len(docs)
        print(f"[flush] wrote {len(docs)} chunks in {dt:.2f}s (total={total})", flush=True)
        if progress_cb:
            try:
                progress_cb(total)
            except Exception:
                pass
        ids, docs, metas = [], [], []

    for path in file_paths:
        # Skip huge local files
        if not (path.lower().startswith("http://") or path.lower().startswith("https://")):
            try:
                sz_mb = os.path.getsize(path) / (1024 * 1024)
                if sz_mb > max_file_mb:
                    print(f"[skip] {os.path.basename(path)} {sz_mb:.1f}MB > {max_file_mb}MB", flush=True)
                    continue
            except Exception:
                pass

        print(f"[parse] {os.path.basename(path) if '://' not in path else path} …", flush=True)
        t_parse = time.time()
        cnt = 0
        for i, chunk in enumerate(iter_chunks_from_pages(iter_any(path), chunk_size, chunk_overlap)):
            ids.append(str(uuid.uuid4()))
            docs.append(chunk)
            metas.append({"source": os.path.basename(path) if os.path.exists(path) else path, "chunk": i})
            cnt += 1
            if len(docs) >= batch_size:
                _flush()
        _flush()
        print(f"[parsed] {cnt} chunks in {time.time() - t_parse:.2f}s", flush=True)

    return total

# ----------------------------
# Retrieval & answering
# ----------------------------

def retrieve(query: str, collection, k: int = 5) -> List[Tuple[str, Dict]]:
    """Return list of (document_text, metadata) pairs for top-k results."""
    res = collection.query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def build_context_snippets(
    pairs: List[Tuple[str, Dict]],
    max_chars: int = 3000
) -> str:
    """Assemble a compact context blob with lightweight inline citations."""
    blobs: List[str] = []
    used = 0
    for doc, meta in pairs:
        tag = f"[{meta.get('source','unknown')}#chunk{meta.get('chunk',0)}]"
        piece = f"{tag}\n{doc}\n"
        if used + len(piece) > max_chars:
            break
        blobs.append(piece)
        used += len(piece)
    return "\n---\n".join(blobs)


def answer_with_context(
    openai_client,
    model: str,
    system_prompt: str,
    user_query: str,
    context_blob: str
) -> str:
    """Minimal answering helper using chat.completions."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Answer ONLY using the context below. If the answer is not present, say "
                "'I don’t see this in the provided documents.'\n\n"
                f"### Context\n{context_blob}\n\n### Question\n{user_query}"
            ),
        },
    ]
    resp = openai_client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content.strip()

# ----------------------------
# Optional helpers
# ----------------------------

def clear_collection(collection) -> None:
    """Delete all docs from a Chroma collection (careful!)."""
    try:
        collection.delete(ids=collection.get()["ids"])
    except Exception:
        try:
            collection.delete(where={})
        except Exception:
            pass


__all__ = [
    "get_chroma",
    "get_collection",
    "iter_any",
    "iter_chunks_from_pages",
    "estimate_chunk_count_from_path",
    "ingest_files_streaming",
    "retrieve",
    "build_context_snippets",
    "answer_with_context",
    "clear_collection",
]
