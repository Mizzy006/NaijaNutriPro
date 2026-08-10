"""
knowledge_builder.py — Ingests knowledge base documents into ChromaDB.

This module handles:
1. Loading all .md files from rag/knowledge_base/
2. Chunking documents into ~500-char segments with overlap
3. Embedding via sentence-transformers (all-MiniLM-L6-v2)
4. Storing in ChromaDB PersistentClient

Usage:
    from rag.knowledge_builder import build_knowledge_base
    build_knowledge_base()  # One-time setup, idempotent
"""

import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# --- Paths (relative to project root) ---
RAG_DIR = Path(__file__).parent
KB_DIR = RAG_DIR / "knowledge_base"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

# --- Constants ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "nigerian_nutrition_kb"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` characters.
    Tries to break at paragraph or sentence boundaries when possible.
    """
    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk_size, finalize the current chunk
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of the current chunk
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _load_documents() -> list[dict]:
    """
    Load all .md files from the knowledge base directory.
    Returns list of dicts with keys: text, source, section_title
    """
    documents = []

    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")

    for md_file in sorted(KB_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        source_name = md_file.stem  # e.g., "health_conditions"

        # Extract section titles (## headings) for metadata
        current_section = source_name
        for line in content.split("\n"):
            if line.startswith("## "):
                current_section = line.lstrip("# ").strip()

        # Chunk the full document
        chunks = _chunk_text(content)
        for i, chunk in enumerate(chunks):
            # Try to extract the most relevant section title for this chunk
            section = source_name
            for line in chunk.split("\n"):
                if line.startswith("## "):
                    section = line.lstrip("# ").strip()
                    break
                elif line.startswith("# "):
                    section = line.lstrip("# ").strip()
                    break

            documents.append({
                "text": chunk,
                "source": source_name,
                "section": section,
                "chunk_id": f"{source_name}_chunk_{i}",
            })

    return documents


def build_knowledge_base(force_rebuild: bool = False) -> chromadb.Collection:
    """
    Build or load the ChromaDB knowledge base.
    
    Args:
        force_rebuild: If True, deletes and rebuilds the entire collection.
    
    Returns:
        The ChromaDB collection ready for querying.
    """
    # Ensure chroma directory exists
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Check if collection already exists and has data
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        if collection.count() > 0 and not force_rebuild:
            print(f"[RAG] Knowledge base already built ({collection.count()} chunks). Skipping.")
            return collection
        else:
            # Rebuild: delete and recreate
            client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet, that's fine

    print("[RAG] Building knowledge base...")

    # Load and chunk documents
    documents = _load_documents()
    if not documents:
        raise ValueError("No documents found in knowledge base directory.")

    print(f"[RAG] Loaded {len(documents)} chunks from {KB_DIR}")

    # Initialize embedding model
    print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    print(f"[RAG] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Create collection and add data
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Nigerian nutrition knowledge base for RAG"}
    )

    collection.add(
        ids=[doc["chunk_id"] for doc in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "source": doc["source"],
            "section": doc["section"],
        } for doc in documents],
    )

    print(f"[RAG] Knowledge base built successfully! {collection.count()} chunks indexed.")
    return collection


# --- Standalone execution for initial setup ---
if __name__ == "__main__":
    build_knowledge_base(force_rebuild=True)
