"""
retriever.py — Semantic search interface for the RAG pipeline.

Queries the ChromaDB vector store to retrieve relevant nutrition knowledge
for augmenting LLM prompts.

Usage:
    from rag.retriever import NutritionRetriever
    retriever = NutritionRetriever()
    results = retriever.retrieve("What foods are good for diabetes?")
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# --- Paths ---
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "nigerian_nutrition_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class NutritionRetriever:
    """
    Retrieves relevant nutrition knowledge from the ChromaDB vector store.
    
    The retriever is initialized lazily — the embedding model and ChromaDB
    connection are only created on first use.
    """

    def __init__(self):
        self._model = None
        self._collection = None

    def _ensure_initialized(self):
        """Lazy initialization of embedding model and ChromaDB connection."""
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if self._collection is None:
            if not CHROMA_DIR.exists():
                raise FileNotFoundError(
                    "ChromaDB not found. Run `python -m rag.knowledge_builder` first."
                )
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                self._collection = client.get_collection(name=COLLECTION_NAME)
            except Exception as e:
                raise RuntimeError(
                    f"Knowledge base collection '{COLLECTION_NAME}' not found. "
                    f"Run `python -m rag.knowledge_builder` to build it. Error: {e}"
                )

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        source_filter: str = None
    ) -> list[dict]:
        """
        Retrieve the top-N most relevant knowledge chunks for a query.
        
        Args:
            query: The user's question or context string.
            n_results: Number of results to return (default 3).
            source_filter: Optional filter by source document name
                          (e.g., "health_conditions", "food_pairings_and_culture").
        
        Returns:
            List of dicts, each containing:
                - text: The retrieved knowledge chunk
                - source: Source document name
                - section: Section title within the document
                - score: Relevance score (lower = more relevant in ChromaDB)
        """
        self._ensure_initialized()

        # Build query parameters
        query_embedding = self._model.encode(query).tolist()
        
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }

        # Optional: filter by source document
        if source_filter:
            query_kwargs["where"] = {"source": source_filter}

        results = self._collection.query(**query_kwargs)

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "section": results["metadatas"][0][i].get("section", "unknown"),
                    "score": results["distances"][0][i] if results["distances"] else None,
                })

        return formatted

    def retrieve_as_context(
        self,
        query: str,
        n_results: int = 3,
        source_filter: str = None
    ) -> str:
        """
        Retrieve and format knowledge chunks as a single context string
        ready for injection into an LLM prompt.
        
        Args:
            query: The user's question or context string.
            n_results: Number of results to return.
            source_filter: Optional filter by source document.
        
        Returns:
            A formatted string containing all retrieved chunks with source attribution.
        """
        results = self.retrieve(query, n_results, source_filter)

        if not results:
            return "No relevant nutrition knowledge found."

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source: {result['source']} — {result['section']}]\n"
                f"{result['text']}"
            )

        return "\n\n---\n\n".join(context_parts)

    @property
    def is_ready(self) -> bool:
        """Check if the knowledge base exists and is populated."""
        try:
            self._ensure_initialized()
            return self._collection.count() > 0
        except Exception:
            return False


# --- Standalone test ---
if __name__ == "__main__":
    retriever = NutritionRetriever()
    
    test_queries = [
        "What foods are good for someone with diabetes?",
        "Best Nigerian foods during pregnancy",
        "How to reduce sodium in Nigerian cooking",
        "What goes well with pounded yam?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = retriever.retrieve(query, n_results=2)
        for r in results:
            print(f"\n[{r['source']} — {r['section']}] (score: {r['score']:.4f})")
            print(r['text'][:200] + "...")
