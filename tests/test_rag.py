"""
test_rag.py — Tests for the RAG (Retrieval-Augmented Generation) pipeline.

Tests knowledge base building, document chunking, ChromaDB storage,
and semantic retrieval quality.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

KB_DIR = Path(__file__).parent.parent / "rag" / "knowledge_base"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


class TestKnowledgeBase:
    """Tests for knowledge base documents."""

    def test_knowledge_base_directory_exists(self):
        """Knowledge base directory should exist."""
        assert KB_DIR.exists()
        assert KB_DIR.is_dir()

    def test_all_documents_present(self):
        """All four knowledge base documents should exist."""
        expected_docs = [
            "nigerian_dietary_guidelines.md",
            "food_pairings_and_culture.md",
            "health_conditions.md",
            "micronutrient_guide.md",
        ]
        for doc in expected_docs:
            path = KB_DIR / doc
            assert path.exists(), f"Missing knowledge base document: {doc}"
            assert path.stat().st_size > 100, f"Document too small: {doc}"

    def test_documents_have_content(self):
        """Each document should have meaningful content."""
        for md_file in KB_DIR.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            assert len(content) > 500, f"{md_file.name} is too short ({len(content)} chars)"
            # Should have headings
            assert "#" in content, f"{md_file.name} has no markdown headings"


class TestKnowledgeBuilder:
    """Tests for the knowledge builder (ingestion pipeline)."""

    def test_chunking_function(self):
        """Text chunking should produce overlapping chunks."""
        from rag.knowledge_builder import _chunk_text

        text = "First paragraph.\n\n" * 20  # ~320 chars
        chunks = _chunk_text(text, chunk_size=100, overlap=30)
        assert len(chunks) > 1
        assert all(len(c) > 0 for c in chunks)

    def test_chunking_preserves_content(self):
        """All original content should appear in at least one chunk."""
        from rag.knowledge_builder import _chunk_text

        text = "Alpha content here.\n\nBeta content here.\n\nGamma content here."
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        combined = " ".join(chunks)
        assert "Alpha" in combined
        assert "Beta" in combined
        assert "Gamma" in combined

    def test_load_documents(self):
        """Should load and chunk all knowledge base documents."""
        from rag.knowledge_builder import _load_documents

        documents = _load_documents()
        assert len(documents) > 0
        # Each doc should have required fields
        for doc in documents:
            assert "text" in doc
            assert "source" in doc
            assert "chunk_id" in doc
            assert len(doc["text"]) > 0

    def test_build_knowledge_base(self):
        """Should build ChromaDB collection successfully."""
        from rag.knowledge_builder import build_knowledge_base

        collection = build_knowledge_base()
        assert collection is not None
        assert collection.count() > 0


class TestRetriever:
    """Tests for the semantic retrieval pipeline."""

    @pytest.fixture(autouse=True)
    def ensure_kb_built(self):
        """Ensure knowledge base is built before retrieval tests."""
        from rag.knowledge_builder import build_knowledge_base
        build_knowledge_base()

    def test_retriever_initialization(self):
        """Retriever should initialize successfully."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        assert retriever.is_ready

    def test_basic_retrieval(self):
        """Should return results for a nutrition query."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results = retriever.retrieve("What foods are good for diabetes?")
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("source" in r for r in results)
        assert all("score" in r for r in results)

    def test_retrieval_relevance(self):
        """Diabetes query should retrieve health_conditions document."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results = retriever.retrieve("diabetes diet management blood sugar")
        sources = [r["source"] for r in results]
        assert "health_conditions" in sources

    def test_food_pairing_retrieval(self):
        """Food pairing query should retrieve pairing document."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results = retriever.retrieve("What goes with pounded yam Yoruba food pairing")
        sources = [r["source"] for r in results]
        assert "food_pairings_and_culture" in sources

    def test_source_filter(self):
        """Source filter should limit results to specified document."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results = retriever.retrieve(
            "iron rich foods",
            source_filter="micronutrient_guide"
        )
        for r in results:
            assert r["source"] == "micronutrient_guide"

    def test_retrieve_as_context(self):
        """Should return a formatted context string."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        context = retriever.retrieve_as_context("Nigerian dietary guidelines")
        assert isinstance(context, str)
        assert len(context) > 50
        assert "[Source:" in context

    def test_n_results_limit(self):
        """Should respect the n_results parameter."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results_1 = retriever.retrieve("nutrition", n_results=1)
        results_3 = retriever.retrieve("nutrition", n_results=3)
        assert len(results_1) == 1
        assert len(results_3) <= 3

    def test_empty_query(self):
        """Empty or short query should still return results (not crash)."""
        from rag.retriever import NutritionRetriever

        retriever = NutritionRetriever()
        results = retriever.retrieve("food")
        assert isinstance(results, list)
