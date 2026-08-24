"""
test_rag_retrieval_relevance.py — RAG Retrieval Relevance Evaluation

Thesis-grade evaluation of the RAG pipeline's retrieval quality.

Methodology:
  1. Define 15 test questions about health conditions and nutrition
  2. For each question, retrieve top-3 ChromaDB chunks
  3. Apply manual relevance judgments (keyword/topic matching)
  4. Report per-question and aggregate % relevant scores

Run:
    python -m pytest tests/test_rag_retrieval_relevance.py -v -s
    (or)
    python tests/test_rag_retrieval_relevance.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 15 TEST QUESTIONS + GROUND-TRUTH RELEVANCE CRITERIA
# ============================================================
# Each entry: question, expected_topics (keywords that a relevant
# chunk MUST contain at least one of), expected_source (which KB
# document should ideally contribute).

TEST_QUESTIONS = [
    # --- Health Conditions (Diabetes) ---
    {
        "id": "Q01",
        "question": "What Nigerian foods should a diabetic person avoid?",
        "relevant_keywords": ["diabetes", "diabetic", "blood sugar", "glycemic", "GI", "carbohydrate control"],
        "expected_source": "health_conditions",
        "topic": "Diabetes — foods to avoid",
    },
    {
        "id": "Q02",
        "question": "Which swallows have a low glycemic index for managing blood sugar?",
        "relevant_keywords": ["glycemic", "GI", "swallow", "amala", "wheat", "blood sugar", "diabetic"],
        "expected_source": "health_conditions",
        "topic": "Diabetes — low GI swallows",
    },
    {
        "id": "Q03",
        "question": "How much carbohydrate per day is recommended for a diabetic?",
        "relevant_keywords": ["carbohydrate", "diabetic", "diabetes", "130", "200", "45", "60"],
        "expected_source": "health_conditions",
        "topic": "Diabetes — carb limits",
    },

    # --- Health Conditions (Hypertension) ---
    {
        "id": "Q04",
        "question": "What foods should I eat to lower my blood pressure?",
        "relevant_keywords": ["hypertension", "blood pressure", "sodium", "potassium", "DASH"],
        "expected_source": "health_conditions",
        "topic": "Hypertension — beneficial foods",
    },
    {
        "id": "Q05",
        "question": "Why are seasoning cubes bad for someone with high blood pressure?",
        "relevant_keywords": ["sodium", "seasoning", "maggi", "knorr", "hypertension", "blood pressure", "salt"],
        "expected_source": "health_conditions",
        "topic": "Hypertension — seasoning cubes & sodium",
    },

    # --- Health Conditions (Pregnancy) ---
    {
        "id": "Q06",
        "question": "What Nigerian foods are rich in folic acid for pregnant women?",
        "relevant_keywords": ["pregnancy", "pregnant", "folic", "folate", "iron", "ugwu", "ewedu"],
        "expected_source": "health_conditions",
        "topic": "Pregnancy — folic acid sources",
    },
    {
        "id": "Q07",
        "question": "How many extra calories does a pregnant woman need daily?",
        "relevant_keywords": ["pregnancy", "pregnant", "calorie", "300", "450", "trimester"],
        "expected_source": "health_conditions",
        "topic": "Pregnancy — calorie requirements",
    },

    # --- Health Conditions (Peptic Ulcer) ---
    {
        "id": "Q08",
        "question": "What foods are safe to eat if I have a stomach ulcer?",
        "relevant_keywords": ["ulcer", "peptic", "stomach", "oatmeal", "pap", "bland", "soothe"],
        "expected_source": "health_conditions",
        "topic": "Peptic ulcer — safe foods",
    },
    {
        "id": "Q09",
        "question": "Should someone with ulcers avoid spicy suya?",
        "relevant_keywords": ["ulcer", "spicy", "suya", "pepper", "irritat", "avoid"],
        "expected_source": "health_conditions",
        "topic": "Peptic ulcer — spicy food avoidance",
    },

    # --- Micronutrients ---
    {
        "id": "Q10",
        "question": "What are the best Nigerian foods high in iron for anemia?",
        "relevant_keywords": ["iron", "anemia", "hemoglobin", "ugwu", "spinach", "heme", "non-heme"],
        "expected_source": "micronutrient_guide",
        "topic": "Iron deficiency — Nigerian sources",
    },
    {
        "id": "Q11",
        "question": "How can I get enough calcium without drinking milk in Nigeria?",
        "relevant_keywords": ["calcium", "bone", "dairy", "ugwu", "ewedu", "crayfish", "snail"],
        "expected_source": "micronutrient_guide",
        "topic": "Calcium — non-dairy Nigerian sources",
    },
    {
        "id": "Q12",
        "question": "Which Nigerian foods are highest in Vitamin A?",
        "relevant_keywords": ["vitamin a", "beta-carotene", "palm oil", "efo riro", "ugwu"],
        "expected_source": "micronutrient_guide",
        "topic": "Vitamin A — Nigerian sources",
    },

    # --- Dietary Guidelines ---
    {
        "id": "Q13",
        "question": "How do I calculate my daily calorie needs using BMR?",
        "relevant_keywords": ["BMR", "mifflin", "calorie", "basal metabolic", "TDEE", "activity"],
        "expected_source": "nigerian_dietary_guidelines",
        "topic": "Calorie computation — BMR formula",
    },
    {
        "id": "Q14",
        "question": "What is a safe calorie deficit for weight loss in Nigeria?",
        "relevant_keywords": ["weight loss", "deficit", "500", "1200", "1500", "calorie"],
        "expected_source": "nigerian_dietary_guidelines",
        "topic": "Weight loss — safe calorie deficit",
    },

    # --- Food Pairings & Culture ---
    {
        "id": "Q15",
        "question": "What soup pairs best with pounded yam in Yoruba cuisine?",
        "relevant_keywords": ["pounded yam", "egusi", "efo riro", "yoruba", "pairing", "swallow"],
        "expected_source": "food_pairings_and_culture",
        "topic": "Food pairings — pounded yam",
    },
]


def judge_relevance(chunk_text: str, keywords: list[str]) -> bool:
    """
    A chunk is RELEVANT if it contains at least one of the ground-truth
    keywords (case-insensitive substring match).
    
    This mirrors the standard human-judgment approach used in IR evaluation
    but is deterministic and reproducible for thesis reporting.
    """
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_evaluation():
    """Execute the full RAG retrieval relevance evaluation."""
    from rag.knowledge_builder import build_knowledge_base
    from rag.retriever import NutritionRetriever

    # Ensure KB is built
    build_knowledge_base()
    retriever = NutritionRetriever()
    assert retriever.is_ready, "Knowledge base is not ready!"

    N_RESULTS = 3
    results_log = []
    total_chunks = 0
    total_relevant = 0
    per_question_scores = []

    print("\n" + "=" * 80)
    print("  RAG RETRIEVAL RELEVANCE EVALUATION")
    print(f"  {len(TEST_QUESTIONS)} questions × top-{N_RESULTS} chunks = "
          f"{len(TEST_QUESTIONS) * N_RESULTS} judgments")
    print("=" * 80)

    for tq in TEST_QUESTIONS:
        qid = tq["id"]
        question = tq["question"]
        keywords = tq["relevant_keywords"]
        expected_src = tq["expected_source"]
        topic = tq["topic"]

        # Retrieve top-3 chunks
        chunks = retriever.retrieve(question, n_results=N_RESULTS)

        q_relevant = 0
        q_total = len(chunks)
        chunk_details = []

        print(f"\n{'-' * 70}")
        print(f"  {qid}: {question}")
        print(f"  Topic: {topic} | Expected source: {expected_src}")
        print(f"{'-' * 70}")

        for i, chunk in enumerate(chunks, 1):
            is_relevant = judge_relevance(chunk["text"], keywords)
            source_match = chunk["source"] == expected_src

            verdict = "[RELEVANT]" if is_relevant else "[NOT RELEVANT]"
            src_icon = "[OK]" if source_match else "[--]"

            if is_relevant:
                q_relevant += 1
                total_relevant += 1
            total_chunks += 1

            # Show a snippet (first 150 chars)
            snippet = chunk["text"][:150].replace("\n", " ") + "..."

            print(f"  Chunk {i}: {verdict}  |  {src_icon} Source: {chunk['source']} "
                  f"— {chunk['section']}  |  Score: {chunk['score']:.4f}")
            print(f"           \"{snippet}\"")

            chunk_details.append({
                "rank": i,
                "source": chunk["source"],
                "section": chunk["section"],
                "distance_score": round(chunk["score"], 4),
                "relevant": is_relevant,
                "source_match": source_match,
                "snippet": snippet,
            })

        q_pct = (q_relevant / q_total * 100) if q_total > 0 else 0
        per_question_scores.append({
            "id": qid,
            "question": question,
            "topic": topic,
            "expected_source": expected_src,
            "relevant_count": q_relevant,
            "total_chunks": q_total,
            "relevance_pct": round(q_pct, 1),
            "chunks": chunk_details,
        })

        status = "PASS" if q_relevant >= 2 else ("PARTIAL" if q_relevant >= 1 else "FAIL")
        print(f"  >> {qid} Result: {q_relevant}/{q_total} relevant ({q_pct:.0f}%) -- {status}")

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    overall_pct = (total_relevant / total_chunks * 100) if total_chunks > 0 else 0
    questions_with_majority_relevant = sum(
        1 for q in per_question_scores if q["relevant_count"] >= 2
    )
    questions_with_any_relevant = sum(
        1 for q in per_question_scores if q["relevant_count"] >= 1
    )

    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)
    print(f"  Total chunks judged:            {total_chunks}")
    print(f"  Relevant chunks:                {total_relevant}")
    print(f"  Overall % Relevant:             {overall_pct:.1f}%")
    print(f"")
    print(f"  Questions with >=2/3 relevant:  {questions_with_majority_relevant}/{len(TEST_QUESTIONS)}"
          f" ({questions_with_majority_relevant/len(TEST_QUESTIONS)*100:.0f}%)")
    print(f"  Questions with >=1/3 relevant:  {questions_with_any_relevant}/{len(TEST_QUESTIONS)}"
          f" ({questions_with_any_relevant/len(TEST_QUESTIONS)*100:.0f}%)")
    print("=" * 80)

    # ============================================================
    # PER-SOURCE BREAKDOWN
    # ============================================================
    source_stats = {}
    for q in per_question_scores:
        src = q["expected_source"]
        if src not in source_stats:
            source_stats[src] = {"relevant": 0, "total": 0, "questions": 0}
        source_stats[src]["relevant"] += q["relevant_count"]
        source_stats[src]["total"] += q["total_chunks"]
        source_stats[src]["questions"] += 1

    print("\n  PER-SOURCE BREAKDOWN:")
    print(f"  {'Source':<35} {'Relevant':<10} {'Total':<8} {'%':>6}")
    print(f"  {'-' * 62}")
    for src, stats in sorted(source_stats.items()):
        pct = stats["relevant"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {src:<35} {stats['relevant']:<10} {stats['total']:<8} {pct:>5.1f}%")

    # ============================================================
    # SAVE DETAILED RESULTS TO JSON
    # ============================================================
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "methodology": "Top-3 ChromaDB retrieval, keyword-based relevance judgment",
        "embedding_model": "all-MiniLM-L6-v2",
        "n_results": N_RESULTS,
        "num_questions": len(TEST_QUESTIONS),
        "total_chunks_judged": total_chunks,
        "total_relevant": total_relevant,
        "overall_relevance_pct": round(overall_pct, 1),
        "questions_with_majority_relevant": questions_with_majority_relevant,
        "questions_with_any_relevant": questions_with_any_relevant,
        "per_source_breakdown": {
            src: {
                "relevant": s["relevant"],
                "total": s["total"],
                "pct": round(s["relevant"] / s["total"] * 100, 1) if s["total"] > 0 else 0
            }
            for src, s in source_stats.items()
        },
        "per_question_results": per_question_scores,
    }

    report_path = Path(__file__).parent.parent / "tests" / "rag_retrieval_relevance_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  [SAVED] Detailed report saved to: {report_path}")
    print("=" * 80)

    return report


# ============================================================
# PYTEST WRAPPER — so `pytest -v` picks this up
# ============================================================
class TestRAGRetrievalRelevance:
    """Thesis evaluation: RAG retrieval relevance on 15 health-condition questions."""

    def test_overall_relevance_above_threshold(self):
        """
        Overall % of relevant retrieved chunks should be ≥ 70%.
        This is a standard RAG quality threshold for domain-specific KBs.
        """
        report = run_evaluation()
        assert report["overall_relevance_pct"] >= 70.0, (
            f"Overall relevance {report['overall_relevance_pct']}% is below "
            f"the 70% threshold. Review knowledge base coverage."
        )

    def test_majority_questions_have_relevant_chunks(self):
        """
        At least 80% of questions should have ≥1 relevant chunk in top-3.
        A question with zero relevant chunks means the KB has a coverage gap.
        """
        report = run_evaluation()
        coverage = report["questions_with_any_relevant"] / report["num_questions"] * 100
        assert coverage >= 80.0, (
            f"Only {coverage:.0f}% of questions had at least 1 relevant chunk. "
            f"Expected ≥80%. Check knowledge base for coverage gaps."
        )


# ============================================================
# STANDALONE EXECUTION
# ============================================================
if __name__ == "__main__":
    report = run_evaluation()
    print(f"\n  FINAL SCORE: {report['overall_relevance_pct']}% retrieval relevance")
    print(f"  (Threshold for thesis: >=70%)")
