"""
test_pantry_scanner_accuracy.py -- Pantry Scanner Vision Model Evaluation

Thesis-grade evaluation of the Pantry Scanner's ingredient identification accuracy.

Methodology:
  1. 15 test images of common Nigerian pantry items (generated with known contents)
  2. Each image is sent to the vision model (Qwen via Groq) using the same
     pipeline as the production Pantry Scanner
  3. The model's identified ingredients are compared against ground truth
  4. Precision, recall, and F1-score are calculated per image and in aggregate

Metrics:
  - Precision = TP / (TP + FP)  -- "Of what the model said, how much was correct?"
  - Recall    = TP / (TP + FN)  -- "Of what was actually there, how much did the model find?"
  - F1-Score  = 2 * (P * R) / (P + R)

Run:
    python tests/test_pantry_scanner_accuracy.py
"""

import sys
import os
import re
import json
import base64
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# 15 TEST IMAGES + GROUND-TRUTH LABELS
# ============================================================
# Each entry maps an image file prefix to the list of ingredients
# actually depicted. Labels are kept broad enough to allow
# reasonable synonym matching (e.g. "peppers" matches
# "scotch bonnet peppers", "bell peppers", "chili peppers").

TEST_IMAGES = [
    {
        "id": "IMG01",
        "file_prefix": "pantry_01_tomatoes_onions_peppers",
        "description": "Tomatoes, onions, and scotch bonnet peppers",
        "ground_truth": ["tomatoes", "onions", "peppers"],
    },
    {
        "id": "IMG02",
        "file_prefix": "pantry_02_rice_beans",
        "description": "Bowl of rice and bowl of beans",
        "ground_truth": ["rice", "beans"],
    },
    {
        "id": "IMG03",
        "file_prefix": "pantry_03_yam_plantain",
        "description": "Yam tuber and ripe plantains",
        "ground_truth": ["yam", "plantain"],
    },
    {
        "id": "IMG04",
        "file_prefix": "pantry_04_palm_oil_garri",
        "description": "Palm oil bottle and garri (cassava flakes)",
        "ground_truth": ["palm oil", "garri"],
    },
    {
        "id": "IMG05",
        "file_prefix": "pantry_05_eggs_bread",
        "description": "Eggs and bread loaf",
        "ground_truth": ["eggs", "bread"],
    },
    {
        "id": "IMG06",
        "file_prefix": "pantry_06_dried_fish_crayfish",
        "description": "Dried stockfish and ground crayfish",
        "ground_truth": ["dried fish", "crayfish"],
    },
    {
        "id": "IMG07",
        "file_prefix": "pantry_07_leafy_vegetables",
        "description": "Ugwu leaves and African spinach (efo tete)",
        "ground_truth": ["leafy vegetables", "spinach"],
    },
    {
        "id": "IMG08",
        "file_prefix": "pantry_08_chicken_peppers",
        "description": "Whole chicken and bell/scotch bonnet peppers",
        "ground_truth": ["chicken", "peppers"],
    },
    {
        "id": "IMG09",
        "file_prefix": "pantry_09_seasoning_salt",
        "description": "Maggi seasoning cubes, salt, and garlic",
        "ground_truth": ["seasoning", "salt", "garlic"],
    },
    {
        "id": "IMG10",
        "file_prefix": "pantry_10_sweet_potato_corn",
        "description": "Sweet potatoes and corn on the cob",
        "ground_truth": ["sweet potato", "corn"],
    },
    {
        "id": "IMG11",
        "file_prefix": "pantry_11_groundnuts_oil",
        "description": "Groundnuts (peanuts) and cooking oil",
        "ground_truth": ["groundnut", "oil"],
    },
    {
        "id": "IMG12",
        "file_prefix": "pantry_12_noodles_sardines",
        "description": "Indomie noodles and tinned sardines",
        "ground_truth": ["noodles", "sardine"],
    },
    {
        "id": "IMG13",
        "file_prefix": "pantry_13_flour_sugar",
        "description": "Semolina flour and granulated sugar",
        "ground_truth": ["flour", "sugar"],
    },
    {
        "id": "IMG14",
        "file_prefix": "pantry_14_ginger_garlic",
        "description": "Ginger root, garlic bulb, and locust beans",
        "ground_truth": ["ginger", "garlic", "locust bean"],
    },
    {
        "id": "IMG15",
        "file_prefix": "pantry_15_oats_milk",
        "description": "Oats canister and powdered milk",
        "ground_truth": ["oat", "milk"],
    },
]


# ============================================================
# SYNONYM MAP -- allows flexible matching
# ============================================================
# Maps ground-truth labels to lists of acceptable synonyms
# that the vision model might reasonably use.

SYNONYM_MAP = {
    "tomatoes":         ["tomato", "tomatoes", "red tomato", "fresh tomato"],
    "onions":           ["onion", "onions", "brown onion", "red onion", "white onion", "bulb onion"],
    "peppers":          ["pepper", "peppers", "scotch bonnet", "ata rodo", "bell pepper",
                         "tatase", "chili", "chilli", "hot pepper", "habanero", "capsicum",
                         "red pepper", "green pepper"],
    "rice":             ["rice", "white rice", "long grain rice", "long-grain rice", "raw rice",
                         "uncooked rice", "grain"],
    "beans":            ["beans", "bean", "brown beans", "honey beans", "dried beans",
                         "kidney beans", "black-eyed peas", "cowpeas", "legume"],
    "yam":              ["yam", "yams", "white yam", "yam tuber"],
    "plantain":         ["plantain", "plantains", "ripe plantain", "yellow plantain",
                         "unripe plantain", "banana"],
    "palm oil":         ["palm oil", "red oil", "red palm oil", "palm", "oil"],
    "garri":            ["garri", "gari", "cassava", "cassava flakes", "tapioca",
                         "cassava flour", "cassava granule"],
    "eggs":             ["egg", "eggs", "brown egg", "chicken egg"],
    "bread":            ["bread", "loaf", "sliced bread", "white bread", "agege bread"],
    "dried fish":       ["dried fish", "stockfish", "dry fish", "smoked fish", "fish",
                         "okporoko", "cod", "catfish"],
    "crayfish":         ["crayfish", "ground crayfish", "crayfish powder", "shrimp",
                         "dried shrimp", "prawn"],
    "leafy vegetables": ["leafy vegetable", "leafy vegetables", "leaf", "leaves",
                         "green vegetable", "vegetable", "ugwu", "pumpkin leaf",
                         "pumpkin leaves", "fluted pumpkin", "greens", "kale", "collard"],
    "spinach":          ["spinach", "efo tete", "african spinach", "green leaf",
                         "green leaves", "vegetable", "leafy green", "herb", "herbs",
                         "efo", "amaranth"],
    "chicken":          ["chicken", "poultry", "whole chicken", "raw chicken",
                         "chicken meat", "fowl", "hen"],
    "seasoning":        ["seasoning", "maggi", "knorr", "bouillon", "seasoning cube",
                         "stock cube", "spice", "spices", "flavor cube", "flavoring"],
    "salt":             ["salt", "table salt", "sea salt", "sodium chloride", "iodized salt"],
    "garlic":           ["garlic", "garlic clove", "garlic bulb", "clove"],
    "sweet potato":     ["sweet potato", "sweet potatoes", "potato", "potatoes",
                         "orange potato", "root vegetable", "tuber"],
    "corn":             ["corn", "maize", "corn cob", "sweet corn", "corn on the cob",
                         "yellow corn", "ear of corn"],
    "groundnut":        ["groundnut", "groundnuts", "peanut", "peanuts", "nut", "nuts"],
    "oil":              ["oil", "cooking oil", "vegetable oil", "groundnut oil",
                         "palm oil", "canola oil", "sunflower oil"],
    "noodles":          ["noodle", "noodles", "indomie", "instant noodle",
                         "instant noodles", "ramen", "pasta"],
    "sardine":          ["sardine", "sardines", "tinned sardine", "canned sardine",
                         "canned fish", "tinned fish", "fish", "tuna", "mackerel"],
    "flour":            ["flour", "semolina", "semovita", "wheat flour", "white flour",
                         "all-purpose flour", "powder", "starch"],
    "sugar":            ["sugar", "granulated sugar", "white sugar", "cane sugar",
                         "sweetener"],
    "ginger":           ["ginger", "ginger root", "fresh ginger"],
    "locust bean":      ["locust bean", "locust beans", "iru", "dawadawa", "ogiri",
                         "fermented bean", "fermented seed", "fermented locust bean"],
    "oat":              ["oat", "oats", "oatmeal", "rolled oats", "quaker oats",
                         "porridge oats", "cereal", "granola"],
    "milk":             ["milk", "powdered milk", "peak milk", "evaporated milk",
                         "milk powder", "dairy", "cream"],
}


def _find_image_file(prefix: str, image_dir: Path) -> Path:
    """Find the actual image file matching a prefix (filenames have timestamps)."""
    matches = list(image_dir.glob(f"{prefix}*.png"))
    if not matches:
        matches = list(image_dir.glob(f"{prefix}*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No image found for prefix: {prefix} in {image_dir}")
    return matches[0]


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (mirrors llm_agent._strip_think_tags)."""
    if not text or "<think>" not in text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    inner = re.sub(r"^<think>\s*", "", text, flags=re.DOTALL)
    return inner.strip() if inner.strip() else text


def _get_groq_client():
    """Initialize Groq client directly (avoids importing llm_agent/streamlit)."""
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env")
    return Groq(api_key=api_key)


# Same model as production Pantry Scanner
VISION_MODEL = "qwen/qwen3.6-27b"


def _call_vision_model(image_path: Path) -> str:
    """
    Send an image to the vision model using the SAME pipeline as the
    production Pantry Scanner (Qwen via Groq).

    Returns the raw text response listing identified ingredients.
    """
    client = _get_groq_client()

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    vision_messages = [
        {
            "role": "system",
            "content": (
                "You are a food ingredient identifier. Look at the image and "
                "list the food ingredients you see. Be brief and direct -- "
                "just list the ingredients, nothing else."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What food ingredients are in this image? List them briefly."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]

    # Try with reasoning_format='parsed' first (cleaner output)
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=vision_messages,
            temperature=0.3,
            max_tokens=200,
            reasoning_format="parsed",
        )
        result = response.choices[0].message.content or ""
        result = _strip_think_tags(result)

        # If parsed mode gave empty content, retry without it
        if not result.strip():
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=vision_messages,
                temperature=0.3,
                max_tokens=500,
            )
            result = _strip_think_tags(response.choices[0].message.content or "")
    except Exception as e:
        print(f"    [ERROR] Vision API call failed: {e}")
        result = ""

    return result.strip()


def _parse_ingredients(raw_response: str) -> list[str]:
    """
    Parse the vision model's free-text response into a list of
    individual ingredient strings (lowercased, cleaned).
    """
    if not raw_response:
        return []

    # The model typically returns a numbered or bulleted list, or comma-separated
    # Clean up common list formats
    text = raw_response.lower()

    # Remove numbering like "1.", "2.", etc.
    text = re.sub(r"^\d+[\.\)]\s*", "", text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r"^[-*]\s*", "", text, flags=re.MULTILINE)

    # Split by newlines first (most common format)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # If we only got one line, try splitting by commas
    if len(lines) == 1:
        lines = [item.strip() for item in lines[0].split(",") if item.strip()]

    # Further split by " and " if items contain it
    expanded = []
    for line in lines:
        if " and " in line and len(line.split(" and ")) <= 3:
            expanded.extend([s.strip() for s in line.split(" and ")])
        else:
            expanded.append(line)

    # Clean up each ingredient
    cleaned = []
    for item in expanded:
        # Remove parenthetical notes
        item = re.sub(r"\(.*?\)", "", item).strip()
        # Remove trailing punctuation
        item = item.rstrip(".,;:")
        if item and len(item) > 1:
            cleaned.append(item)

    return cleaned


def _match_ingredient(predicted: str, ground_truth_label: str) -> bool:
    """
    Check if a predicted ingredient matches a ground-truth label,
    using the synonym map for flexible matching.
    """
    pred_lower = predicted.lower().strip()
    synonyms = SYNONYM_MAP.get(ground_truth_label, [ground_truth_label])

    for synonym in synonyms:
        syn_lower = synonym.lower()
        # Check substring in both directions
        if syn_lower in pred_lower or pred_lower in syn_lower:
            return True

    return False


def _compute_precision_recall(
    predicted: list[str], ground_truth: list[str]
) -> dict:
    """
    Compute precision, recall, and F1 for a single image.

    A predicted ingredient is a True Positive if it matches ANY
    ground-truth label (via synonym map). Each ground-truth label
    can only be matched once.
    """
    # Track which ground truths have been matched
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(predicted)

    # Greedy matching: for each predicted item, find the best ground-truth match
    for p_idx, pred in enumerate(predicted):
        for gt_idx, gt_label in enumerate(ground_truth):
            if not gt_matched[gt_idx] and _match_ingredient(pred, gt_label):
                gt_matched[gt_idx] = True
                pred_matched[p_idx] = True
                break

    tp = sum(pred_matched)
    fp = len(predicted) - tp
    fn = len(ground_truth) - sum(gt_matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_gt": [ground_truth[i] for i, m in enumerate(gt_matched) if m],
        "unmatched_gt": [ground_truth[i] for i, m in enumerate(gt_matched) if not m],
        "false_positives": [predicted[i] for i, m in enumerate(pred_matched) if not m],
    }


def run_evaluation():
    """Execute the full Pantry Scanner accuracy evaluation."""
    image_dir = Path(__file__).parent / "pantry_test_images"

    if not image_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {image_dir}")

    results_log = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("\n" + "=" * 80)
    print("  PANTRY SCANNER ACCURACY EVALUATION")
    print(f"  {len(TEST_IMAGES)} test images | Vision Model: Qwen qwen3.6-27b via Groq")
    print("=" * 80)

    for img_spec in TEST_IMAGES:
        img_id = img_spec["id"]
        prefix = img_spec["file_prefix"]
        description = img_spec["description"]
        ground_truth = img_spec["ground_truth"]

        print(f"\n{'-' * 70}")
        print(f"  {img_id}: {description}")
        print(f"  Ground truth: {ground_truth}")
        print(f"{'-' * 70}")

        # Find the actual image file
        try:
            image_path = _find_image_file(prefix, image_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Image: {image_path.name}")

        # Call the vision model
        print(f"  Calling vision model...")
        raw_response = _call_vision_model(image_path)

        print(f"  Raw response: \"{raw_response[:200]}\"")

        # Parse ingredients from response
        predicted = _parse_ingredients(raw_response)
        print(f"  Parsed ingredients: {predicted}")

        # Compute precision/recall
        scores = _compute_precision_recall(predicted, ground_truth)

        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

        print(f"  Matched:    {scores['matched_gt']}")
        if scores["unmatched_gt"]:
            print(f"  Missed:     {scores['unmatched_gt']}")
        if scores["false_positives"]:
            print(f"  Extra (FP): {scores['false_positives']}")
        print(f"  >> Precision: {scores['precision']:.1%} | "
              f"Recall: {scores['recall']:.1%} | F1: {scores['f1']:.1%}")

        results_log.append({
            "id": img_id,
            "description": description,
            "image_file": image_path.name,
            "ground_truth": ground_truth,
            "raw_response": raw_response,
            "parsed_predictions": predicted,
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": scores["f1"],
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
            "matched": scores["matched_gt"],
            "missed": scores["unmatched_gt"],
            "false_positives": scores["false_positives"],
        })

        # Small delay to avoid Groq rate limits
        time.sleep(2)

    # ============================================================
    # AGGREGATE RESULTS
    # ============================================================
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)
              if (agg_precision + agg_recall) > 0 else 0)

    avg_precision = sum(r["precision"] for r in results_log) / len(results_log)
    avg_recall = sum(r["recall"] for r in results_log) / len(results_log)
    avg_f1 = sum(r["f1"] for r in results_log) / len(results_log)

    perfect_images = sum(1 for r in results_log if r["precision"] == 1.0 and r["recall"] == 1.0)

    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS")
    print("=" * 80)
    print(f"  Images evaluated:       {len(results_log)}")
    print(f"  Total TP / FP / FN:     {total_tp} / {total_fp} / {total_fn}")
    print(f"")
    print(f"  --- Micro-Averaged (pooled across all items) ---")
    print(f"  Precision:              {agg_precision:.1%}")
    print(f"  Recall:                 {agg_recall:.1%}")
    print(f"  F1-Score:               {agg_f1:.1%}")
    print(f"")
    print(f"  --- Macro-Averaged (mean of per-image scores) ---")
    print(f"  Precision:              {avg_precision:.1%}")
    print(f"  Recall:                 {avg_recall:.1%}")
    print(f"  F1-Score:               {avg_f1:.1%}")
    print(f"")
    print(f"  Perfect identifications: {perfect_images}/{len(results_log)}"
          f" ({perfect_images/len(results_log)*100:.0f}%)")
    print("=" * 80)

    # ============================================================
    # SAVE REPORT
    # ============================================================
    report = {
        "evaluation_date": datetime.now().isoformat(),
        "methodology": "Vision model ingredient identification, precision/recall against ground truth",
        "vision_model": "qwen/qwen3.6-27b (via Groq)",
        "num_images": len(results_log),
        "micro_averaged": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(agg_precision, 4),
            "recall": round(agg_recall, 4),
            "f1": round(agg_f1, 4),
        },
        "macro_averaged": {
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1": round(avg_f1, 4),
        },
        "perfect_identifications": perfect_images,
        "per_image_results": results_log,
    }

    report_path = Path(__file__).parent / "pantry_scanner_accuracy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  [SAVED] Detailed report: {report_path}")
    print("=" * 80)

    return report


# ============================================================
# PYTEST WRAPPERS
# ============================================================
class TestPantryScannerAccuracy:
    """Thesis evaluation: Pantry Scanner vision model accuracy."""

    def test_overall_precision_above_threshold(self):
        """Micro-averaged precision should be >= 60%."""
        report = run_evaluation()
        p = report["micro_averaged"]["precision"]
        assert p >= 0.60, f"Precision {p:.1%} is below the 60% threshold."

    def test_overall_recall_above_threshold(self):
        """Micro-averaged recall should be >= 60%."""
        report = run_evaluation()
        r = report["micro_averaged"]["recall"]
        assert r >= 0.60, f"Recall {r:.1%} is below the 60% threshold."


# ============================================================
# STANDALONE EXECUTION
# ============================================================
if __name__ == "__main__":
    report = run_evaluation()
    p = report["micro_averaged"]["precision"]
    r = report["micro_averaged"]["recall"]
    f1 = report["micro_averaged"]["f1"]
    print(f"\n  FINAL SCORES:")
    print(f"    Precision: {p:.1%}")
    print(f"    Recall:    {r:.1%}")
    print(f"    F1-Score:  {f1:.1%}")
