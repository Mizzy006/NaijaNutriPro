"""
engine.py — Hybrid ML Meal Plan Orchestrator.

This module orchestrates the meal planning process by combining:
1. Content-Based ML Scoring (ml/food_scorer.py) — ranks foods by nutritional fit
2. PuLP Constraint Optimization (ml/optimizer.py) — selects optimal combos
3. Rule-Based Conflict Detection — cultural food pairing rules (fallback)

The original greedy heuristic is preserved as a fallback when the optimizer
cannot find a solution (e.g., very tight budgets).
"""

import pandas as pd
from pathlib import Path

from ml.food_scorer import FoodScorer
from ml.optimizer import MealOptimizer, SAME_BASE_GROUPS

# Build a lookup: food name → set of sibling names (same ingredient base).
# Used for within-day dedup: if "Indomie + Egg" is at Lunch,
# "Indomie (2 Packs)" is blocked from Dinner.
_SIBLING_MAP: dict[str, set[str]] = {}
for _group in SAME_BASE_GROUPS:
    for _name in _group:
        _SIBLING_MAP.setdefault(_name, set()).update(_group - {_name})

# Resolve data path relative to this file
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "nigerian_food_nutrition.csv"


# ============================================================
# HEALTH CONDITION → CONSTRAINT MAPPING
# ============================================================

def get_health_constraints(health_conditions: list = None) -> dict:
    """
    Map health conditions to ML scoring constraints.
    
    Args:
        health_conditions: List of condition strings, e.g., ["Diabetes", "Hypertension"]
    
    Returns:
        Dict of constraints for FoodScorer.score_foods()
    """
    if not health_conditions:
        return {}

    constraints = {}
    prefer_tags = []

    for condition in health_conditions:
        if condition == "Diabetes":
            constraints['avoid_sugar'] = True
            prefer_tags.append('diabetic-friendly')
        elif condition == "Hypertension":
            constraints['max_sodium'] = 600  # per meal limit
            prefer_tags.append('heart-healthy')
        elif condition == "Pregnancy":
            prefer_tags.append('pregnancy-safe')
        elif condition == "Ulcer":
            constraints['avoid_spicy'] = True
            constraints['avoid_oily'] = True
            prefer_tags.append('ulcer-friendly')

    if prefer_tags:
        constraints['prefer_tags'] = prefer_tags

    return constraints


# ============================================================
# DIETARY FILTERING (preserved from original)
# ============================================================

def filter_by_diet(df: pd.DataFrame, dietary_preference: str) -> pd.DataFrame:
    """Filter food database by dietary preference."""
    if dietary_preference == "None":
        return df

    forbidden_words = {
        "Vegetarian": ["meat", "fish", "chicken", "beef", "pork", "poultry",
                       "turkey", "ponmo", "snail", "goat", "suya", "kilishi", "asun"],
        "Pescatarian": ["meat", "chicken", "beef", "pork", "poultry",
                        "turkey", "ponmo", "goat", "suya", "kilishi", "asun"],
        "Halal": ["pork", "bacon", "ham", "alcohol"],
    }.get(dietary_preference, [])

    if not forbidden_words:
        return df

    pattern = '|'.join(forbidden_words)
    filtered = df[
        ~df['Name'].str.lower().str.contains(pattern, na=False) &
        ~df['Category'].str.lower().str.contains(pattern, na=False) &
        ~df['Tags'].str.lower().str.contains(pattern, na=False)
    ]

    return filtered


# ============================================================
# EMERGENCY FOOD (Last Resort — preserved from original)
# ============================================================

def get_emergency_food(df: pd.DataFrame, available_budget: float) -> list[dict] | None:
    """Get the highest-calorie affordable food when all else fails."""
    candidates = df[
        (df['Price_Avg_Naira'] <= available_budget) &
        (~df['Category'].isin(["Drink", "Spice", "Ingredient", "Soup"]))
    ]

    # Also exclude condiment items (Stockfish, Ponmo, raw vegetables, etc.)
    if 'Pair_Category' in df.columns:
        candidates = candidates[candidates['Pair_Category'] != 'condiment']

    if candidates.empty:
        return None

    best_item = candidates.sort_values('Calories_kcal', ascending=False).iloc[0]

    return [{
        "Name": best_item['Name'] + " (Budget Saver)",
        "Calories": int(best_item['Calories_kcal']),
        "Price": int(best_item['Price_Avg_Naira']),
        "Category": "Emergency",
        "ml_score": 0.0,
    }]


# ============================================================
# HYBRID ML MEAL GENERATION
# ============================================================

def generate_weekly_plan(
    target_calories_daily: float,
    daily_budget_limit: float,
    dietary_preference: str = "None",
    health_conditions: list = None,
) -> dict | str:
    """
    Generate a 7-day meal plan using the Hybrid ML pipeline:
    
    1. Load and filter food database
    2. Initialize FoodScorer (content-based ML) and MealOptimizer (PuLP)
    3. For each day × meal:
       a. Score foods with content-based recommender
       b. Optimize meal selection with PuLP integer programming
       c. Fall back to emergency food if needed
    4. Track variety and budget across days
    
    Args:
        target_calories_daily: Target daily calorie intake.
        daily_budget_limit: Daily food budget in Naira.
        dietary_preference: "None", "Halal", "Vegetarian", or "Pescatarian".
        health_conditions: List of conditions, e.g., ["Diabetes", "Hypertension"].
    
    Returns:
        Weekly plan dict, or error string.
    """
    # --- 1. LOAD DATA ---
    try:
        df = pd.read_csv(CSV_PATH)
    except (FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return f"Error: Could not load food data — {e}"

    # --- 2. DIETARY FILTERING ---
    df = filter_by_diet(df, dietary_preference)
    if df.empty:
        return f"Error: No foods available for the {dietary_preference} diet in the database."

    # --- 3. INITIALIZE ML COMPONENTS ---
    scorer = FoodScorer(df)
    optimizer = MealOptimizer()
    health_constraints = get_health_constraints(health_conditions)

    # --- 4. GENERATE WEEKLY PLAN ---
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_plan = {}
    recent_meals = []
    current_wallet = 0.0

    for day_index, day in enumerate(days):
        current_wallet += daily_budget_limit
        daily_menu = {}
        daily_cost = 0
        daily_cal = 0
        meals = ["Breakfast", "Lunch", "Dinner"]
        today_meals = []  # Track foods selected today for within-day dedup

        for meal_type in meals:
            # Budget allocation per meal — ENSURE all 3 meals get funded
            # Breakfast: max 25% of daily budget
            # Lunch:     max 40% of daily budget
            # Dinner:    gets whatever remains (at least 35%)
            if meal_type == "Breakfast":
                meal_budget_cap = min(current_wallet, daily_budget_limit * 0.25)
            elif meal_type == "Lunch":
                meal_budget_cap = min(current_wallet, daily_budget_limit * 0.40)
            else:  # Dinner — use whatever is left
                meal_budget_cap = current_wallet

            # Calorie target per meal (Breakfast 25%, Lunch 40%, Dinner 35%)
            if meal_type == "Lunch":
                meal_cal_target = target_calories_daily * 0.40
            elif meal_type == "Dinner":
                meal_cal_target = target_calories_daily * 0.35
            else:  # Breakfast
                meal_cal_target = target_calories_daily * 0.25

            # --- ML SCORING (day-aware for variety) ---
            scored_foods = scorer.score_foods(
                target_calories=meal_cal_target,
                meal_type=meal_type,
                health_constraints=health_constraints,
                day_index=day_index,
            )

            # --- PuLP OPTIMIZATION (with day-level dedup) ---
            combo = optimizer.optimize_meal(
                scored_foods=scored_foods,
                target_calories=meal_cal_target,
                budget=meal_budget_cap,
                meal_type=meal_type,
                last_meals=recent_meals,
                today_meals=today_meals,
            )

            # Fallback: retry without variety constraint
            if not combo:
                combo = optimizer.optimize_meal(
                    scored_foods=scored_foods,
                    target_calories=meal_cal_target,
                    budget=meal_budget_cap,
                    meal_type=meal_type,
                    last_meals=[],
                    today_meals=today_meals,  # Still enforce within-day dedup
                    calorie_tolerance=300,  # More lenient
                )

            # Fallback: emergency food
            if not combo and current_wallet >= 150:
                combo = get_emergency_food(df, current_wallet)

            if combo:
                for item in combo:
                    name = item['Name']
                    recent_meals.append(name)
                    today_meals.append(name)
                    # Also block same-base siblings for within-day dedup
                    # e.g. "Indomie + Egg" blocks "Indomie (2 Packs)"
                    for sibling in _SIBLING_MAP.get(name, []):
                        if sibling not in today_meals:
                            today_meals.append(sibling)
                # Keep full week history — no truncation.
                # All meals eaten so far are tracked for variety.

                meal_cost = sum(item['Price'] for item in combo)
                current_wallet -= meal_cost

                daily_menu[meal_type] = combo
                daily_cost += meal_cost
                daily_cal += sum(item['Calories'] for item in combo)
            else:
                daily_menu[meal_type] = [{
                    "Name": "Skipped (Insufficient Funds)",
                    "Calories": 0, "Price": 0, "ml_score": 0.0
                }]

        weekly_plan[day] = {
            "Plan": daily_menu,
            "Total_Cost": daily_cost,
            "Total_Calories": daily_cal,
            "Wallet_Balance": round(current_wallet, 2),
        }

    return weekly_plan