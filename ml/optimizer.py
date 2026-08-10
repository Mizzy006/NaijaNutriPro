"""
optimizer.py — PuLP Constraint Optimization for Meal Selection.

Uses Pair_Category-based cultural pairing rules + PuLP integer programming
to select culturally coherent, nutritionally optimal food combinations.

The optimization problem:
    Maximize: sum(ml_score[i] * x[i])  — prefer ML-recommended foods
    Subject to:
        - Budget constraint (hard limit)
        - Calorie target (minimize deviation)
        - Pair_Category compatibility (cultural pairing rules)
        - Swallow ↔ Soup bidirectional pairing
        - Same-base-group exclusion (no yam + yam, no beans + beans)
        - At most 1 oily item
        - Meal size: 1–2 items
        - At least 1 "main" food item
        - At most 1 item per Pair_Category
        - Variety: penalize recently eaten foods
"""

import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, PULP_CBC_CMD
)


# ============================================================
# PAIR CATEGORY COMPATIBILITY
# ============================================================
# Defines which Pair_Category values can coexist in the same meal.
# Foods with incompatible pair categories CANNOT be selected together.
# This is the PRIMARY mechanism for ensuring cultural coherence.

COMPATIBLE_PAIRS = {
    'needs-soup':       {'needs-swallow', 'protein'},   # Swallow + soup + protein OK
    'needs-swallow':    {'needs-soup', 'protein'},       # Soup + swallow + protein OK
    'rice-main':        {'protein', 'side', 'drink'},    # Rice + chicken, rice + dodo
    'standalone':       {'drink'},                       # Complete meal, only + drink
    'light-standalone': {'drink', 'breakfast-base'},     # Moi-Moi + pap, or alone
    'breakfast-base':   {'breakfast-side', 'protein', 'light-standalone'},
    'breakfast-side':   {'breakfast-base'},               # Akara only with pap/bread
    'protein':          {'rice-main', 'breakfast-base',
                         'needs-soup', 'needs-swallow', 'side'},  # Addon role
    'side':             {'rice-main', 'protein'},         # Dodo + rice, or dodo + chicken
    'drink':            {'rice-main', 'standalone', 'light-standalone',
                         'peppersoup', 'snack'},
    'peppersoup':       {'drink'},                       # Peppersoup + drink
    'snack':            {'drink'},                        # Snack + drink
    'condiment':        set(),                            # Never selected
}

# Pair categories that count as a "meal base" — at least 1 required per meal
MAIN_PAIR_CATS = {
    'needs-soup', 'needs-swallow', 'rice-main', 'standalone',
    'light-standalone', 'peppersoup', 'breakfast-base', 'snack',
}

# Allowed pair categories per meal type
ALLOWED_PAIR_CATS = {
    'Breakfast': {'standalone', 'rice-main', 'breakfast-base', 'breakfast-side',
                  'light-standalone', 'protein', 'snack', 'drink'},
    'Lunch':     {'standalone', 'rice-main', 'needs-soup', 'needs-swallow',
                  'light-standalone', 'peppersoup', 'protein', 'side', 'drink'},
    'Dinner':    {'standalone', 'rice-main', 'needs-soup', 'needs-swallow',
                  'light-standalone', 'peppersoup', 'protein', 'side', 'drink'},
}


# ============================================================
# SAFETY CONSTRAINTS
# ============================================================

# Same-ingredient groups: at most 1 from each group (within a meal)
# Also used by engine.py for within-day sibling dedup across meals.
SAME_BASE_GROUPS = [
    {"Akara", "Moi-Moi (Plain)", "Moi-Moi (Egg/Fish)", "Beans (Ewa Aganyin)",
     "Beans Porridge", "Ekuru (White Moi-Moi)", "Ewa Ibeji (Beans & Corn)",
     "Gbegiri Soup", "Adalu (Beans & Corn)", "Bread & Beans",
     "Dan Wake (Bean Dumplings)", "White Rice & Beans", "Moi-Moi & Pap Combo"},
    {"Fried Plantain (Dodo)", "Boiled Plantain", "Roasted Plantain (Boli)",
     "Plantain Chips", "Fried Plantain & Egg (Dodo & Egg)"},
    {"Fried Yam (Dundun)", "Boiled Yam & Egg", "Yam Pottage (Asaro)",
     "Yam & Palmoil Sauce", "Ikokore (Water Yam Pottage)",
     "Dundun & Egg Sauce", "Yam Pepper Soup"},
    # Egg group: prevents Bread & Egg + Fried Egg style redundancy
    {"Boiled Egg", "Fried Egg", "Scrambled Eggs", "Vegetable Omelette",
     "Egg Sauce (Ata Dindin)", "Bread & Egg", "Indomie + Egg",
     "Boiled Yam & Egg", "Dundun & Egg Sauce", "Ogi & Akara Combo",
     "Fried Plantain & Egg (Dodo & Egg)"},
    # Noodle group: Indomie variants are the same base food
    {"Indomie (2 Packs)", "Indomie + Egg"},
    # Pasta group
    {"Spaghetti Jollof", "Macaroni & Stew"},
    # Oatmeal group
    {"Oatmeal (Plain)", "Oatmeal & Banana"},
    # Bread group
    {"Bread", "Bread & Egg", "Bread & Beans", "Tea & Bread"},
    # Peppersoup group
    {"Peppersoup (Fish)", "Peppersoup (Goat)", "Yam Pepper Soup"},
    # Chicken group
    {"Chicken (Fried)", "Chicken (Grilled/Peppered)"},
]

# Breakfast-only proteins — excluded from Lunch/Dinner
BREAKFAST_PROTEINS = {
    "Egg Sauce (Ata Dindin)", "Scrambled Eggs", "Vegetable Omelette",
}


class MealOptimizer:
    """
    Uses PuLP Integer Linear Programming to select optimal food combinations.

    Cultural pairing is enforced via the Pair_Category column in the food
    database. Each food's Pair_Category determines what it can and cannot
    be combined with, using the COMPATIBLE_PAIRS matrix above.
    """

    def optimize_meal(
        self,
        scored_foods: pd.DataFrame,
        target_calories: float,
        budget: float,
        meal_type: str,
        last_meals: list = None,
        calorie_tolerance: float = 250,
        today_meals: list = None,
    ) -> list[dict] | None:
        """
        Select the optimal combination of foods for a single meal.

        Args:
            scored_foods: DataFrame with 'ml_score' and 'Pair_Category' columns.
            target_calories: Target calorie intake for this meal.
            budget: Maximum cost for this meal in Naira.
            meal_type: "Breakfast", "Lunch", or "Dinner".
            last_meals: List of recently eaten food names (for variety).
            calorie_tolerance: Acceptable deviation from target (default 150 kcal).
            today_meals: List of food names already selected earlier today.

        Returns:
            List of food dicts, or None if no feasible solution.
        """
        if last_meals is None:
            last_meals = []
        if today_meals is None:
            today_meals = []

        # --- FILTER CANDIDATES ---
        allowed_cats = ALLOWED_PAIR_CATS.get(meal_type, ALLOWED_PAIR_CATS['Lunch'])

        candidates = scored_foods[
            (scored_foods['Pair_Category'].isin(allowed_cats)) &
            (scored_foods['Price_Avg_Naira'] <= budget)
        ].copy()

        # Exclude breakfast-only proteins from Lunch/Dinner
        if meal_type in ("Lunch", "Dinner"):
            candidates = candidates[~candidates['Name'].isin(BREAKFAST_PROTEINS)]

        # Day-level deduplication: hard-exclude foods already eaten today
        if today_meals:
            candidates = candidates[~candidates['Name'].isin(today_meals)]

        # Variety: heavily penalize recently eaten foods across the week
        if last_meals:
            recently_eaten = candidates['Name'].isin(last_meals)
            candidates.loc[recently_eaten, 'ml_score'] *= 0.05

        if candidates.empty:
            return None

        # Quality filter: prefer high-confidence items, fallback if too few
        MIN_ML_SCORE = 0.35
        high_quality = candidates[candidates['ml_score'] >= MIN_ML_SCORE]
        if len(high_quality) >= 4:
            candidates = high_quality

        candidates = candidates.reset_index(drop=True)
        n = len(candidates)
        pair_cats = candidates['Pair_Category'].values

        # --- BUILD LP ---
        prob = LpProblem(f"MealOptimizer_{meal_type}", LpMaximize)
        x = [LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

        cal_over = LpVariable("cal_over", lowBound=0)
        cal_under = LpVariable("cal_under", lowBound=0)

        # OBJECTIVE: Maximize ML score, penalize calorie deviation
        ml_scores = candidates['ml_score'].values
        prob += (
            lpSum(ml_scores[i] * x[i] for i in range(n))
            - 0.002 * cal_over
            - 0.002 * cal_under
        )

        # --- CONSTRAINTS ---

        # 1. Budget (hard)
        prices = candidates['Price_Avg_Naira'].values
        prob += lpSum(prices[i] * x[i] for i in range(n)) <= budget

        # 2. Calorie target (deviation bounded)
        calories = candidates['Calories_kcal'].values
        total_cal = lpSum(calories[i] * x[i] for i in range(n))
        prob += total_cal - target_calories <= cal_over
        prob += target_calories - total_cal <= cal_under
        prob += cal_over <= calorie_tolerance
        prob += cal_under <= calorie_tolerance

        # 3. Meal size: 1–2 items (Breakfast), 1–3 items (Lunch/Dinner)
        max_items = 3 if meal_type in ('Lunch', 'Dinner') else 2
        prob += lpSum(x[i] for i in range(n)) >= 1
        prob += lpSum(x[i] for i in range(n)) <= max_items

        # 4. At least 1 "main" food (meal base)
        main_indices = [i for i in range(n) if pair_cats[i] in MAIN_PAIR_CATS]
        if main_indices:
            prob += lpSum(x[i] for i in main_indices) >= 1

        # 5. At most 1 item per Pair_Category
        unique_cats = set(pair_cats)
        for cat in unique_cats:
            cat_indices = [i for i in range(n) if pair_cats[i] == cat]
            if len(cat_indices) > 1:
                prob += lpSum(x[i] for i in cat_indices) <= 1

        # 6. Pair_Category compatibility
        # For each pair of incompatible categories present in candidates,
        # prevent simultaneous selection.
        cat_list = list(unique_cats)
        for a_idx in range(len(cat_list)):
            for b_idx in range(a_idx + 1, len(cat_list)):
                cat_a = cat_list[a_idx]
                cat_b = cat_list[b_idx]
                compat_a = COMPATIBLE_PAIRS.get(cat_a, set())
                compat_b = COMPATIBLE_PAIRS.get(cat_b, set())
                if cat_b not in compat_a and cat_a not in compat_b:
                    # These categories are incompatible — block co-selection
                    a_indices = [i for i in range(n) if pair_cats[i] == cat_a]
                    b_indices = [i for i in range(n) if pair_cats[i] == cat_b]
                    if a_indices and b_indices:
                        prob += (
                            lpSum(x[i] for i in a_indices) +
                            lpSum(x[j] for j in b_indices)
                        ) <= 1

        # 7. Swallow ↔ Soup bidirectional pairing
        # needs-soup = swallows (they need a soup); needs-swallow = soups
        swallow_indices = [i for i in range(n) if pair_cats[i] == 'needs-soup']
        soup_indices = [i for i in range(n) if pair_cats[i] == 'needs-swallow']

        if swallow_indices and soup_indices:
            # If a swallow is selected, at least 1 soup must be selected
            for si in swallow_indices:
                prob += x[si] <= lpSum(x[j] for j in soup_indices)
            # If a soup is selected, at least 1 swallow must be selected
            for sj in soup_indices:
                prob += x[sj] <= lpSum(x[i] for i in swallow_indices)
        elif swallow_indices and not soup_indices:
            # No soups available → swallows cannot be selected
            for si in swallow_indices:
                prob += x[si] == 0
        elif soup_indices and not swallow_indices:
            # No swallows available → soups cannot be selected
            for sj in soup_indices:
                prob += x[sj] == 0

        # 8. Same-base-group exclusion (safety net)
        for group in SAME_BASE_GROUPS:
            group_indices = [
                i for i in range(n)
                if candidates.iloc[i]['Name'] in group
            ]
            if len(group_indices) > 1:
                prob += lpSum(x[i] for i in group_indices) <= 1

        # 9. At most 1 oily item
        oily_indices = [
            i for i in range(n)
            if 'oily' in str(candidates.iloc[i].get('Tags', '')).lower()
        ]
        if len(oily_indices) > 1:
            prob += lpSum(x[i] for i in oily_indices) <= 1

        # --- SOLVE ---
        solver = PULP_CBC_CMD(msg=0, timeLimit=5)
        status = prob.solve(solver)

        if LpStatus[status] != "Optimal":
            return None

        # --- EXTRACT SOLUTION ---
        selected = []
        for i in range(n):
            if x[i].varValue and x[i].varValue > 0.5:
                row = candidates.iloc[i]
                selected.append({
                    'Name': row['Name'],
                    'Calories': int(row['Calories_kcal']),
                    'Price': int(row['Price_Avg_Naira']),
                    'Category': row['Category'],
                    'ml_score': round(float(row['ml_score']), 3),
                })

        return selected if selected else None


# --- Standalone test ---
if __name__ == "__main__":
    from pathlib import Path
    from food_scorer import FoodScorer

    csv_path = Path(__file__).parent.parent / "data" / "nigerian_food_nutrition.csv"
    df = pd.read_csv(csv_path)

    scorer = FoodScorer(df)
    optimizer = MealOptimizer()

    for meal_type in ["Breakfast", "Lunch", "Dinner"]:
        print(f"\n{'='*60}")
        print(f"  Optimized {meal_type} (Target: 600 kcal, Budget: N2000)")
        print(f"{'='*60}")

        scored = scorer.score_foods(target_calories=600, meal_type=meal_type)
        result = optimizer.optimize_meal(scored, 600, 2000, meal_type)

        if result:
            total_cal = sum(item['Calories'] for item in result)
            total_cost = sum(item['Price'] for item in result)
            for item in result:
                print(f"  {item['Name']:30s} | {item['Calories']} kcal | N{item['Price']} | ML: {item['ml_score']}")
            print(f"  {'TOTAL':30s} | {total_cal} kcal | N{total_cost}")
        else:
            print("  No feasible solution found.")
