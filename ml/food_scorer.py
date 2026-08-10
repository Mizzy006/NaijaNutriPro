"""
food_scorer.py — Content-Based Food Recommender.

This module implements the ML component of the hybrid recommender system:
1. Encodes each food as a nutritional feature vector
2. Normalizes features using StandardScaler
3. Scores foods against a user's nutritional needs using cosine similarity
4. Provides food similarity lookups for swap recommendations

This is the "content-based filtering" part of the hybrid ML system.
The "constraint optimization" part is in optimizer.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Nutritional features used for ML scoring
NUTRITION_FEATURES = [
    'Calories_kcal', 'Protein_g', 'Carbs_g', 'Fat_g', 'Fiber_g',
    'Iron_mg', 'Calcium_mg', 'Vitamin_A_mcg', 'Vitamin_C_mg', 'Sodium_mg'
]


class FoodScorer:
    """
    Content-based food scoring and similarity engine.
    
    Encodes foods as numerical feature vectors and uses cosine similarity
    to rank foods by how well they match a user's nutritional needs.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the scorer by fitting the scaler on the food database.
        
        Args:
            df: DataFrame with columns matching NUTRITION_FEATURES.
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        
        # Extract and normalize the feature matrix
        # Fill any missing columns with 0
        for col in NUTRITION_FEATURES:
            if col not in self.df.columns:
                self.df[col] = 0.0
        
        self.feature_matrix = self.df[NUTRITION_FEATURES].fillna(0).values
        self.scaled_features = self.scaler.fit_transform(self.feature_matrix)
        
        # Pre-compute the food similarity matrix
        self._similarity_matrix = cosine_similarity(self.scaled_features)
        
        # Build name-to-index lookup
        self._name_to_idx = {
            name: idx for idx, name in enumerate(self.df['Name'].values)
        }

    def score_foods(
        self,
        target_calories: float,
        target_protein: float = None,
        meal_type: str = "Lunch",
        health_constraints: dict = None,
        category_filter: list = None,
        day_index: int = 0,
    ) -> pd.DataFrame:
        """
        Score all foods based on how well they match the user's nutritional needs.
        
        Args:
            target_calories: Target calorie intake for this meal.
            target_protein: Target protein intake (optional; estimated if None).
            meal_type: One of "Breakfast", "Lunch", "Dinner".
            health_constraints: Dict of nutrient limits, e.g.,
                                {'Sodium_mg': 600, 'max_sugar': True}
            category_filter: List of allowed food categories.
            day_index: Day of the week (0=Monday, 6=Sunday). Used to introduce
                       deterministic variety so different days produce different
                       top-ranked foods.
        
        Returns:
            DataFrame with original columns plus 'ml_score' (0-1, higher = better fit).
        """
        # Build the ideal meal vector for this user
        if target_protein is None:
            # Estimate: ~25% of meal calories from protein, protein = 4 cal/g
            target_protein = (target_calories * 0.25) / 4

        # Ideal nutrient profile for this meal
        ideal_vector = np.array([
            target_calories,
            target_protein,
            target_calories * 0.5 / 4,   # ~50% cals from carbs (carb = 4 cal/g)
            target_calories * 0.25 / 9,   # ~25% cals from fat (fat = 9 cal/g)
            8.0,                           # target 8g fiber per meal
            6.0,                           # target 6mg iron per meal
            300.0,                         # target 300mg calcium per meal
            250.0,                         # target 250mcg Vitamin A per meal
            25.0,                          # target 25mg Vitamin C per meal
            500.0,                         # target sodium limit per meal
        ]).reshape(1, -1)

        # Scale the ideal vector using the fitted scaler
        ideal_scaled = self.scaler.transform(ideal_vector)

        # Compute cosine similarity between each food and the ideal
        similarities = cosine_similarity(self.scaled_features, ideal_scaled).flatten()

        # Normalize to 0-1 range
        min_sim = similarities.min()
        max_sim = similarities.max()
        if max_sim > min_sim:
            scores = (similarities - min_sim) / (max_sim - min_sim)
        else:
            scores = np.ones_like(similarities) * 0.5

        # --- DAY-AWARE VARIETY PERTURBATION ---
        # Two-layer perturbation for maximum variety:
        # 1. Hash-based ±20%: deterministic day-to-day variety (same inputs → same ranking per day)
        # 2. Random ±15%: generation-to-generation variety (each run produces different plans)
        import hashlib
        import random
        perturbations = np.zeros(len(scores))
        for idx, name in enumerate(self.df['Name'].values):
            hash_input = f"{name}_{day_index}_{meal_type}"
            hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
            hash_perturb = (hash_val / 0xFFFFFFFF) * 0.40 - 0.20    # ±20%
            rand_perturb = random.uniform(-0.15, 0.15)                # ±15%
            perturbations[idx] = hash_perturb + rand_perturb
        scores = np.clip(scores + perturbations, 0.0, 1.0)

        # Add scores to dataframe
        result = self.df.copy()
        result['ml_score'] = scores

        # --- MEAL-TYPE TAG BOOST ---
        # Foods whose Tags mention the current meal time get a relevance boost.
        # This ensures breakfast-tagged foods rank higher at breakfast, etc.
        meal_tag_map = {'Breakfast': 'breakfast', 'Lunch': 'lunch', 'Dinner': 'dinner'}
        current_tag = meal_tag_map.get(meal_type, '')
        if current_tag and 'Tags' in result.columns:
            has_tag = result['Tags'].str.contains(current_tag, case=False, na=False)
            result.loc[has_tag, 'ml_score'] = (result.loc[has_tag, 'ml_score'] * 1.30).clip(upper=1.0)

        # Apply category filter
        if category_filter:
            result = result[result['Category'].isin(category_filter)]

        # Apply health constraints as penalty factors
        if health_constraints:
            result = self._apply_health_penalties(result, health_constraints)

        return result.sort_values('ml_score', ascending=False)

    def _apply_health_penalties(self, df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
        """
        Apply health-based penalties to ML scores.
        
        Foods that violate health constraints get their scores reduced.
        """
        df = df.copy()

        # Sodium limit (for hypertension)
        if 'max_sodium' in constraints:
            max_na = constraints['max_sodium']
            over_sodium = df['Sodium_mg'] > max_na
            df.loc[over_sodium, 'ml_score'] *= 0.3  # Heavy penalty

        # Sugar avoidance (for diabetes)
        if constraints.get('avoid_sugar', False):
            sugar_tags = df['Tags'].str.contains('sugar', case=False, na=False)
            df.loc[sugar_tags, 'ml_score'] *= 0.2  # Very heavy penalty

        # Spicy avoidance (for ulcers)
        if constraints.get('avoid_spicy', False):
            spicy_tags = df['Tags'].str.contains('spicy', case=False, na=False)
            df.loc[spicy_tags, 'ml_score'] *= 0.3

        # Oily avoidance (for ulcers, heart conditions)
        if constraints.get('avoid_oily', False):
            oily_tags = df['Tags'].str.contains('oily', case=False, na=False)
            df.loc[oily_tags, 'ml_score'] *= 0.4

        # Prefer health-tagged foods
        if 'prefer_tags' in constraints:
            for tag in constraints['prefer_tags']:
                has_tag = df['Health_Tags'].str.contains(tag, case=False, na=False)
                df.loc[has_tag, 'ml_score'] *= 1.3  # Boost

        return df

    def get_similar_foods(self, food_name: str, top_n: int = 5) -> list[dict]:
        """
        Find culturally compatible foods with similar nutritional profiles.
        
        Filters candidates to the same Pair_Category first (e.g., soup → soups,
        swallow → swallows), then ranks by cosine similarity on nutrition.
        Falls back to same Category if too few Pair_Category matches exist.
        
        Args:
            food_name: Name of the food to find alternatives for.
            top_n: Number of similar foods to return.
        
        Returns:
            List of dicts with 'name', 'similarity_score', 'calories', 'price'.
        """
        if food_name not in self._name_to_idx:
            # Try partial match
            matches = [n for n in self._name_to_idx if food_name.lower() in n.lower()]
            if not matches:
                return []
            food_name = matches[0]

        idx = self._name_to_idx[food_name]
        source_row = self.df.iloc[idx]
        similarities = self._similarity_matrix[idx]

        # Get all indices sorted by similarity (excluding self)
        all_sorted = np.argsort(similarities)[::-1]
        all_sorted = [i for i in all_sorted if i != idx]

        # --- CULTURAL FILTERING ---
        # Only return alternatives that serve the same role in a meal
        source_pair_cat = source_row.get('Pair_Category', '')
        
        if source_pair_cat:
            # Primary: same Pair_Category (soup→soups, swallow→swallows)
            culturally_valid = [
                i for i in all_sorted
                if self.df.iloc[i].get('Pair_Category', '') == source_pair_cat
            ]
            # Fallback: same Category if too few Pair_Category matches
            if len(culturally_valid) < top_n:
                source_cat = source_row.get('Category', '')
                same_cat_extras = [
                    i for i in all_sorted
                    if self.df.iloc[i].get('Category', '') == source_cat
                    and i not in culturally_valid
                ]
                culturally_valid.extend(same_cat_extras)
            similar_indices = culturally_valid[:top_n]
        else:
            similar_indices = all_sorted[:top_n]

        results = []
        for i in similar_indices:
            row = self.df.iloc[i]
            results.append({
                'name': row['Name'],
                'similarity_score': round(float(similarities[i]), 3),
                'calories': row['Calories_kcal'],
                'price': row['Price_Avg_Naira'],
                'category': row['Category'],
            })

        return results


# --- Standalone test ---
if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "data" / "nigerian_food_nutrition.csv"
    df = pd.read_csv(csv_path)

    scorer = FoodScorer(df)

    # Test scoring
    print("=== Top 10 foods for a 600 kcal lunch ===")
    scored = scorer.score_foods(target_calories=600, meal_type="Lunch")
    for _, row in scored.head(10).iterrows():
        print(f"  {row['Name']:30s} | Score: {row['ml_score']:.3f} | {row['Calories_kcal']} kcal | N{row['Price_Avg_Naira']}")

    print("\n=== Foods similar to Jollof Rice ===")
    similar = scorer.get_similar_foods("Jollof Rice", top_n=5)
    for s in similar:
        print(f"  {s['name']:30s} | Similarity: {s['similarity_score']:.3f} | {s['calories']} kcal")

    print("\n=== Diabetic-friendly scoring ===")
    scored_diabetic = scorer.score_foods(
        target_calories=500,
        health_constraints={'avoid_sugar': True, 'prefer_tags': ['diabetic-friendly']},
    )
    for _, row in scored_diabetic.head(10).iterrows():
        print(f"  {row['Name']:30s} | Score: {row['ml_score']:.3f} | {row['Calories_kcal']} kcal")
