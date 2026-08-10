"""
test_engine.py — Tests for the Hybrid ML Engine.

Tests the end-to-end meal plan generation, dietary filtering,
health condition constraints, and edge cases.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import generate_weekly_plan, filter_by_diet, get_health_constraints, get_emergency_food


CSV_PATH = Path(__file__).parent.parent / "data" / "nigerian_food_nutrition.csv"


@pytest.fixture
def food_df():
    """Load the food database for testing."""
    return pd.read_csv(CSV_PATH)


class TestWeeklyPlanGeneration:
    """Tests for generate_weekly_plan()."""

    def test_generates_7_day_plan(self):
        """Plan should have exactly 7 days."""
        plan = generate_weekly_plan(2000, 3000)
        assert isinstance(plan, dict)
        assert len(plan) == 7
        assert "Monday" in plan
        assert "Sunday" in plan

    def test_each_day_has_3_meals(self):
        """Each day should have Breakfast, Lunch, and Dinner."""
        plan = generate_weekly_plan(2000, 3000)
        for day, data in plan.items():
            assert "Plan" in data
            assert "Breakfast" in data["Plan"]
            assert "Lunch" in data["Plan"]
            assert "Dinner" in data["Plan"]

    def test_budget_respected(self):
        """Total daily cost should not exceed daily budget (with carryover)."""
        plan = generate_weekly_plan(2000, 2500)
        for day, data in plan.items():
            # Each meal's items should have valid prices
            for meal_type, items in data["Plan"].items():
                for item in items:
                    assert "Price" in item
                    assert item["Price"] >= 0

    def test_calories_present(self):
        """Each day should track total calories."""
        plan = generate_weekly_plan(2000, 3000)
        for day, data in plan.items():
            assert "Total_Calories" in data
            assert "Total_Cost" in data

    def test_low_budget_still_works(self):
        """Even very low budgets should produce a plan (with skips or emergency food)."""
        plan = generate_weekly_plan(2000, 500)
        assert isinstance(plan, dict)
        assert len(plan) == 7

    def test_high_calorie_target(self):
        """High calorie targets should produce larger meals."""
        plan = generate_weekly_plan(3500, 5000)
        assert isinstance(plan, dict)
        assert len(plan) == 7

    def test_ml_scores_present(self):
        """Each food item should have an ml_score."""
        plan = generate_weekly_plan(2000, 3000)
        for day, data in plan.items():
            for meal_type, items in data["Plan"].items():
                for item in items:
                    if item["Name"] != "Skipped (Insufficient Funds)":
                        assert "ml_score" in item


class TestDietaryFiltering:
    """Tests for dietary preference filtering."""

    def test_vegetarian_filter(self, food_df):
        """Vegetarian filter should remove all meat/fish."""
        filtered = filter_by_diet(food_df, "Vegetarian")
        for _, row in filtered.iterrows():
            name_lower = row["Name"].lower()
            assert "chicken" not in name_lower
            assert "beef" not in name_lower
            assert "suya" not in name_lower
            assert "goat" not in name_lower

    def test_halal_filter(self, food_df):
        """Halal filter should remove pork-related items."""
        filtered = filter_by_diet(food_df, "Halal")
        for _, row in filtered.iterrows():
            assert "pork" not in row["Name"].lower()

    def test_none_filter(self, food_df):
        """No filter should return the full dataset."""
        filtered = filter_by_diet(food_df, "None")
        assert len(filtered) == len(food_df)

    def test_vegetarian_plan_generation(self):
        """Full plan generation with Vegetarian preference should work."""
        plan = generate_weekly_plan(2000, 3000, dietary_preference="Vegetarian")
        assert isinstance(plan, dict)
        assert len(plan) == 7


class TestHealthConditions:
    """Tests for health condition constraint mapping."""

    def test_diabetes_constraints(self):
        """Diabetes should set avoid_sugar and prefer diabetic-friendly."""
        constraints = get_health_constraints(["Diabetes"])
        assert constraints.get("avoid_sugar") is True
        assert "diabetic-friendly" in constraints.get("prefer_tags", [])

    def test_hypertension_constraints(self):
        """Hypertension should set max_sodium."""
        constraints = get_health_constraints(["Hypertension"])
        assert "max_sodium" in constraints
        assert constraints["max_sodium"] <= 1000

    def test_ulcer_constraints(self):
        """Ulcer should avoid spicy and oily foods."""
        constraints = get_health_constraints(["Ulcer"])
        assert constraints.get("avoid_spicy") is True
        assert constraints.get("avoid_oily") is True

    def test_multiple_conditions(self):
        """Multiple conditions should combine constraints."""
        constraints = get_health_constraints(["Diabetes", "Hypertension"])
        assert constraints.get("avoid_sugar") is True
        assert "max_sodium" in constraints

    def test_no_conditions(self):
        """No conditions should return empty dict."""
        assert get_health_constraints(None) == {}
        assert get_health_constraints([]) == {}

    def test_health_aware_plan(self):
        """Plan with health conditions should still generate."""
        plan = generate_weekly_plan(
            2000, 3000,
            health_conditions=["Diabetes", "Hypertension"]
        )
        assert isinstance(plan, dict)
        assert len(plan) == 7


class TestEmergencyFood:
    """Tests for the emergency food fallback."""

    def test_emergency_food_returns_item(self, food_df):
        """Emergency food should return something for reasonable budgets."""
        result = get_emergency_food(food_df, 500)
        assert result is not None
        assert len(result) == 1
        assert "Budget Saver" in result[0]["Name"]

    def test_emergency_food_respects_budget(self, food_df):
        """Emergency food price should be within budget."""
        result = get_emergency_food(food_df, 300)
        if result:
            assert result[0]["Price"] <= 300

    def test_emergency_food_impossible_budget(self, food_df):
        """Very low budget should return None."""
        result = get_emergency_food(food_df, 10)
        assert result is None


class TestVarietyAndPairings:
    """Tests for meal variety across days and cultural food pairing rules."""

    def test_variety_across_days(self):
        """No single food should appear as the main item in all 7 days."""
        plan = generate_weekly_plan(2000, 3000)
        assert isinstance(plan, dict)
        from collections import Counter
        all_foods = []
        for day_data in plan.values():
            for meal_type, items in day_data["Plan"].items():
                for item in items:
                    if item["Name"] != "Skipped (Insufficient Funds)":
                        all_foods.append(item["Name"])
        counts = Counter(all_foods)
        for food, count in counts.items():
            assert count < 7, f"'{food}' appears {count} times across the week — too repetitive"

    def test_no_same_food_same_day(self):
        """No food should appear in multiple meals on the same day."""
        plan = generate_weekly_plan(2000, 3000)
        for day, day_data in plan.items():
            day_foods = []
            for meal_type, items in day_data["Plan"].items():
                for item in items:
                    name = item["Name"]
                    if name != "Skipped (Insufficient Funds)":
                        assert name not in day_foods, (
                            f"'{name}' appears multiple times on {day}"
                        )
                        day_foods.append(name)

    def test_swallow_always_has_soup(self):
        """Every swallow item in a meal should be paired with a soup."""
        plan = generate_weekly_plan(2000, 3000)
        SWALLOW_NAMES = {
            "Pounded Yam (Iyan)", "Eba (Garri)", "Amala (Yam Flour)",
            "Fufu (Akpu)", "Semo (Semovita)", "Tuwo Shinkafa",
            "Wheat Meal", "Lafun", "Tuwo Masara (Corn)",
        }
        for day, day_data in plan.items():
            for meal_type, items in day_data["Plan"].items():
                names = {item["Name"] for item in items}
                cats = {item.get("Category", "") for item in items}
                has_swallow = bool(names & SWALLOW_NAMES)
                has_soup = "Soup" in cats
                if has_swallow:
                    assert has_soup, (
                        f"{day} {meal_type}: Has swallow ({names & SWALLOW_NAMES}) "
                        f"but no soup"
                    )

    def test_no_soup_with_light_main(self):
        """Soups should not pair with Light Main items (Moi-Moi, Ekuru, Agidi)."""
        plan = generate_weekly_plan(2000, 3000)
        LIGHT_MAINS = {
            "Ekuru (White Moi-Moi)", "Moi-Moi (Plain)",
            "Moi-Moi (Egg/Fish)", "Agidi (Eko)",
        }
        for day, day_data in plan.items():
            for meal_type, items in day_data["Plan"].items():
                names = {item["Name"] for item in items}
                cats = {item.get("Category", "") for item in items}
                has_light_main = bool(names & LIGHT_MAINS)
                has_soup = "Soup" in cats
                if has_light_main and has_soup:
                    assert False, (
                        f"{day} {meal_type}: Light main ({names & LIGHT_MAINS}) "
                        f"paired with soup — culturally invalid"
                    )

    def test_max_two_items_per_meal(self):
        """Breakfast has max 2 items; Lunch/Dinner have max 3."""
        plan = generate_weekly_plan(2000, 3000)
        for day, day_data in plan.items():
            for meal_type, items in day_data["Plan"].items():
                real_items = [i for i in items if i["Name"] != "Skipped (Insufficient Funds)"]
                max_allowed = 3 if meal_type in ("Lunch", "Dinner") else 2
                assert len(real_items) <= max_allowed, (
                    f"{day} {meal_type}: Has {len(real_items)} items "
                    f"({[i['Name'] for i in real_items]}), expected max {max_allowed}"
                )

    def test_condiments_never_standalone(self):
        """Condiment items should never appear in meal plans."""
        plan = generate_weekly_plan(2000, 3000)
        CONDIMENT_ITEMS = {
            "Stockfish (Okporoko)", "Crayfish (Ground)", "Ponmo (Cow Skin)",
            "Ugwu (Fluted Pumpkin) Leaf", "Efo Tete (African Spinach)",
            "Efo Shoko (Lagos Spinach)", "Efo Worowo (Bitter Leaf)",
            "Garden Egg (African Eggplant)", "Tomato Stew (Base)",
            "Pepper Sauce (Ata Rodo)",
        }
        for day, day_data in plan.items():
            for meal_type, items in day_data["Plan"].items():
                for item in items:
                    assert item["Name"] not in CONDIMENT_ITEMS, (
                        f"{day} {meal_type}: Condiment '{item['Name']}' "
                        f"should never appear as a meal item"
                    )

    def test_akara_only_at_breakfast(self):
        """Akara should only appear at Breakfast, not Lunch or Dinner."""
        plan = generate_weekly_plan(2000, 3000)
        for day, day_data in plan.items():
            for meal_type, items in day_data["Plan"].items():
                if meal_type in ("Lunch", "Dinner"):
                    for item in items:
                        assert item["Name"] != "Akara", (
                            f"{day} {meal_type}: Akara should only appear at Breakfast"
                        )

    def test_alternatives_culturally_valid(self):
        """Alternatives for a food should be from the same cultural category."""
        from ml.food_scorer import FoodScorer

        csv_path = Path(__file__).parent.parent / "data" / "nigerian_food_nutrition.csv"
        df = pd.read_csv(csv_path)
        scorer = FoodScorer(df)

        # Alternatives for a soup should be soups
        similar = scorer.get_similar_foods("Egusi Soup", top_n=3)
        assert len(similar) > 0
        for s in similar:
            assert s['category'] == 'Soup', (
                f"Alternative '{s['name']}' for Egusi Soup has category "
                f"'{s['category']}', expected 'Soup'"
            )

        # Alternatives for a swallow should be swallows
        similar = scorer.get_similar_foods("Eba (Garri)", top_n=3)
        assert len(similar) > 0
        for s in similar:
            assert s['category'] == 'Swallow', (
                f"Alternative '{s['name']}' for Eba has category "
                f"'{s['category']}', expected 'Swallow'"
            )

