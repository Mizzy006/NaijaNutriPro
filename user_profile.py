"""
user_profile.py — User Biometrics and Health Profile Calculator.

Calculates BMR (Basal Metabolic Rate), TDEE (Total Daily Energy Expenditure),
and calorie targets using the Mifflin-St Jeor equation.

Enhanced with health condition awareness for the hybrid ML recommender.
"""


def calculate_biometrics(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity_level: str,
    goal: str,
    health_conditions: list = None,
) -> dict:
    """
    Calculates the Total Daily Energy Expenditure (TDEE) and Calorie Target.
    
    Args:
        weight_kg: User's weight in kilograms.
        height_cm: User's height in centimeters.
        age: User's age in years.
        gender: "male" or "female".
        activity_level: "sedentary", "light", "moderate", or "active".
        goal: "maintain_weight", "lose_weight", or "gain_weight".
        health_conditions: Optional list of health conditions.
    
    Returns:
        Dict with BMR, TDEE, Target_Calories, and nutrient constraints.
    """
    # 1. Calculate BMR (Mifflin-St Jeor Equation)
    if gender.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    # 2. Activity Multipliers
    multipliers = {
        "sedentary": 1.2,      # Desk job / Student sitting in class
        "light": 1.375,        # Walking around campus
        "moderate": 1.55,      # Gym/Sports 3-5 days
        "active": 1.725        # Athlete
    }

    # Default to sedentary if input is wrong
    multiplier = multipliers.get(activity_level.lower(), 1.2)
    tdee = int(bmr * multiplier)

    # 3. Adjust for Goal
    if goal == "lose_weight":
        target_calories = tdee - 500  # Safe deficit
    elif goal == "gain_weight":
        target_calories = tdee + 500  # Surplus
    else:
        target_calories = tdee  # Maintenance

    # 4. Pregnancy adjustment
    if health_conditions and "Pregnancy" in health_conditions:
        target_calories += 300  # Additional calories for pregnancy

    # 5. Build result
    result = {
        "BMR": int(bmr),
        "TDEE": tdee,
        "Target_Calories": target_calories,
    }

    # 6. Add nutrient constraints based on health conditions
    if health_conditions:
        constraints = {}
        for condition in health_conditions:
            if condition == "Diabetes":
                constraints["max_daily_carbs_g"] = 200
                constraints["max_daily_sugar_g"] = 25
                constraints["min_daily_fiber_g"] = 25
            elif condition == "Hypertension":
                constraints["max_daily_sodium_mg"] = 1500
                constraints["min_daily_potassium_mg"] = 3500
            elif condition == "Pregnancy":
                constraints["min_daily_iron_mg"] = 27
                constraints["min_daily_calcium_mg"] = 1000
                constraints["min_daily_protein_g"] = 71
            elif condition == "Ulcer":
                constraints["avoid_spicy"] = True
                constraints["avoid_acidic"] = True
                constraints["meal_frequency"] = "5-6 small meals"
        
        result["health_constraints"] = constraints

    return result