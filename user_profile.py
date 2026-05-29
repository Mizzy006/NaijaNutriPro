def calculate_biometrics(weight_kg, height_cm, age, gender, activity_level, goal):
    """
    Calculates the Total Daily Energy Expenditure (TDEE) and Calorie Target.
    """
    
    # Calculate BMR (Mifflin-St Jeor Equation)
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

    return {
        "BMR": int(bmr),
        "TDEE": tdee,
        "Target_Calories": target_calories
    }