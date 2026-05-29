import pandas as pd
import random

# PASTRY CHECK 
def is_pastry(food_name):
    """Checks if a food is dough-based to prevent double pastries."""
    # List of items that shouldn't be eaten together
    pastries = ["Puff Puff", "Egg Roll", "Meat Pie", "Buns", "Donut", "Chin Chin", "Bread", "Gala"]
    return any(p in food_name for p in pastries)
# --- CONFLICT RULES ---

# Tags that should appear at most ONCE in a combo
EXCLUSIVE_TAGS = {"oily", "sugar"}

# Food families — only ONE item from each group per combo
SAME_BASE_GROUPS = [
    {"Akara", "Moi-Moi (Plain)", "Moi-Moi (Egg/Fish)", "Beans (Ewa Aganyin)", "Beans Porridge"},
    {"Fried Plantain (Dodo)", "Boiled Plantain"},
    {"Fried Yam (Dundun)", "Boiled Yam & Egg", "Yam Pottage (Asaro)"},
]

def has_conflict(combo, candidate_row, df):
    """
    Returns True if adding candidate_row to the combo creates a bad pairing.
    Checks:
      1. Exclusive tags — e.g. don't combine two 'oily' items
      2. Same-base groups — e.g. don't combine Akara with Moi-Moi (both beans)
    """
    candidate_name = candidate_row['Name']
    candidate_tags = set(str(candidate_row['Tags']).split(';'))
    combo_names = [item['Name'] for item in combo]

    for item_name in combo_names:
        item_rows = df[df['Name'] == item_name]
        if item_rows.empty:
            continue
        item_tags = set(str(item_rows.iloc[0]['Tags']).split(';'))

        # Rule 1: Block if they share an exclusive tag (e.g. both oily)
        if candidate_tags & item_tags & EXCLUSIVE_TAGS:
            return True

    # Rule 2: Block if candidate shares a base-ingredient group with anything in combo
    for group in SAME_BASE_GROUPS:
        if candidate_name in group:
            if any(name in group for name in combo_names):
                return True
    # Rule 4: Mutual exclusion — oily items and pastries don't belong together
    candidate_is_pastry = is_pastry(candidate_name)
    candidate_is_oily = "oily" in candidate_tags

    for item_name in combo_names:
        item_rows = df[df['Name'] == item_name]
        if item_rows.empty:
            continue
        item_tags = set(str(item_rows.iloc[0]['Tags']).split(';'))

        if candidate_is_pastry and "oily" in item_tags:
            return True  # Candidate is pastry, combo already has oily item

        if candidate_is_oily and is_pastry(item_name):
            return True  # Candidate is oily, combo already has a pastry

    return False


    # Rule 3: Block light oily snacks when a heavy main is already in the combo
    HEAVY_MAINS = {
        "Beans Porridge", "Beans (Ewa Aganyin)", "Pounded Yam (Iyan)",
        "Eba (Garri)", "Amala (Yam Flour)", "Fufu (Akpu)", "Semo (Semovita)",
        "Tuwo Shinkafa", "Wheat Meal", "Jollof Rice", "Fried Rice",
        "Yam Pottage (Asaro)", "Fried Yam (Dundun)"
    }
    LIGHT_SNACKS_ONLY = {"Puff Puff", "Gala (Sausage Roll)", "Chin Chin"}

    if candidate_name in LIGHT_SNACKS_ONLY:
        if any(name in HEAVY_MAINS for name in combo_names):
            return True

    return False
# 1. EMERGENCY FOOD (Last Resort) 
def get_emergency_food(df, available_budget):
    candidates = df[
        (df['Price_Avg_Naira'] <= available_budget) & 
        (~df['Category'].isin(["Drink", "Spice", "Ingredient", "Soup"])) 
    ]
    
    if candidates.empty:
        return None 

    best_item = candidates.sort_values('Calories_kcal', ascending=False).iloc[0]
    
    return [{
        "Name": best_item['Name'] + " (Budget Saver)", 
        "Calories": best_item['Calories_kcal'],
        "Price": best_item['Price_Avg_Naira'],
        "Category": "Emergency"
    }]

# 2. SMART COMBINATION ENGINE 
def get_smart_combination(df, target_cals, available_budget, meal_type, last_meals=[]):
    
    # --- DEFINE CATEGORIES ---
    if meal_type == "Breakfast":
        main_cats = ["Breakfast", "Carb", "Noodle", "Rice", "Beans"]
    elif meal_type == "Lunch":
        main_cats = ["Rice", "Swallow", "Beans", "Pasta", "Tuber", "Light Main"]
    else: # Dinner
        main_cats = ["Swallow", "Rice", "Tuber", "Light", "Beans", "Light Main"] 

    # Filter Candidates
    candidates = df[
        (df['Category'].isin(main_cats)) & 
        (df['Price_Avg_Naira'] <= available_budget)
    ]
    
    if candidates.empty: return None 

    # Variety Check
    if len(candidates) > 4:
        candidates = candidates[~candidates['Name'].isin(last_meals)]

    # MAIN DISH SELECTION
    if target_cals > 800: 
        perfect_matches = candidates[candidates['Calories_kcal'] >= target_cals * 0.5]
    else:
        perfect_matches = candidates[
            (candidates['Calories_kcal'] >= target_cals * 0.4) &
            (candidates['Calories_kcal'] <= target_cals * 1.0)
        ]
    
    if not perfect_matches.empty:
        main_dish = perfect_matches.sample(1).iloc[0]
    else:
        candidates = candidates.copy()
        candidates['diff'] = abs(candidates['Calories_kcal'] - target_cals)
        main_dish = candidates.sort_values('diff').head(3).sample(1).iloc[0]

    # Init Combo
    combo = [{
        "Name": main_dish['Name'], "Calories": main_dish['Calories_kcal'], 
        "Price": main_dish['Price_Avg_Naira'], "Category": main_dish['Category']
    }]
    current_cost = main_dish['Price_Avg_Naira']
    current_cals = main_dish['Calories_kcal']
    remaining_budget = available_budget - current_cost

    # PAIRING LOGIC (Mandatory)
    mandatory_pairing = False
    pair_category = []
    if main_dish['Category'] == "Swallow":
        mandatory_pairing = True
        pair_category = ["Soup"]
    elif main_dish['Name'] == "Bread":
        pair_category = ["Beans", "Side", "Drink"] 
    elif main_dish['Category'] in ["Rice", "Pasta", "Tuber"]:
        pair_category = ["Protein", "Side", "Meat"]

    # FILLER LOGIC (Sides) 
    if remaining_budget >= 50:
        if pair_category:
            side_candidates = df[df['Category'].isin(pair_category)]
        else:
            side_candidates = df[df['Category'].isin(["Protein", "Side", "Drink", "Snack"])]
            
        affordable_sides = side_candidates[side_candidates['Price_Avg_Naira'] <= remaining_budget]
        
        if not affordable_sides.empty:
            if not mandatory_pairing:
                affordable_sides = affordable_sides[
                    (affordable_sides['Calories_kcal'] + current_cals) <= (target_cals + 100)
                ]
            
            # --- If ANY item in combo is a pastry, block all other pastries ---
            if any(is_pastry(item['Name']) for item in combo):
                affordable_sides = affordable_sides[~affordable_sides['Name'].apply(is_pastry)]
            # --- Filter out conflicting sides (oily+oily, same-base, etc.) ---
            affordable_sides = affordable_sides[
                ~affordable_sides.apply(lambda row: has_conflict(combo, row, df), axis=1)
            ]

            if not affordable_sides.empty:
                affordable_sides = affordable_sides.copy()
                affordable_sides['cal_diff'] = abs((current_cals + affordable_sides['Calories_kcal']) - target_cals)
                best_side = affordable_sides.sort_values('cal_diff').iloc[0]
                
                if best_side['Name'] not in [x['Name'] for x in combo]:
                    combo.append({
                        "Name": best_side['Name'], "Calories": best_side['Calories_kcal'], 
                        "Price": best_side['Price_Avg_Naira']
                    })
                    current_cost += best_side['Price_Avg_Naira']
                    current_cals += best_side['Calories_kcal']
                    remaining_budget -= best_side['Price_Avg_Naira']

    # FILLER LOOP (Extras) 
    loop_safety = 0
    while remaining_budget > 100 and current_cals < (target_cals - 50) and loop_safety < 3:
        
        extras = df[
            (df['Category'].isin(["Drink", "Protein", "Snack", "Side"])) & 
            (df['Price_Avg_Naira'] <= remaining_budget) &
            (~df['Category'].isin(["Soup"])) 
        ]
        
        if extras.empty: break 
        
        valid_extras = extras[
            (extras['Calories_kcal'] + current_cals) <= (target_cals + 100)
        ]
        

        # Check if we ALREADY have a pastry in the combo
        current_names = [x['Name'] for x in combo]
        has_pastry = any(is_pastry(n) for n in current_names)
        
        # If we have a pastry, filter out all pastries from the extras list
        if has_pastry:
            valid_extras = valid_extras[~valid_extras['Name'].apply(is_pastry)]
        # --- Filter out conflicting extras (oily+oily, same-base, etc.) ---
        valid_extras = valid_extras[
            ~valid_extras.apply(lambda row: has_conflict(combo, row, df), axis=1)
        ]
        
        if valid_extras.empty: break
            
        possible_extras = valid_extras.sort_values('Calories_kcal', ascending=False).head(5)
        unique_extras = possible_extras[~possible_extras['Name'].isin(current_names)]
        
        if unique_extras.empty: break
            
        extra_item = unique_extras.sample(1).iloc[0]
        
        combo.append({
            "Name": extra_item['Name'],
            "Calories": extra_item['Calories_kcal'],
            "Price": extra_item['Price_Avg_Naira']
        })
        
        current_cost += extra_item['Price_Avg_Naira']
        current_cals += extra_item['Calories_kcal']
        remaining_budget -= extra_item['Price_Avg_Naira']
        loop_safety += 1

    # VALIDATION
    if mandatory_pairing and len(combo) < 2: return None 
    
    return combo

#  WEEKLY GENERATOR 
def generate_weekly_plan(target_calories_daily, daily_budget_limit, dietary_preference="None"):
    try:
        df = pd.read_csv("nigerian_food_nutrition.csv")
    except:
        return "Error: Data not found."

    # ==========================================
    # NEW LOGIC: DIETARY FILTERING
    # ==========================================
    if dietary_preference != "None":
        # Define the forbidden keywords for each diet
        if dietary_preference == "Vegetarian":
            forbidden_words = ["meat", "fish", "chicken", "beef", "pork", "poultry", "turkey", "ponmo", "snail"]
        elif dietary_preference == "Pescatarian":
            forbidden_words = ["meat", "chicken", "beef", "pork", "poultry", "turkey", "ponmo"]
        elif dietary_preference == "Halal":
            forbidden_words = ["pork", "bacon", "ham", "alcohol"]
        else:
            forbidden_words = []

        if forbidden_words:
            # Create a regex pattern to match any forbidden word (case insensitive)
            pattern = '|'.join(forbidden_words)
            
            # Filter out rows where the Name, Category, or Tags contain forbidden words
            df = df[
                ~df['Name'].str.lower().str.contains(pattern, na=False) &
                ~df['Category'].str.lower().str.contains(pattern, na=False) &
                ~df['Tags'].str.lower().str.contains(pattern, na=False)
            ]

        # Safety check in case the filter removed every single food in the database!
        if df.empty:
            return f"Error: No foods available for the {dietary_preference} diet in the database."
    # ==========================================

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_plan = {}
    recent_meals = []
    current_wallet = 0
    
    for day in days:
        current_wallet += daily_budget_limit 
        daily_menu = {}
        daily_cost = 0
        daily_cal = 0
        meals = ["Breakfast", "Lunch", "Dinner"]
        
        for meal_type in meals:
            if meal_type == "Dinner":
                meal_budget_cap = current_wallet
            elif meal_type == "Lunch":
                meal_budget_cap = min(current_wallet, daily_budget_limit * 0.7)
            else:  # Breakfast
                meal_budget_cap = min(current_wallet, daily_budget_limit * 0.5)
            
            if meal_type == "Lunch": 
                meal_cal_target = target_calories_daily * 0.4
            else: 
                meal_cal_target = target_calories_daily * 0.3

            combo = get_smart_combination(df, meal_cal_target, meal_budget_cap, meal_type, last_meals=recent_meals[-4:])
            
            if not combo:
                 combo = get_smart_combination(df, meal_cal_target, meal_budget_cap, meal_type, last_meals=[])

            if not combo and current_wallet >= 150:
                 combo = get_emergency_food(df, current_wallet)

            if combo:
                recent_meals.append(combo[0]['Name'])
                if len(recent_meals) > 8: recent_meals.pop(0)
                
                meal_cost = sum(item['Price'] for item in combo)
                current_wallet -= meal_cost
                
                daily_menu[meal_type] = combo
                daily_cost += meal_cost
                daily_cal += sum(item['Calories'] for item in combo)
            else:
                daily_menu[meal_type] = [{"Name": "Skipped (Insufficient Funds)", "Calories": 0, "Price": 0}]

        weekly_plan[day] = {
            "Plan": daily_menu, "Total_Cost": daily_cost, 
            "Total_Calories": daily_cal, "Wallet_Balance": current_wallet
        }
        
    return weekly_plan