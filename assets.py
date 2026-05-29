# assets.py

def get_food_image(food_name):
    """
    Returns a URL for the given food name.
    If no specific image exists, returns a generic category image.
    """
    
    # Map specific foods to high-quality image URLs
    images = {
        # --- RICE & PASTA ---
        "Jollof Rice": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Jollof_Rice_with_Stew.jpg/640px-Jollof_Rice_with_Stew.jpg",
        "Fried Rice": "https://upload.wikimedia.org/wikipedia/commons/1/1e/Nigerian_fried_rice.jpg",
        "White Rice": "https://upload.wikimedia.org/wikipedia/commons/d/d7/Rice_and_stew.jpg",
        "Coconut Rice": "https://upload.wikimedia.org/wikipedia/commons/6/62/Coconut_Rice.jpg",
        "Concoction Rice": "https://upload.wikimedia.org/wikipedia/commons/0/0a/Jollof_Rice_with_Stew.jpg", # Fallback
        "Ofada Rice": "https://upload.wikimedia.org/wikipedia/commons/6/6f/Ofada_Rice_and_stew.jpg",
        "Spaghetti": "https://images.unsplash.com/photo-1598514982205-f36b96d1e8d4?q=80&w=600",
        
        # --- SWALLOWS ---
        "Pounded Yam": "https://upload.wikimedia.org/wikipedia/commons/a/af/Pounded_Yam_and_Egusi_Soup.jpg",
        "Eba": "https://upload.wikimedia.org/wikipedia/commons/3/32/Eba_and_Okro_soup.jpg",
        "Amala": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Amala_and_Ewedu_soup.jpg/640px-Amala_and_Ewedu_soup.jpg",
        "Fufu": "https://upload.wikimedia.org/wikipedia/commons/b/b2/Fufu_and_Light_Soup.jpg",
        "Semovita": "https://upload.wikimedia.org/wikipedia/commons/a/af/Pounded_Yam_and_Egusi_Soup.jpg", # Similar look
        "Tuwo Shinkafa": "https://upload.wikimedia.org/wikipedia/commons/6/6d/Tuwo_shinkafa.jpg",
        
        # --- SOUPS ---
        "Egusi Soup": "https://upload.wikimedia.org/wikipedia/commons/5/5a/Egusi_Soup.jpg",
        "Ewedu Soup": "https://upload.wikimedia.org/wikipedia/commons/0/05/Amala_dudu_ati_Ewedu_abula.jpg",
        "Okro Soup": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Okro_soup_with_Eba.jpg",
        "Ogbono Soup": "https://upload.wikimedia.org/wikipedia/commons/7/77/Ogbono_Soup.jpg",
        "Vegetable Soup": "https://upload.wikimedia.org/wikipedia/commons/0/02/Vegetable_Soup_with_Ugu_and_Waterleaf.jpg",
        "Banga Soup": "https://upload.wikimedia.org/wikipedia/commons/a/a2/Banga_Soup_and_Starch.jpg",
        "Pepper Soup": "https://upload.wikimedia.org/wikipedia/commons/5/53/Catfish_pepper_soup.jpg",
        
        # --- BREAKFAST & SNACKS ---
        "Akara": "https://upload.wikimedia.org/wikipedia/commons/4/41/Akara.jpg",
        "Moi-Moi": "https://upload.wikimedia.org/wikipedia/commons/2/22/Moin_Moin.jpg",
        "Pap (Ogi)": "https://upload.wikimedia.org/wikipedia/commons/3/30/Pap_and_Akara.jpg",
        "Custard": "https://images.unsplash.com/photo-1627308595229-7830a5c91f9f?q=80&w=600",
        "Bread (Agege)": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Agege_Bread.jpg",
        "Fried Plantain (Dodo)": "https://upload.wikimedia.org/wikipedia/commons/7/7e/Fried_Plantain.jpg",
        "Boiled Plantain": "https://upload.wikimedia.org/wikipedia/commons/e/e2/Boiled_Plantain_and_Egg_Sauce.jpg",
        "Yam (Boiled)": "https://upload.wikimedia.org/wikipedia/commons/8/86/Boiled_Yam_and_Garden_Egg_Sauce.jpg",
        "Yam (Fried)": "https://upload.wikimedia.org/wikipedia/commons/1/15/Fried_Yam.jpg",
        "Indomie": "https://images.unsplash.com/photo-1612929633738-8fe44f7ec841?q=80&w=600",
        "Puff Puff": "https://upload.wikimedia.org/wikipedia/commons/9/90/Puff_Puff.jpg",
        "Meat Pie": "https://upload.wikimedia.org/wikipedia/commons/8/8f/Nigerian_Meat_Pie.jpg",
        "Egg Roll": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Egg_roll.jpg",
        
        # --- PROTEINS ---
        "Chicken": "https://images.unsplash.com/photo-1626082927389-6cd097cdc6ec?q=80&w=600",
        "Beef": "https://images.unsplash.com/photo-1615937651199-2364c7674254?q=80&w=600",
        "Fish": "https://images.unsplash.com/photo-1519708227418-c8fd9a32b7a2?q=80&w=600",
        "Egg (Boiled)": "https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?q=80&w=600",
        "Egg (Fried)": "https://images.unsplash.com/photo-1525351463929-491a9953fc33?q=80&w=600",
        "Ponmo": "https://upload.wikimedia.org/wikipedia/commons/3/39/Ponmo_IJebu.jpg",
        
        # --- FALLBACKS (If name doesn't match exactly) ---
        "Default": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=600"
    }
    
    # Try exact match first
    for key in images:
        if key.lower() in food_name.lower():
            return images[key]
            
    return images["Default"]