import os
import json
import base64
from dotenv import load_dotenv  
from groq import Groq


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if hasattr(obj, 'item'): # This safely catches numpy int64, float64, etc.
            return obj.item()
        return super().default(obj)

# Load the environment variables from the .env file
load_dotenv() 

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Defining Models (Using Groq's Free Tier Models)
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def explain_meal_plan(user_profile, plan_data, day, style="Standard"):
    """
    Provides a static summary of a specific day's meal plan.
    """
    tone = "Speak in a friendly, professional tone." if style == "Standard" else "Speak in witty, polite Nigerian Pidgin English."
    
    prompt = f"""
    You are NaijaNutri, an expert Nigerian dietitian AI.
    User Profile: {user_profile}
    Meal Plan for {day}: {json.dumps(plan_data[day], indent=2, cls=NumpyEncoder)}
    
    Explain why these specific foods were chosen for this user today. 
    Focus on the budget savings and the nutritional value of the indigenous foods.
    Keep it under 3 paragraphs. {tone}
    """

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful, context-aware Nigerian dietary assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Omo, network glitch happen o. (API Error: {str(e)})"


def process_chat_query(user_prompt, plan_data, chat_history, style="Standard"):
    """
    Handles interactive user chat and returns an UPDATED JSON plan.
    """
    tone = "professional and encouraging" if style == "Standard" else "friendly Nigerian Pidgin"
    
    system_prompt = f"""
    You are NaijaNutri, an intelligent Nigerian Dietitian. Tone: {tone}.
    
    Current Meal Plan Context (JSON):
    {json.dumps(plan_data, indent=2, cls=NumpyEncoder)}
    
    CRITICAL INSTRUCTION:
    You MUST respond in valid JSON format. Your JSON object must contain exactly these two keys:
    1. "message": Your conversational response explaining what you did.
    2. "updated_plan": The complete, modified meal plan JSON object. If the user asks for a swap, modify the specific food items, prices, and calories in this JSON to reflect the change. If no changes are requested, just return the original plan.
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Sending the text messages as history, not huge JSON blocks
    for msg in chat_history[1:]: 
        if msg["role"] in ["user", "assistant"]:
            # Only append if it's a simple string to save tokens
            if isinstance(msg["content"], str):
                messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.3, # Low temp so the JSON is strictly formatted
            response_format={"type": "json_object"}, # Force Groq to return pure JSON
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback empty JSON if it fails
        return json.dumps({"message": f"API Error: {str(e)}", "updated_plan": plan_data}, cls=NumpyEncoder)


def process_multimodal_query(user_prompt, image_file, budget, metrics, chat_history):
    """
    Handles multimodal chat. Routes to Vision model if an image is provided, 
    otherwise routes to standard Text model.
    """
    system_prompt = f"""
    You are NaijaNutri, an expert Nigerian Chef and Dietitian.
    User's Daily Budget: N{budget}. 
    User's Daily Calorie Target: {metrics['Target_Calories']} kcal.
    
    Instructions:
    1. If the user provides an image, identify the ingredients and suggest ONE practical Nigerian meal they can cook that fits their budget and calories.
    2. If they only provide text (e.g., "I want pounded yam"), fulfill their request while keeping their budget and calories in mind.
    3. Always explain how the meal helps them meet their goals.
    """

    # Build the message history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add the last few chat messages for context
    for msg in chat_history[-4:]: # Keep it to the last 4 to save tokens
        if msg["role"] in ["user", "assistant"]:
            if isinstance(msg["content"], str):
                messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        if image_file:
            # --- IMAGE + TEXT PATH (Vision Model) ---
            image_bytes = image_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # If the user didn't type anything, give a default prompt
            final_prompt = user_prompt if user_prompt else "What can I make with these ingredients?"
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
            model_to_use = VISION_MODEL
        else:
            # --- TEXT ONLY PATH (Text Model) ---
            messages.append({"role": "user", "content": user_prompt})
            model_to_use = TEXT_MODEL

        # Call Groq API
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Oops! I couldn't process that request. (Error: {str(e)})"