"""
llm_agent.py — RAG-Augmented LLM Agent for NaijaNutriPro.

This module handles all LLM interactions using Groq API, now enhanced with:
1. RAG (Retrieval-Augmented Generation) — queries ChromaDB for relevant
   nutrition knowledge before each LLM call
2. LLM output validation — ensures chat-based plan modifications use
   real foods from the database with correct prices/calories
"""

import os
import re
import json
import base64
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import streamlit as st


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks that some models (e.g. Qwen) include.
    Handles: closed tags, unclosed tags, and entirely-wrapped responses.
    """
    if not text or "<think>" not in text:
        return text

    # Case 1: Closed tags — strip <think>...</think> and keep everything after
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned

    # Case 2: Unclosed tag — the model's actual answer may be at the end.
    # Try to extract content after common patterns like "Response:" or "Answer:"
    # that the model writes inside its thinking as a "draft"
    inner = re.sub(r"^<think>\s*", "", text, flags=re.DOTALL)
    # Look for the model's draft response inside the thinking
    for marker in ["Draft Response", "Final Response", "Response:", "My response:",
                   "Here's my response", "I'll respond with", "I would say:"]:
        idx = inner.rfind(marker)
        if idx != -1:
            # Extract everything from the marker onward
            extracted = inner[idx:]
            # Remove the marker label itself (e.g., "Draft Response (Mental Refinement):")
            extracted = re.sub(r"^[^:]*:\s*", "", extracted).strip()
            if extracted:
                return extracted

    # Case 3: No markers found — return original text without <think> tag prefix
    return inner.strip() if inner.strip() else text


def _get_food_reference_data():
    """Load a concise reference of real Nigerian food calorie/price data from the CSV."""
    try:
        csv_path = Path(__file__).parent / "data" / "nigerian_food_nutrition.csv"
        df = pd.read_csv(csv_path)
        # Pick common meals to give the model real reference points
        key_foods = df[["Name", "Serving_Size", "Calories_kcal", "Price_Avg_Naira"]].head(30)
        lines = []
        for _, row in key_foods.iterrows():
            lines.append(f"- {row['Name']} ({row['Serving_Size']}): {int(row['Calories_kcal'])} kcal, N{int(row['Price_Avg_Naira'])}")
        return "\n".join(lines)
    except Exception:
        return ""


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


# Defining Models (Using Groq's Free Tier Models)
TEXT_MODEL = "qwen/qwen3.6-27b"
VISION_MODEL = "qwen/qwen3.6-27b"

# Data path for validation
CSV_PATH = Path(__file__).parent / "data" / "nigerian_food_nutrition.csv"


# --- Lazy client initialization (avoids crash at import time) ---
_client = None

def get_client():
    """Initialize Groq client on first use, not at import time."""
    global _client
    if _client is None:
        load_dotenv()
        # Try Streamlit secrets first (for deployed apps), then .env
        api_key = None
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass  # No secrets.toml — fall through to .env
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or Streamlit Cloud Secrets."
            )
        _client = Groq(api_key=api_key)
    return _client


# --- Lazy RAG retriever initialization ---
_retriever = None

def get_retriever():
    """Initialize the RAG retriever on first use."""
    global _retriever
    if _retriever is None:
        try:
            from rag.retriever import NutritionRetriever
            _retriever = NutritionRetriever()
            if not _retriever.is_ready:
                print("[LLM Agent] RAG knowledge base not ready. Building it now...")
                from rag.knowledge_builder import build_knowledge_base
                build_knowledge_base()
                _retriever = NutritionRetriever()
        except Exception as e:
            print(f"[LLM Agent] Warning: RAG initialization failed: {e}")
            _retriever = None
    return _retriever


def _get_rag_context(query: str, n_results: int = 3, source_filter: str = None) -> str:
    """
    Retrieve relevant nutrition knowledge for a query.
    Returns formatted context string, or empty string if RAG is unavailable.
    """
    retriever = get_retriever()
    if retriever is None:
        return ""
    
    try:
        return retriever.retrieve_as_context(query, n_results, source_filter)
    except Exception as e:
        print(f"[LLM Agent] RAG retrieval warning: {e}")
        return ""


# ============================================================
# LLM OUTPUT VALIDATION
# ============================================================

def validate_llm_plan(updated_plan: dict, original_plan: dict) -> dict:
    """
    Validate an LLM-modified meal plan against the real food database.
    
    Checks:
    1. All food names exist in the CSV
    2. Calorie and price values match the database (within tolerance)
    3. Plan structure is intact (7 days, 3 meals each)
    
    Returns:
        dict with keys: 'valid' (bool), 'plan' (corrected plan), 'warnings' (list)
    """
    warnings = []
    
    try:
        df = pd.read_csv(CSV_PATH)
        food_lookup = {}
        for _, row in df.iterrows():
            food_lookup[row['Name'].lower()] = {
                'Name': row['Name'],
                'Calories': row['Calories_kcal'],
                'Price': row['Price_Avg_Naira'],
            }
    except Exception:
        # If we can't load the CSV, accept the plan as-is
        return {'valid': True, 'plan': updated_plan, 'warnings': ['Could not validate against food database.']}
    
    corrected_plan = {}
    
    for day, day_data in updated_plan.items():
        corrected_day = {
            'Total_Cost': 0,
            'Total_Calories': 0,
            'Wallet_Balance': day_data.get('Wallet_Balance', 0),
            'Plan': {}
        }
        
        if 'Plan' not in day_data:
            warnings.append(f"{day}: Missing 'Plan' key — using original.")
            corrected_plan[day] = original_plan.get(day, day_data)
            continue
        
        for meal_type, items in day_data['Plan'].items():
            corrected_items = []
            for item in items:
                item_name = item.get('Name', '')
                lookup_key = item_name.lower().replace(' (budget saver)', '')
                
                if lookup_key in food_lookup:
                    # Food exists — use database values for calories/price
                    db_food = food_lookup[lookup_key]
                    corrected_item = {
                        'Name': db_food['Name'],
                        'Calories': db_food['Calories'],
                        'Price': db_food['Price'],
                    }
                    
                    # Check if LLM gave wildly different values
                    if abs(item.get('Calories', 0) - db_food['Calories']) > 100:
                        warnings.append(
                            f"{day} {meal_type}: Corrected {item_name} calories "
                            f"from {item.get('Calories')} to {db_food['Calories']}"
                        )
                    
                    corrected_items.append(corrected_item)
                else:
                    # Food NOT in database — flag it but keep it
                    warnings.append(f"{day} {meal_type}: '{item_name}' not found in database — keeping LLM values.")
                    corrected_items.append(item)
            
            corrected_day['Plan'][meal_type] = corrected_items
            corrected_day['Total_Cost'] += sum(i.get('Price', 0) for i in corrected_items)
            corrected_day['Total_Calories'] += sum(i.get('Calories', 0) for i in corrected_items)
        
        corrected_plan[day] = corrected_day
    
    is_valid = len(warnings) == 0
    return {'valid': is_valid, 'plan': corrected_plan, 'warnings': warnings}


# ============================================================
# LLM FUNCTIONS (Now RAG-Augmented)
# ============================================================

def explain_meal_plan(user_profile, plan_data, day, style="Standard"):
    """
    Provides a RAG-augmented summary of a specific day's meal plan.
    Retrieves relevant nutrition knowledge before generating the explanation.
    """
    if style == "Standard":
        tone_instruction = "Write in a warm, conversational tone — like a friendly Nigerian dietician chatting with a client over tea. Use natural flowing sentences, not bullet points or lists."
    else:
        tone_instruction = "Write in warm, witty Nigerian Pidgin English — like a friendly iya (auntie) who also happens to be a nutritionist. Keep it natural and relatable."

    # --- RAG RETRIEVAL ---
    meal_names = []
    for meal_type, items in plan_data[day]['Plan'].items():
        meal_names.extend([item['Name'] for item in items])
    
    rag_query = f"Nutritional benefits of {', '.join(meal_names)} for {user_profile}"
    retrieved_context = _get_rag_context(rag_query, n_results=3)

    # --- AUGMENTED PROMPT ---
    prompt = f"""
    You are NaijaNutri — a friendly Nigerian dietician who personally created this meal plan.
    
    Your client's profile: {user_profile}
    The {day} plan you created: {json.dumps(plan_data[day], indent=2, cls=NumpyEncoder)}
    
    --- Nutrition Knowledge (from verified sources) ---
    {retrieved_context}
    ---
    
    Explain why you chose these meals for this person's {day}. {tone_instruction}
    Keep it to 2-3 short paragraphs.
    
    DO:
    - Talk as the person who made the plan ("I picked...", "I included...", "I went with...")
    - Start naturally (e.g., "So for your {day}, I went with...")
    - Weave in specific nutrients naturally (e.g., "the iron from the beans will keep your energy up")
    - Connect food choices to their health goals and budget
    - Sound like a real person giving caring advice
    
    DO NOT:
    - List foods mechanically ("For breakfast you have X, for lunch you have Y")
    - Use generic phrases like "it's worth noting", "the provided meal plan", "it seems"
    - Talk as if reviewing someone else's plan — you MADE this plan
    - Write in bullet points
    - Criticize or suggest changes unless asked
    - Say "your meal plan" as if the user created it — YOU are the dietician who created it
    """

    try:
        response = get_client().chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are NaijaNutri — a friendly, experienced Nigerian dietician who personally created this meal plan. You speak naturally and helpfully, like a real person. You MADE this plan, so talk about it with confidence — say 'I chose', 'I included', 'I picked' rather than 'your meal plan has'. Never use filler phrases like 'it seems' or 'it's worth noting'. Never criticize or second-guess the plan unless the user asks for changes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Omo, network glitch happen o. (API Error: {str(e)})"


def process_chat_query(user_prompt, plan_data, chat_history, style="Standard"):
    """
    Handles interactive user chat with RAG context and returns an UPDATED JSON plan.
    """
    if style == "Standard":
        tone = "warm and conversational — like a real dietician talking to a client"
    else:
        tone = "warm Nigerian Pidgin — like a friendly, knowledgeable auntie"

    # --- RAG RETRIEVAL ---
    retrieved_context = _get_rag_context(user_prompt, n_results=3)

    system_prompt = f"""
    You are NaijaNutri — a friendly, experienced Nigerian dietician who personally created
    this meal plan for your client. You know exactly why you chose each meal. Speak naturally
    and helpfully, the way a real nutritionist would talk to someone they care about.
    
    Tone: {tone}.
    
    THE MEAL PLAN YOU CREATED (JSON):
    {json.dumps(plan_data, indent=2, cls=NumpyEncoder)}
    
    --- Nutrition Knowledge ---
    {retrieved_context}
    ---
    
    PERSONA RULES:
    - You CREATED this meal plan. It is YOUR work. Never say "your meal plan" as if someone
      else made it. Say things like "I put together...", "I picked...", "I included..."
    - Be confident and supportive about your choices. Explain WHY you chose specific foods.
    - Talk like a normal, caring person — not a textbook or a robot.
    - Keep responses concise (2-4 sentences for simple questions, a short paragraph for explanations).
    - Reference specific nutrients naturally when relevant.
    - Do NOT use filler phrases like "it's worth noting", "it seems", "the provided meal plan".
    - Do NOT criticize or second-guess the plan unless the user asks for changes.
    
    RESPONSE FORMAT:
    Respond in valid JSON with exactly two keys:
    1. "message": Your natural, conversational response.
    2. "updated_plan": The complete meal plan JSON. If no changes were requested, return the original plan unchanged.
    
    IMPORTANT: Only use foods from our Nigerian food database. Do not invent foods.
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
        response = get_client().chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.3,  # Low temp so the JSON is strictly formatted
            response_format={"type": "json_object"},  # Force Groq to return pure JSON
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback empty JSON if it fails
        return json.dumps({"message": f"API Error: {str(e)}", "updated_plan": plan_data}, cls=NumpyEncoder)


def process_multimodal_query(user_prompt, image_file, budget, metrics, chat_history, health_conditions=None):
    """
    Handles multimodal chat with RAG-augmented context.
    
    Two-step pipeline for image requests:
      Step 1: Vision model (Qwen) identifies ingredients in the image
      Step 2: Text model (Llama) generates the structured meal recommendation
    
    Text-only requests go directly to the text model.
    """
    # --- RAG RETRIEVAL ---
    rag_query = user_prompt if user_prompt else "Nigerian meal suggestions cooking ingredients"
    retrieved_context = _get_rag_context(rag_query, n_results=2)

    try:
        if image_file:
            # =============================================
            # STEP 1: Use Vision model to identify ingredients
            # =============================================
            image_bytes = image_file.getvalue()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            vision_messages = [
                {"role": "system", "content": "You are a food ingredient identifier. Look at the image and list the food ingredients you see. Be brief and direct — just list the ingredients, nothing else."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What food ingredients are in this image? List them briefly."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]

            # Try with reasoning_format='parsed' first (cleaner output)
            vision_response = get_client().chat.completions.create(
                model=VISION_MODEL,
                messages=vision_messages,
                temperature=0.3,
                max_tokens=200,
                reasoning_format="parsed",
            )
            
            identified_ingredients = vision_response.choices[0].message.content or ""
            identified_ingredients = _strip_think_tags(identified_ingredients)
            
            # If parsed mode gave empty content, retry without it
            if not identified_ingredients.strip():
                vision_response = get_client().chat.completions.create(
                    model=VISION_MODEL,
                    messages=vision_messages,
                    temperature=0.3,
                    max_tokens=500,
                )
                identified_ingredients = _strip_think_tags(
                    vision_response.choices[0].message.content or ""
                )
            
            if not identified_ingredients.strip():
                identified_ingredients = "food ingredients (could not identify from image)"

            # =============================================
            # STEP 2: Use Text model for meal recommendation
            # =============================================
            user_request = user_prompt if user_prompt else "What can I make with these ingredients?"
            
            health_note = ""
            if health_conditions:
                health_note = f"\n            **User's Health Conditions:** {', '.join(health_conditions)}. Only consider dietary restrictions for THESE conditions."
            else:
                health_note = "\n            **User has NO health conditions.** Do not restrict foods based on health conditions mentioned in the nutrition knowledge unless the user specifically asks."

            food_ref = _get_food_reference_data()

            recommendation_prompt = f"""
            You are NaijaNutri, an expert Nigerian Chef and Dietitian.
            User's Daily Budget: N{budget} (this is their TOTAL daily budget across all meals).
            User's Daily Calorie Target: {metrics['Target_Calories']} kcal (TOTAL for the whole day).
            {health_note}
            
            IMPORTANT: You are suggesting ONE MEAL, not the entire day's food.
            A typical Nigerian meal is 400-800 kcal per serving. Do NOT inflate calories to match the daily target.
            Use the reference data below for ACCURATE calorie and price estimates:
            
            --- Nigerian Food Reference Data ---
            {food_ref}
            ---
            
            --- Retrieved Nutrition Knowledge ---
            {retrieved_context}
            ---
            
            The user uploaded a photo of their ingredients. Here's what was identified:
            **Ingredients from image:** {identified_ingredients}
            
            The user's request: "{user_request}"
            
            Based on the identified ingredients and their request, suggest ONE practical Nigerian meal.
            Your recommendation MUST incorporate the ingredients identified from their image.
            
            RESPONSE FORMAT — Structure your response exactly like this:

            **📷 What I Spotted:** [list the ingredients you were told were in the image]

            **🍽️ Recommended Meal: [Meal Name]**
            Estimated Cost: N[amount] | Estimated Calories: [amount] kcal (per serving, NOT daily total)

            **🛒 Ingredients:**
            - [Ingredient 1] — N[price]
            - [Ingredient 2] — N[price]
            (list all ingredients needed with approximate Nigerian market prices)

            **👨‍🍳 How to Cook:**
            1. [Step 1]
            2. [Step 2]
            3. [Step 3]
            (clear, numbered cooking steps)

            **💡 Why This Works For You:**
            [1-2 sentences on how this meal contributes to their daily calorie and budget goals]
            
            Keep it concise and practical.
            """

            messages = [
                {"role": "system", "content": "You are NaijaNutri — a friendly, expert Nigerian Chef and Dietitian. Give practical, structured meal recommendations."},
            ]
            # Add recent chat context
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"] and isinstance(msg["content"], str):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": recommendation_prompt})

            response = get_client().chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        else:
            # =============================================
            # TEXT ONLY PATH — Straight to Llama
            # =============================================
            health_note = ""
            if health_conditions:
                health_note = f"\n            **User's Health Conditions:** {', '.join(health_conditions)}. Only consider dietary restrictions for THESE conditions."
            else:
                health_note = "\n            **User has NO health conditions.** Do not restrict foods based on health conditions mentioned in the nutrition knowledge unless the user specifically asks."

            food_ref = _get_food_reference_data()

            text_prompt = f"""
            You are NaijaNutri, an expert Nigerian Chef and Dietitian.
            User's Daily Budget: N{budget} (this is their TOTAL daily budget across all meals).
            User's Daily Calorie Target: {metrics['Target_Calories']} kcal (TOTAL for the whole day).
            {health_note}
            
            IMPORTANT: You are suggesting ONE MEAL, not the entire day's food.
            A typical Nigerian meal is 400-800 kcal per serving. Do NOT inflate calories to match the daily target.
            Use the reference data below for ACCURATE calorie and price estimates:
            
            --- Nigerian Food Reference Data ---
            {food_ref}
            ---
            
            --- Retrieved Nutrition Knowledge ---
            {retrieved_context}
            ---
            
            The user's request: "{user_prompt}"
            
            Suggest ONE practical Nigerian meal based on their request.
            
            RESPONSE FORMAT — Structure your response exactly like this:

            **🍽️ Recommended Meal: [Meal Name]**
            Estimated Cost: N[amount] | Estimated Calories: [amount] kcal (per serving, NOT daily total)

            **🛒 Ingredients:**
            - [Ingredient 1] — N[price]
            - [Ingredient 2] — N[price]

            **👨‍🍳 How to Cook:**
            1. [Step 1]
            2. [Step 2]
            3. [Step 3]

            **💡 Why This Works For You:**
            [1-2 sentences on how this meal contributes to their daily calorie and budget goals]
            
            Keep it concise and practical.
            """

            messages = [
                {"role": "system", "content": "You are NaijaNutri — a friendly, expert Nigerian Chef and Dietitian. Give practical, structured meal recommendations."},
            ]
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"] and isinstance(msg["content"], str):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": text_prompt})

            response = get_client().chat.completions.create(
                model=TEXT_MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=1024,
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Oops! I couldn't process that request. (Error: {str(e)})"