import time
import sys
import os
from groq import Groq

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_agent import explain_meal_plan, get_client, TEXT_MODEL

def test_pipeline_latency():
    user_profile = "Male, 30 years old, 75kg, 180cm, Active, Goal: Muscle Gain"
    plan_data = {
        "Monday": {
            "Plan": {
                "Breakfast": [{"Name": "Oats and Milk"}],
                "Lunch": [{"Name": "Jollof Rice with Chicken"}],
                "Dinner": [{"Name": "Pounded Yam with Egusi Soup"}]
            }
        }
    }
    day = "Monday"

    print("Testing Pipeline Latency (includes RAG + LLM)...")
    start_time = time.time()
    try:
        response = explain_meal_plan(user_profile, plan_data, day)
        end_time = time.time()
        print(f"Pipeline latency: {end_time - start_time:.2f} seconds")
        print(f"Response snippet: {response[:150]}...\n")
    except Exception as e:
        print(f"Pipeline Error: {e}\n")

def test_pure_llm_latency():
    print("Testing Pure LLM Latency (direct Groq API call)...")
    client = get_client()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the nutritional benefits of Jollof Rice."}
    ]
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=500,
        )
        end_time = time.time()
        print(f"Pure LLM latency: {end_time - start_time:.2f} seconds")
        print(f"Response snippet: {response.choices[0].message.content[:150]}...\n")
    except Exception as e:
        print(f"LLM Error: {e}\n")

if __name__ == "__main__":
    print(f"Using Model: {TEXT_MODEL}\n")
    test_pure_llm_latency()
    test_pipeline_latency()
