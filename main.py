import streamlit as st
import pandas as pd
from user_profile import calculate_biometrics
from engine import generate_weekly_plan
from llm_agent import explain_meal_plan, process_chat_query, process_multimodal_query # 
from utils import create_pdf
import json

# PAGE CONFIGURATION 
st.set_page_config(page_title="NaijaNutri AI", page_icon="🇳🇬", layout="centered")

#  INITIALIZE SESSION STATE 
if "plan_data" not in st.session_state:
    st.session_state.plan_data = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "How far! How I fit adjust this meal plan for you?"}]
if "vision_history" not in st.session_state:
    st.session_state.vision_history = [{"role": "assistant", "content": "Upload a picture of your ingredients, or just tell me what you're craving!"}]

# CUSTOM CSS 
st.markdown("""
    <style>
    .stButton>button {
        background-color: #008751;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #006b3f;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #008751 !important;
        color: white !important;
    }
    /* Chat message styling */
    .stChatMessage {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

#  HEADER SECTION 
st.title("🇳🇬 NaijaNutri: Context-Aware Dietitian")
st.write("Intelligent Meal Planning using **Constraint Optimization**, **Multimodal AI**, & **RAG**.")
st.divider()

#  SIDEBAR (USER INPUTS) 
st.sidebar.header("📝 User Profile")
gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age (Years)", 16, 60, 22)
height = st.sidebar.number_input("Height (cm)", 140, 220, 175)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)
activity_level = st.sidebar.select_slider(
    "Activity Level", 
    options=["Sedentary", "Light", "Moderate", "Active"], 
    value="Moderate",
    help="""
    **How active are you?**
    * **Sedentary:** Little to no exercise (e.g., desk job, studying all day).
    * **Light:** Light exercise or sports 1-3 days a week.
    * **Moderate:** Moderate exercise or sports 3-5 days a week.
    * **Active:** Hard exercise or physical labor 6-7 days a week.
    """
)
goal = st.sidebar.selectbox("Health Goal", ["Maintain Weight", "Lose Weight", "Gain Weight"])

st.sidebar.divider()
st.sidebar.header("💰 Constraints")
budget = st.sidebar.slider("Daily Food Budget (₦)", 500, 10000, 2500, 100)
# Enforce Halal dietary preferences automatically in your engine logic based on user profile
# In app.py
dietary_pref = st.sidebar.selectbox(
    "Dietary Preference", 
    ["None", "Halal", "Vegetarian", "Pescatarian"],
    help="""
    **Filter your meals:**
    * **None:** Eat everything.
    * **Halal:** Excludes pork and non-halal meats (adheres to Islamic dietary laws).
    * **Vegetarian:** Plant-based only (no meat, poultry, or fish).
    * **Pescatarian:** Plant-based + Fish/Seafood (no meat or poultry).
    """
)

st.sidebar.divider()
st.sidebar.header("🤖 AI Persona")
persona = st.sidebar.radio("Tone", ["Standard (Friendly)", "Pidgin (Witty)"])


# MAIN APP LAYOUT (TABS) 
main_tab, vision_tab = st.tabs(["📅 Core Meal Planner", "📸 Pantry Scanner (Vision)"])

# TAB 1: CORE MEAL PLANNER & CHAT

with main_tab:
    if st.sidebar.button("Generate Meal Plan 🚀"):
        with st.spinner("Optimizing your budget and calories..."):
            # 1. Calculations
            st.session_state.metrics = calculate_biometrics(weight, height, age, gender.lower(), activity_level.lower(), goal.lower().replace(" ", "_"))
            
            # 2. Generate Plan and save to session state
            st.session_state.plan_data = generate_weekly_plan(st.session_state.metrics['Target_Calories'], budget, dietary_pref)
            
            # Reset chat history for a new plan
            st.session_state.chat_history = [{"role": "assistant", "content": "Plan generated! You can ask me to swap meals, make it lower carb, or explain my choices."}]

    # Only display if a plan has been generated and saved in memory
    if st.session_state.plan_data is not None:
        if isinstance(st.session_state.plan_data, str):
            st.error(st.session_state.plan_data)
        else:
            metrics = st.session_state.metrics
            plan_data = st.session_state.plan_data
            
            st.success(f"**Target: {metrics['Target_Calories']} kcal** | **TDEE: {metrics['TDEE']} kcal**")
            
            # Download Button
            user_data = {"Age": age, "Gender": gender, "Goal": goal, "Budget": f"N{budget}"}
            pdf_bytes = create_pdf(user_data, plan_data)
            
            st.download_button(
                label="📄 Download Weekly Plan (PDF)",
                data=pdf_bytes,
                file_name="NaijaNutri_Plan.pdf",
                mime="application/pdf",
            )
            st.divider()
            
            # Display Plan Days
            days = list(plan_data.keys())
            day_tabs = st.tabs(days)
            
            for i, day in enumerate(days):
                day_data = plan_data[day]
                with day_tabs[i]:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cost", f"₦{day_data['Total_Cost']}")
                    c2.metric("Savings", f"₦{day_data['Wallet_Balance']}")
                    c3.metric("Calories", f"{day_data['Total_Calories']} kcal", delta=f"{day_data['Total_Calories'] - metrics['Target_Calories']}")
                    
                    st.markdown("---")
                    
                    for meal_type, items in day_data['Plan'].items():
                        st.markdown(f"#### {meal_type}")
                        for item in items:
                            st.write(f"**{item['Name']}** - {item['Calories']} kcal (₦{item['Price']})")
                        st.divider()

            # INTERACTIVE CHAT SECTION 
            st.markdown("### 💬 Chat with your Plan")
            
            # Display chat messages from history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("E.g., 'Make Tuesday's dinner low-carb' or 'Swap the amala for rice'"):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Call LLM Agent to process the request
                with st.chat_message("assistant"):
                    with st.spinner("Processing request and updating dashboard..."):
                        selected_style = "Pidgin" if "Pidgin" in persona else "Standard"
                        raw_response = process_chat_query(prompt, st.session_state.plan_data, st.session_state.chat_history, selected_style)
                        
                        try:
                            import json # Ensure json is imported at the top of app.py
                            
                            # Parse the JSON response from Groq
                            response_data = json.loads(raw_response)
                            assistant_message = response_data.get("message", "Here is your updated plan!")
                            new_plan = response_data.get("updated_plan", st.session_state.plan_data)
                            
                            st.markdown(assistant_message)
                            
                            # Check if the AI actually changed the plan data
                            if new_plan != st.session_state.plan_data:
                                # Overwrite the session state with the new plan
                                st.session_state.plan_data = new_plan
                                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                                # Rerun the whole app so the UI updates instantly!
                                st.rerun()
                            else:
                                st.session_state.chat_history.append({"role": "assistant", "content": assistant_message})
                                
                        except Exception as e:
                            st.error("Failed to apply changes to the dashboard.")
                            st.write(raw_response) # Show raw text if JSON parsing fails

# ==========================================
# TAB 2: MULTIMODAL PANTRY ASSISTANT
# ==========================================
with vision_tab:
    st.markdown("### 📸 & 💬 Multimodal Pantry Assistant")
    
    if st.session_state.metrics is None:
        st.warning("⚠️ Please click **'Generate Meal Plan'** in the sidebar first so I know your calorie targets!")
    else:
        # 1. The Image Input Area (Stays at the top)
        st.write("Upload an image (optional), then type your request below.")
        input_method = st.radio("Choose Image Source", ["Upload Image", "Use Camera", "Text Only (No Image)"], horizontal=True)
        
        image_data = None
        if input_method == "Upload Image":
            image_data = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        elif input_method == "Use Camera":
            image_data = st.camera_input("Take a picture")

        if image_data:
            # Show a small preview of the uploaded image
            st.image(image_data, caption="Image ready for analysis", width=250)
            
        st.divider()

        # 2. The Chat Interface
        # Display chat messages from history
        for message in st.session_state.vision_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. The Chat Input box
        if prompt := st.chat_input("E.g., 'What can I make with this?' or 'I want pounded yam today'"):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.vision_history.append({"role": "user", "content": prompt})

            # Process the request
            with st.chat_message("assistant"):
                with st.spinner("Analyzing request..."):
                    # Call our new multimodal function
                    response = process_multimodal_query(
                        user_prompt=prompt,
                        image_file=image_data,
                        budget=budget,
                        metrics=st.session_state.metrics,
                        chat_history=st.session_state.vision_history
                    )
                    st.markdown(response)
            
            # Save response to history
            st.session_state.vision_history.append({"role": "assistant", "content": response})