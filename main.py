"""
main.py — DietPal: Context-Aware Nigerian Nutritional Recommender.

Streamlit application integrating:
- Hybrid ML meal planning (content-based scoring + PuLP optimization)
- RAG-augmented LLM chat (ChromaDB + sentence-transformers + Groq)
- Multimodal vision (image-based ingredient recognition)
- Health condition awareness (Diabetes, Hypertension, Pregnancy, Ulcer)
"""

import streamlit as st
import json
from user_profile import calculate_biometrics
from engine import generate_weekly_plan
from llm_agent import (
    explain_meal_plan, process_chat_query,
    process_multimodal_query, validate_llm_plan
)
from utils import create_pdf


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="DietPal AI",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "plan_data" not in st.session_state:
    st.session_state.plan_data = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Your meal plan is ready. Ask me to swap meals, adjust macros, or explain any choices."}
    ]
if "vision_history" not in st.session_state:
    st.session_state.vision_history = [
        {"role": "assistant", "content": "Upload a photo of your ingredients or describe what you have, and I'll suggest meals."}
    ]
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# ============================================================
# ONE-TIME RAG INITIALIZATION
# ============================================================
if not st.session_state.rag_ready:
    try:
        from rag.knowledge_builder import build_knowledge_base
        build_knowledge_base()  # Idempotent — skips if already built
        st.session_state.rag_ready = True
    except Exception as e:
        # RAG is optional — app works without it, just no retrieval
        st.session_state.rag_ready = False
        print(f"[App] RAG init warning: {e}")


# ============================================================
# CUSTOM CSS — Professional, Responsive Design
# ============================================================
st.markdown("""
    <style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, .stApp, .stMarkdown, .stText,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* ── App Container ── */
    .stApp {
        background-color: #ffffff;
    }

    /* ── Header ── */
    .app-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem;
        color: #6b7280;
        margin: 0;
        font-weight: 400;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #008751;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem;
        padding: 0.6rem 1.25rem;
        transition: background-color 0.2s ease, box-shadow 0.2s ease;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover {
        background-color: #006b3f;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 135, 81, 0.25);
    }
    .stButton > button:active {
        background-color: #005a34;
        color: white;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background-color: #f7f8fa;
        color: #1a1a2e;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        font-weight: 500;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
        color: #1a1a2e;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e8e8e8;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem;
        color: #6b7280;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        transition: color 0.2s ease, border-color 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #008751;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #008751 !important;
        font-weight: 600;
        border-bottom: 2px solid #008751 !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1rem;
        transition: box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.75rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #f7f8fa;
        border-right: 1px solid #e8e8e8;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 1rem;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        background-color: #f9fafb;
        border: 1px solid #f0f0f0;
        border-radius: 10px;
        padding: 0.875rem;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Chat Input ── */
    [data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem;
        border-radius: 10px;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem;
        font-weight: 500;
        color: #374151;
    }

    /* ── Match Badges ── */
    .match-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-left: 6px;
        vertical-align: middle;
        letter-spacing: 0.02em;
    }
    .match-excellent {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .match-good {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .match-fair {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .match-budget {
        background: #f0f4ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }

    /* ── Similarity Badges ── */
    .sim-badge {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 20px;
        font-size: 0.68rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        margin-left: 4px;
    }
    .sim-high {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .sim-med {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .sim-low {
        background: #f0f4ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }

    /* ── Role Tags ── */
    .role-tag {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        margin-right: 6px;
        vertical-align: middle;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .role-snack {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }
    .role-side {
        background: #faf5ff;
        color: #7c3aed;
        border: 1px solid #ddd6fe;
    }
    .role-drink {
        background: #ecfeff;
        color: #0e7490;
        border: 1px solid #a5f3fc;
    }
    .role-protein {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }

    /* ── Alerts/Info Boxes ── */
    .stAlert, [data-testid="stAlert"] {
        border-radius: 8px;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem;
    }

    /* ── Section Divider ── */
    .section-divider {
        border: none;
        height: 1px;
        background: #e8e8e8;
        margin: 1.5rem 0;
    }

    /* ── Target Banner ── */
    .target-banner {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: #065f46;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Health Notice ── */
    .health-notice {
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.825rem;
        color: #3730a3;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Responsive: Mobile ── */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 1.35rem;
        }
        .app-header p {
            font-size: 0.8rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.05rem;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.65rem;
        }
        [data-testid="stMetric"] {
            padding: 0.6rem;
        }
        .match-badge {
            font-size: 0.62rem;
            padding: 1px 7px;
        }
        .role-tag {
            font-size: 0.58rem;
            padding: 1px 6px;
        }
        [data-testid="stChatMessage"] {
            padding: 0.6rem;
        }
    }

    /* ── Responsive: Tablet ── */
    @media (min-width: 769px) and (max-width: 1024px) {
        .app-header h1 {
            font-size: 1.5rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.1rem;
        }
    }

    /* ── Hide default Streamlit header decoration ── */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ── Hide sidebar collapse button (broken icon font fallback) ── */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# HEADER SECTION
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>DietPal</h1>
    <p>Context-aware meal planning tailored to Nigerian diets</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPER: Convert ML score to user-friendly label
# ============================================================
def get_match_badge(ml_score: float) -> str:
    """Convert a raw 0-1 ML score to a professional match badge (HTML)."""
    pct = int(ml_score * 100)
    if pct >= 80:
        return '<span class="match-badge match-excellent">Excellent Fit</span>'
    elif pct >= 60:
        return '<span class="match-badge match-good">Good Fit</span>'
    elif pct >= 40:
        return '<span class="match-badge match-fair">Fair Fit</span>'
    else:
        return '<span class="match-badge match-budget">Budget Pick</span>'


def get_similarity_label(sim_score: float) -> str:
    """Convert a raw similarity score to a user-friendly label (HTML)."""
    pct = int(sim_score * 100)
    if pct >= 85:
        return '<span class="sim-badge sim-high">Very Similar</span>'
    elif pct >= 65:
        return '<span class="sim-badge sim-med">Similar</span>'
    else:
        return '<span class="sim-badge sim-low">Related</span>'


def get_role_tag(category: str) -> str:
    """
    Return an HTML role tag for non-main food categories.
    Helps users distinguish snacks, sides, proteins, and drinks from main dishes.
    Returns empty string for main dish categories (no tag needed).
    """
    MAIN_CATEGORIES = {"Rice", "Swallow", "Beans", "Pasta", "Tuber",
                       "Breakfast", "Carb", "Noodle", "Light Main"}
    if category in MAIN_CATEGORIES:
        return ""
    role_map = {
        "Snack":   '<span class="role-tag role-snack">Snack</span>',
        "Side":    '<span class="role-tag role-side">Side</span>',
        "Drink":   '<span class="role-tag role-drink">Drink</span>',
        "Protein": '<span class="role-tag role-protein">Protein</span>',
        "Soup":    '',  # Soup is integral to swallow meals, no tag needed
    }
    return role_map.get(category, "")


# ============================================================
# SIDEBAR (USER INPUTS)
# ============================================================
st.sidebar.header("Your Profile")
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
st.sidebar.header("Constraints")
budget = st.sidebar.slider("Daily Food Budget (NGN)", 500, 10000, 2500, 100)
dietary_pref = st.sidebar.selectbox(
    "Dietary Preference",
    ["None", "Halal", "Vegetarian", "Pescatarian"],
    help="""
    **Filter your meals:**
    * **None:** No restrictions.
    * **Halal:** Excludes pork and non-halal meats.
    * **Vegetarian:** Plant-based only (no meat, poultry, or fish).
    * **Pescatarian:** Plant-based + Fish/Seafood.
    """
)

st.sidebar.divider()
st.sidebar.header("Health Conditions")
health_conditions = st.sidebar.multiselect(
    "Select any that apply",
    ["Diabetes", "Hypertension", "Pregnancy", "Ulcer"],
    help="""
    **Your meal plan will be adjusted for:**
    * **Diabetes:** Lower carbs, no sugar, high fiber foods preferred.
    * **Hypertension:** Low sodium, potassium-rich foods preferred.
    * **Pregnancy:** Higher calories, iron-rich and calcium-rich foods.
    * **Ulcer:** Avoids spicy and fried foods, prefers gentle options.
    """
)

st.sidebar.divider()
st.sidebar.header("AI Persona")
persona = st.sidebar.radio("Tone", ["Standard (Friendly)", "Pidgin (Witty)"])


# ============================================================
# MAIN APP LAYOUT (TABS)
# ============================================================
main_tab, vision_tab = st.tabs(["Meal Planner", "Pantry Scanner"])


# ============================================================
# TAB 1: CORE MEAL PLANNER & CHAT
# ============================================================
with main_tab:
    if st.sidebar.button("Generate Meal Plan"):
        with st.spinner("Running ML scoring & constraint optimization..."):
            # 1. Calculate biometrics with health conditions
            st.session_state.metrics = calculate_biometrics(
                weight, height, age,
                gender.lower(),
                activity_level.lower(),
                goal.lower().replace(" ", "_"),
                health_conditions=health_conditions if health_conditions else None,
            )

            # 2. Generate plan using Hybrid ML engine
            st.session_state.plan_data = generate_weekly_plan(
                st.session_state.metrics['Target_Calories'],
                budget,
                dietary_pref,
                health_conditions=health_conditions if health_conditions else None,
            )

            # Reset chat history for a new plan
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Plan generated. You can ask me to swap meals, adjust macros, or explain my choices."}
            ]

    # Only display if a plan has been generated and saved in memory
    if st.session_state.plan_data is not None:
        if isinstance(st.session_state.plan_data, str):
            st.error(st.session_state.plan_data)
        else:
            metrics = st.session_state.metrics
            plan_data = st.session_state.plan_data

            # Target summary banner
            st.markdown(
                f'<div class="target-banner">Daily Target: <strong>{metrics["Target_Calories"]} kcal</strong> &nbsp;&middot;&nbsp; TDEE: <strong>{metrics["TDEE"]} kcal</strong></div>',
                unsafe_allow_html=True,
            )

            # Show active health conditions
            if health_conditions:
                st.markdown(
                    f'<div class="health-notice">Health-aware planning active for: <strong>{", ".join(health_conditions)}</strong></div>',
                    unsafe_allow_html=True,
                )

            # Download Button
            user_data = {
                "Age": age, "Gender": gender, "Goal": goal,
                "Budget": f"N{budget}",
                "Health Conditions": ", ".join(health_conditions) if health_conditions else "None",
            }
            pdf_bytes = create_pdf(user_data, plan_data)

            st.download_button(
                label="Download Weekly Plan (PDF)",
                data=pdf_bytes,
                file_name="DietPal_Plan.pdf",
                mime="application/pdf",
            )

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # Match badge legend
            with st.expander("What do the match labels mean?"):
                st.markdown("""
Each food item has a **match label** showing how well it fits your personal targets:
- **Excellent Fit** — Closely matches your calorie & nutrition goals
- **Good Fit** — Solid nutritional fit for this meal
- **Fair Fit** — Reasonable choice within your budget
- **Budget Pick** — Selected to keep you within your daily budget
                """)

            # Display Plan Days
            days = list(plan_data.keys())
            day_tabs = st.tabs(days)

            for i, day in enumerate(days):
                day_data = plan_data[day]
                with day_tabs[i]:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cost", f"₦{day_data['Total_Cost']}")
                    c2.metric("Savings", f"₦{day_data['Wallet_Balance']}")
                    c3.metric(
                        "Calories",
                        f"{day_data['Total_Calories']} kcal",
                        delta=f"{day_data['Total_Calories'] - metrics['Target_Calories']}"
                    )

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    for meal_type, items in day_data['Plan'].items():
                        st.markdown(f"#### {meal_type}")
                        for item in items:
                            # Build role tag (Snack, Side, Protein, Drink)
                            category = item.get('Category', '')
                            role_html = get_role_tag(category)

                            # Display user-friendly match badge
                            ml_score = item.get('ml_score', None)
                            if ml_score is not None and ml_score > 0:
                                badge_html = get_match_badge(ml_score)
                                st.markdown(
                                    f"{role_html}**{item['Name']}** — {item['Calories']} kcal "
                                    f"(₦{item['Price']}) {badge_html}",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"{role_html}**{item['Name']}** — {item['Calories']} kcal (₦{item['Price']})",
                                    unsafe_allow_html=True,
                                )
                            # Similar foods expander
                            if ml_score and ml_score > 0 and item['Name'] != "Skipped (Insufficient Funds)":
                                try:
                                    from ml.food_scorer import FoodScorer
                                    import pandas as pd
                                    from pathlib import Path

                                    csv_path = Path(__file__).parent / "data" / "nigerian_food_nutrition.csv"
                                    df = pd.read_csv(csv_path)
                                    scorer = FoodScorer(df)
                                    similar = scorer.get_similar_foods(item['Name'], top_n=3)
                                    if similar:
                                        with st.expander(f"Alternatives for {item['Name']}"):
                                            for s in similar:
                                                sim_label = get_similarity_label(s['similarity_score'])
                                                st.markdown(
                                                    f"• **{s['name']}** — {s['calories']} kcal "
                                                    f"(₦{s['price']}) {sim_label}",
                                                    unsafe_allow_html=True,
                                                )
                                except Exception:
                                    pass

                        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # ============================================================
            # INTERACTIVE CHAT SECTION (RAG-Augmented)
            # ============================================================
            st.markdown("### Chat with your Plan")

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

                # Call RAG-augmented LLM Agent
                with st.chat_message("assistant"):
                    with st.spinner("Processing your request..."):
                        selected_style = "Pidgin" if "Pidgin" in persona else "Standard"
                        raw_response = process_chat_query(
                            prompt, st.session_state.plan_data,
                            st.session_state.chat_history, selected_style
                        )

                        try:
                            response_data = json.loads(raw_response)
                            assistant_message = response_data.get("message", "Here is your updated plan.")
                            new_plan = response_data.get("updated_plan", st.session_state.plan_data)

                            st.markdown(assistant_message)

                            # Validate LLM output against food database
                            if new_plan != st.session_state.plan_data:
                                validation = validate_llm_plan(new_plan, st.session_state.plan_data)
                                
                                if validation['warnings']:
                                    with st.expander("Validation Notes"):
                                        for w in validation['warnings']:
                                            st.write(f"• {w}")
                                
                                st.session_state.plan_data = validation['plan']
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": assistant_message}
                                )
                                st.rerun()
                            else:
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": assistant_message}
                                )

                        except Exception as e:
                            st.error("Failed to apply changes to the dashboard.")
                            st.write(raw_response)


# ============================================================
# TAB 2: MULTIMODAL PANTRY ASSISTANT
# ============================================================
with vision_tab:
    st.markdown("### Pantry Scanner")

    if st.session_state.metrics is None:
        st.warning("Please generate a meal plan from the sidebar first so your calorie targets are set.")
    else:
        # 1. The Image Input Area
        st.write("Upload an image (optional), then type your request below.")
        input_method = st.radio(
            "Image Source",
            ["Upload Image", "Use Camera", "Text Only"],
            horizontal=True
        )

        image_data = None
        if input_method == "Upload Image":
            image_data = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        elif input_method == "Use Camera":
            image_data = st.camera_input("Take a picture")

        if image_data:
            st.image(image_data, caption="Image ready for analysis", width=250)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # 2. The Chat Interface
        for message in st.session_state.vision_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. The Chat Input box
        if prompt := st.chat_input("E.g., 'What can I make with this?' or 'I want pounded yam today'"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.vision_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = process_multimodal_query(
                        user_prompt=prompt,
                        image_file=image_data,
                        budget=budget,
                        metrics=st.session_state.metrics,
                        chat_history=st.session_state.vision_history,
                        health_conditions=health_conditions if health_conditions else None,
                    )
                    st.markdown(response)

            st.session_state.vision_history.append({"role": "assistant", "content": response})
