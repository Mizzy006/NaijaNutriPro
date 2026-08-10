"""
utils.py — Utility functions for NaijaNutriPro.

Provides PDF export functionality for weekly meal plans.
"""

from fpdf import FPDF


def create_pdf(user_profile: dict, weekly_plan: dict) -> bytes:
    """
    Generate a PDF report of the weekly meal plan.
    
    Args:
        user_profile: Dict with user details (Age, Gender, Goal, Budget, etc.)
        weekly_plan: The generated weekly plan dict.
    
    Returns:
        PDF file contents as bytes.
    """

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'DietPal - Weekly Meal Plan', 0, 1, 'C')
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, 'Powered by Hybrid ML + RAG', 0, 1, 'C')
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # 1. User Profile Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "User Profile:", 0, 1)
    pdf.set_font("Arial", size=11)
    for key, value in user_profile.items():
        pdf.cell(0, 7, f"- {key}: {value}", 0, 1)
    pdf.ln(5)

    # 2. Weekly Plan Section
    for day, data in weekly_plan.items():
        # Day Header
        pdf.set_font("Arial", 'B', 14)
        pdf.set_fill_color(200, 220, 200)  # Light Green background
        pdf.cell(
            0, 10,
            f"{day} (Total: {data['Total_Calories']} kcal | N{data['Total_Cost']})",
            1, 1, 'L', fill=True
        )

        # Meals
        pdf.set_font("Arial", size=11)
        for meal_type, items in data['Plan'].items():
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"{meal_type}:", 0, 1)
            pdf.set_font("Arial", size=11)

            for item in items:
                ml_score = item.get('ml_score', None)
                score_text = f" [ML: {int(ml_score * 100)}%]" if ml_score and ml_score > 0 else ""
                pdf.cell(10)  # Indent
                pdf.cell(
                    0, 6,
                    f"* {item['Name']} ({item['Calories']} kcal) - N{item['Price']}{score_text}",
                    0, 1
                )

        pdf.ln(5)

    return bytes(pdf.output())