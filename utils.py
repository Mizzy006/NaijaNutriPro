from fpdf import FPDF
import base64

def create_pdf(user_profile, weekly_plan):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'NaijaNutri - Weekly Meal Plan', 0, 1, 'C')
            self.ln(5)

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
        pdf.set_fill_color(200, 220, 200) # Light Green background
        pdf.cell(0, 10, f"{day} (Total: {data['Total_Calories']} kcal | N{data['Total_Cost']})", 1, 1, 'L', fill=True)
        
        # Meals
        pdf.set_font("Arial", size=11)
        for meal_type, items in data['Plan'].items():
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, f"{meal_type}:", 0, 1)
            pdf.set_font("Arial", size=11)
            
            for item in items:
                pdf.cell(10) # Indent
                pdf.cell(0, 6, f"* {item['Name']} ({item['Calories']} kcal) - N{item['Price']}", 0, 1)
        
        pdf.ln(5)

    return bytes(pdf.output())