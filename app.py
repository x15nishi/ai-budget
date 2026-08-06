#======STEP 1: LOAD MODULES=====#
import os
import ssl
import smtplib
import urllib.parse
from email.mime.text import MIMEText

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title = "AI Personal Budget Planner", layout = "wide", page_icon = "💰")

#===============STEP 2: UI ADDUP (custom css)==============
st.markdown("""
<style>
.stMetric {background-color:#F5F7FA; padding:10px; border-radius:10px; border:1px solid #E0E4EA;}
div.stButton > button {border-radius:8px; font-weight:600;}
.main-title {text-align:center; color:#1E3A5F;}
</style>
""", unsafe_allow_html = True)

st.markdown("<h1 class='main-title'>AI Personal Budget Planner 💰</h1>", unsafe_allow_html = True)
st.header("""Track your income, expenses, and get AI powered budget advice""")

#===============STEP 3: LOAD ENV AND API-KEYS==============
st.sidebar.title("Give API KEY")
env_key = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type = "password", value = env_key if env_key else "")

if GEMINI_API_KEY:
    st.sidebar.success("API key Loaded!!")
    genai.configure(api_key = GEMINI_API_KEY)
else:
    st.sidebar.info("Give API key")
    url = "https://aistudio.google.com/app/apikey"
    st.sidebar.markdown(f"Get Gemini API Key-{url}")

st.sidebar.divider()
st.sidebar.title("Email Setup (optional)")
SENDER_EMAIL = st.sidebar.text_input("Your Gmail")
SENDER_APP_PASSWORD = st.sidebar.text_input("Gmail App Password", type = "password")
st.sidebar.caption("Needed only if you want to send the report by mail. Use a Gmail App Password, not your normal password.")

#=========================STEP 4: SESSION STATE========================#
# session_state is used so streamlit does not forget our expense table
# and chat history on every rerun
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns = ["Category", "Remark", "Amount"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#=====================STEP 5: INCOME INPUT======================
st.subheader("1. Monthly Income")
monthly_income = st.number_input("Enter Monthly Income (₹)", min_value = 0.0, step = 500.0)

#=====================STEP 6: ADD EXPENSE FORM====================
st.subheader("2. Add Expense")

col1, col2, col3, col4 = st.columns(4)

with col1:
    category = st.selectbox("Category", ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"])
    custom_category = ""
    if category == "Other":
        custom_category = st.text_input("Custom Category Name")
with col2:
    remark = st.text_input("Remark (optional)", placeholder = "e.g. Zomato order, movie tickets")
with col3:
    amount = st.number_input("Amount (₹)", min_value = 0.0, step = 50.0, key = "amount_input")
with col4:
    st.write("")
    st.write("")
    add_clicked = st.button("➕ Add Expense")

if add_clicked:
    final_category = custom_category.strip() if category == "Other" and custom_category.strip() else category
    if amount > 0:
        new_row = pd.DataFrame([{"Category": final_category, "Remark": remark, "Amount": amount}])
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index = True)
        st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors = "coerce").fillna(0.0)
        st.success(f"Added {final_category} : ₹{amount}")
    else:
        st.warning("Amount should be greater than 0")

#=====================STEP 7: SHOW EXPENSE TABLE====================
st.subheader("3. Expense Table")

if st.session_state.expenses.empty:
    st.info("No expenses added yet")
else:
    st.dataframe(st.session_state.expenses, use_container_width = True)
    if st.button("🗑️ Clear All"):
        st.session_state.expenses = pd.DataFrame(columns = ["Category", "Remark", "Amount"])
        st.rerun()

#=====================STEP 8: CALCULATE TOTALS====================
total_expense = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
balance = monthly_income - total_expense
savings = balance

st.subheader("4. Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Income", f"₹{monthly_income:,.2f}")
c2.metric("Total Expense", f"₹{total_expense:,.2f}")
c3.metric("Balance", f"₹{balance:,.2f}")
c4.metric("Savings", f"₹{savings:,.2f}")

#=====================STEP 9: PIE CHART====================
st.subheader("5. Expense Breakdown")

if not st.session_state.expenses.empty:
    category_totals = st.session_state.expenses.groupby("Category")["Amount"].sum().astype(float)
    fig, ax = plt.subplots()
    ax.pie(category_totals.values, labels = category_totals.index, autopct = "%1.1f%%", startangle = 90)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("Add expenses to see the chart")

#=====================STEP 10: FORECAST / TRENDS====================
st.subheader("6. Forecast & Trends")
st.caption("Simple projection based on this month's numbers, not a guarantee of the future.")

forecast_type = st.radio("Forecast Horizon", ["Monthly", "Yearly"], horizontal = True)

if monthly_income > 0:
    if forecast_type == "Monthly":
        months = list(range(1, 7))
        projected_savings = [savings * m for m in months]
        x_label = "Next 6 Months"
    else:
        months = list(range(1, 6))
        projected_savings = [savings * 12 * m for m in months]
        x_label = "Next 5 Years"

    fig2, ax2 = plt.subplots()
    ax2.plot(months, projected_savings, marker = "o", color = "#1E7F4F")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Projected Savings (₹)")
    ax2.set_title("Savings Trend Projection")
    st.pyplot(fig2)

    st.write(f"Projected Yearly Expense : ₹{total_expense * 12:,.2f}")
    st.write(f"Projected Yearly Savings : ₹{savings * 12:,.2f}")
else:
    st.info("Enter income to see forecast")

#=====================STEP 11: SAVING TIPS====================
st.subheader("7. Saving Tips")

if monthly_income > 0:
    savings_rate = (savings / monthly_income) * 100

    if savings_rate < 20:
        st.warning(f"Your savings are only {savings_rate:.1f}% of income. Try to save at least 20%.")
    else:
        st.success(f"Good job! You are saving {savings_rate:.1f}% of your income.")

    if not st.session_state.expenses.empty:
        food_total = st.session_state.expenses.loc[st.session_state.expenses["Category"] == "Food", "Amount"].sum()
        if food_total > 0.15 * monthly_income:
            st.warning(f"Food expense ₹{food_total:,.2f} is high. Try cooking at home more.")

    if total_expense > monthly_income:
        st.error("You are spending more than you earn this month!")

    st.markdown("""
    - Automate a fixed amount to savings on salary day itself.
    - Track small daily expenses too, they add up fast.
    - Review subscriptions every 3 months and cancel unused ones.
    - Keep an emergency fund of at least 3 months expenses.
    """)
else:
    st.info("Enter income to see tips")

#=====================STEP 12: AI FINANCIAL ADVISOR====================
st.subheader("8. AI Financial Advisor")

def get_budget_context():
    """This function builds a text summary of the
    current budget, used both for AI advice and chat"""
    expense_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
    context = f"""
    Monthly Income: ₹{monthly_income}
    Total Expense: ₹{total_expense}
    Balance: ₹{balance}
    Savings: ₹{savings}
    Expense Breakdown:
    {expense_summary}
    """
    return context

def get_ai_advice():
    """This function sends the budget summary
    to gemini model and returns personalized
    financial advice based on given data"""
    prompt = f"""
    You are a friendly personal finance advisor.
    {get_budget_context()}

    Give me 3-5 short, practical tips to improve my budget and savings.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

if st.button("🤖 Get AI Financial Advice"):
    if not GEMINI_API_KEY:
        st.error("Give API key first in the sidebar")
    elif monthly_income == 0:
        st.warning("Enter monthly income first")
    else:
        with st.spinner("Asking Gemini for advice.."):
            try:
                advice = get_ai_advice()
                st.markdown("### 💡 Gemini's Advice")
                st.write(advice)
            except Exception as err:
                st.error(f"Error Code: {err}")

#=====================STEP 13: CHAT WITH PLANNER====================
st.subheader("9. Chat with your Planner 💬")
st.caption("Ask anything about your budget, like 'where can I cut costs' or 'can I afford a ₹5000 trip'.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask your budget planner...")

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.write(user_msg)

    if not GEMINI_API_KEY:
        reply = "Give API key first in the sidebar to use chat."
    else:
        try:
            chat_prompt = f"""
            You are a helpful budget planning assistant.
            Here is the user's current budget data:
            {get_budget_context()}

            User question: {user_msg}
            Answer clearly and keep it short.
            """
            model = genai.GenerativeModel("gemini-1.5-flash")
            reply = model.generate_content(chat_prompt).text
        except Exception as err:
            reply = f"Error Code: {err}"

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

#=====================STEP 14: DISCLAIMER====================
st.warning("⚠️ Disclaimer: Yeh AI advice sirf general guidance ke liye hai. Kripya koi bhi bada financial decision lene se pehle ek certified financial adviser se consult karo.")

#=====================STEP 15: DOWNLOAD CSV====================
st.subheader("10. Export Data")

if not st.session_state.expenses.empty:
    csv_data = st.session_state.expenses.to_csv(index = False)
    st.download_button("⬇️ Download Expenses CSV", data = csv_data, file_name = "my_expenses.csv", mime = "text/csv")
else:
    st.info("Add expenses to enable download")

#=====================STEP 16: SEND REPORT VIA MAIL / WHATSAPP====================
st.subheader("11. Share Report")

def build_report_text():
    """This function builds the plain text report
    used for both email and whatsapp share"""
    report = f"""AI Personal Budget Planner Report

Monthly Income: Rs.{monthly_income:,.2f}
Total Expense: Rs.{total_expense:,.2f}
Balance: Rs.{balance:,.2f}
Savings: Rs.{savings:,.2f}

Expense Breakdown:
{st.session_state.expenses.groupby('Category')['Amount'].sum().to_string() if not st.session_state.expenses.empty else 'No expenses recorded'}

Generated by AI Personal Budget Planner"""
    return report

report_text = build_report_text()

colA, colB = st.columns(2)

with colA:
    st.write("**Send via Email**")
    recipient_email = st.text_input("Recipient Email")
    if st.button("📧 Send Report by Mail"):
        if not (SENDER_EMAIL and SENDER_APP_PASSWORD):
            st.error("Give sender email and app password in the sidebar first")
        elif not recipient_email:
            st.warning("Give recipient email")
        else:
            with st.spinner("Sending mail.."):
                try:
                    msg = MIMEText(report_text)
                    msg["Subject"] = "Your Monthly Budget Report"
                    msg["From"] = SENDER_EMAIL
                    msg["To"] = recipient_email

                    context = ssl.create_default_context()
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls(context = context)
                        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
                        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
                    st.success("Report sent successfully!!")
                except Exception as err:
                    st.error(f"Error Code: {err}")

with colB:
    st.write("**Send via WhatsApp**")
    whatsapp_text = urllib.parse.quote(report_text)
    whatsapp_url = f"https://wa.me/?text={whatsapp_text}"
    st.link_button("💬 Share on WhatsApp", whatsapp_url)
    st.caption("Opens WhatsApp, pick a contact and hit send.")
