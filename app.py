#======STEP 1: LOAD MODULES=====#
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import google.generativeai as genai

st.set_page_config(page_title = "AI Personal Budget Planner", layout = "wide")
st.title("AI Personal Budget Planner 💰")
st.header("""Track your income, expenses, and get AI powered budget advice""")

#===============STEP 2: LOAD API-KEY==============
st.sidebar.title("Give API KEY")
GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type = "password")

if GEMINI_API_KEY:
    st.sidebar.success("API key Loaded!!")
    genai.configure(api_key = GEMINI_API_KEY)
else:
    st.sidebar.info("Give API key")
    url = "https://aistudio.google.com/app/apikey"
    st.sidebar.markdown(f"Get Gemini API Key-{url}")

#=========================STEP 3: SESSION STATE========================#
# session_state is used so streamlit does not forget our expense table
# on every rerun
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns = ["Category", "Amount"])

#=====================STEP 4: INCOME INPUT======================
st.subheader("Monthly Income")
monthly_income = st.number_input("Enter Monthly Income (₹)", min_value = 0.0, step = 500.0)

#=====================STEP 5: ADD EXPENSE FORM====================
st.subheader("Add Expense")

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox("Category", ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"])
with col2:
    amount = st.number_input("Amount (₹)", min_value = 0.0, step = 50.0, key = "amount_input")
with col3:
    st.write("")
    st.write("")
    add_clicked = st.button("➕ Add Expense")

if add_clicked:
    if amount > 0:
        new_row = pd.DataFrame([{"Category": category, "Amount": amount}])
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index = True)
        st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors = "coerce").fillna(0.0)
        st.success(f"Added {category} : ₹{amount}")
    else:
        st.warning("Amount should be greater than 0")

#=====================STEP 6: SHOW EXPENSE TABLE====================
st.subheader("Expense Table")

if st.session_state.expenses.empty:
    st.info("No expenses added yet")
else:
    st.dataframe(st.session_state.expenses, use_container_width = True)
    if st.button("🗑️ Clear All"):
        st.session_state.expenses = pd.DataFrame(columns = ["Category", "Amount"])
        st.rerun()

#=====================STEP 7: CALCULATE TOTALS====================
total_expense = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
balance = monthly_income - total_expense
savings = balance

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Income", f"₹{monthly_income:,.2f}")
c2.metric("Total Expense", f"₹{total_expense:,.2f}")
c3.metric("Balance", f"₹{balance:,.2f}")
c4.metric("Savings", f"₹{savings:,.2f}")

#=====================STEP 8: PIE CHART====================
st.subheader("Expense Breakdown")

if not st.session_state.expenses.empty:
    category_totals = st.session_state.expenses.groupby("Category")["Amount"].sum().astype(float)
    fig, ax = plt.subplots()
    ax.pie(category_totals.values, labels = category_totals.index, autopct = "%1.1f%%", startangle = 90)
    ax.axis("equal")
    st.pyplot(fig)
else:
    st.info("Add expenses to see the chart")

#=====================STEP 9: RULE BASED TIPS====================
st.subheader("Quick Tips")

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
else:
    st.info("Enter income to see tips")

#=====================STEP 10: AI FINANCIAL ADVISOR====================
st.subheader("AI Financial Advisor")

def get_ai_advice(income, expense, balance, savings, expense_summary):
    """This function sends the budget summary
    to gemini model and returns personalized
    financial advice based on given data"""
    prompt = f"""
    You are a friendly personal finance advisor.
    Monthly Income: ₹{income}
    Total Expense: ₹{expense}
    Balance: ₹{balance}
    Savings: ₹{savings}
    Expense Breakdown:
    {expense_summary}

    Give me 3-5 short, practical tips to improve my budget and savings.
    """
    model = genai.GenerativeModel("gemini-3.5-flash-lite")
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
                expense_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
                advice = get_ai_advice(monthly_income, total_expense, balance, savings, expense_summary)
                st.markdown("### 💡 Gemini's Advice")
                st.write(advice)
            except Exception as err:
                st.error(f"Error Code: {err}")

#=====================STEP 11: DOWNLOAD CSV====================
st.subheader("Export Data")

if not st.session_state.expenses.empty:
    csv_data = st.session_state.expenses.to_csv(index = False)
    st.download_button("⬇️ Download Expenses CSV", data = csv_data, file_name = "my_expenses.csv", mime = "text/csv")
else:
    st.info("Add expenses to enable download")
