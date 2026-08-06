#======STEP 1: LOAD MODULES=====#
import os
import smtplib
import urllib.parse
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title = "AI Personal Budget Planner", layout = "wide")
st.title("AI Personal Budget Planner 💰")
st.header("""Track your income, expenses, and get AI powered budget advice""")

#===============STEP 2: LOAD API-KEY==============
st.sidebar.title("Give API KEY")
GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type = "password")

if GEMINI_API_KEY:
    st.sidebar.success("API key Loaded!!")
else:
    st.sidebar.info("Give API key")
    url = "https://aistudio.google.com/app/apikey"
    st.sidebar.markdown(f"Get Gemini API Key-{url}")

#=========================STEP 3: SESSION STATE========================#
# session_state is used so streamlit does not forget our expense table,
# chat history, and savings goal on every rerun
if "expenses" not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns = ["Category", "Amount", "Remark"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "goal" not in st.session_state:
    st.session_state.goal = {"name": "", "amount": 0.0, "months": 0, "type": "Short-term (< 1 year)"}

#=====================STEP 4: INCOME INPUT======================
st.subheader("Monthly Income")
monthly_income = st.number_input("Enter Monthly Income (₹)", min_value = 0.0, step = 500.0)

#=====================STEP 5: ADD EXPENSE FORM====================
st.subheader("Add Expense")

col1, col2, col3, col4 = st.columns([1, 1, 1.4, 0.8])

with col1:
    category = st.selectbox("Category", ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"])
with col2:
    amount = st.number_input("Amount (₹)", min_value = 0.0, step = 50.0, key = "amount_input")
with col3:
    remark = st.text_input("Custom Name / Remark (optional)", key = "remark_input",
                            placeholder = "e.g. Netflix, Diwali shopping...")
with col4:
    st.write("")
    st.write("")
    add_clicked = st.button("➕ Add Expense")

if add_clicked:
    if amount > 0:
        # if "Other" is picked and a remark is given, use the remark as the
        # visible category name so the table shows something meaningful
        display_category = remark.strip() if (category == "Other" and remark.strip()) else category
        new_row = pd.DataFrame([{"Category": display_category, "Amount": amount, "Remark": remark.strip()}])
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index = True)
        st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors = "coerce").fillna(0.0)
        st.success(f"Added {display_category} : ₹{amount}")
    else:
        st.warning("Amount should be greater than 0")

#=====================STEP 6: SHOW EXPENSE TABLE====================
st.subheader("Expense Table")

if st.session_state.expenses.empty:
    st.info("No expenses added yet")
else:
    st.dataframe(st.session_state.expenses, use_container_width = True)
    if st.button("🗑️ Clear All"):
        st.session_state.expenses = pd.DataFrame(columns = ["Category", "Amount", "Remark"])
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
        food_total = st.session_state.expenses.loc[
            st.session_state.expenses["Category"].astype(str).str.lower() == "food", "Amount"
        ].sum()
        if food_total > 0.15 * monthly_income:
            st.warning(f"Food expense ₹{food_total:,.2f} is high. Try cooking at home more.")

    if total_expense > monthly_income:
        st.error("You are spending more than you earn this month!")
else:
    st.info("Enter income to see tips")

#=====================STEP 10: SAVINGS GOAL PLANNER====================
st.subheader("🎯 Savings Goal Planner")

g1, g2, g3, g4 = st.columns(4)
with g1:
    goal_name = st.text_input("Goal Name", value = st.session_state.goal["name"], placeholder = "e.g. Emergency Fund")
with g2:
    goal_amount = st.number_input("Target Amount (₹)", min_value = 0.0, step = 1000.0, value = st.session_state.goal["amount"])
with g3:
    goal_months = st.number_input("Timeframe (months)", min_value = 0, step = 1, value = st.session_state.goal["months"])
with g4:
    goal_type = st.selectbox(
        "Saving Type",
        ["Short-term (< 1 year)", "Long-term (>= 1 year)"],
        index = 0 if st.session_state.goal["type"].startswith("Short") else 1
    )

st.session_state.goal = {"name": goal_name, "amount": goal_amount, "months": goal_months, "type": goal_type}

if goal_amount > 0 and goal_months > 0:
    required_monthly_saving = goal_amount / goal_months
    st.write(
        f"To reach **{goal_name or 'your goal'}** (₹{goal_amount:,.2f}) in {goal_months} months, "
        f"you need to save **₹{required_monthly_saving:,.2f}/month**."
    )

    progress = min(max(savings / required_monthly_saving, 0.0), 1.0) if required_monthly_saving > 0 else 0.0
    st.progress(progress)

    if savings >= required_monthly_saving:
        st.success("You're on track to hit this goal at your current savings rate! 🎉")
    else:
        shortfall = required_monthly_saving - savings
        st.warning(f"You're short by ₹{shortfall:,.2f}/month to hit this goal on time.")
else:
    st.info("Set a goal amount and timeframe to see your plan")

#=====================STEP 11: BUDGET FORECAST====================
st.subheader("📈 Budget Forecast")

f1, f2, f3, f4 = st.columns(4)
with f1:
    forecast_trend_type = st.selectbox("Forecast Trend Type", ["Monthly", "Yearly"])
with f2:
    forecast_periods = st.number_input(
        f"Number of {forecast_trend_type.lower()}s to forecast",
        min_value = 1, max_value = 60, value = 6, step = 1
    )
with f3:
    income_growth = st.number_input("Expected Income Growth (% per period)", value = 0.0, step = 0.5)
with f4:
    expense_growth = st.number_input("Expected Expense Growth (% per period)", value = 0.0, step = 0.5)

if monthly_income > 0:
    periods = list(range(1, int(forecast_periods) + 1))
    forecast_income = [monthly_income * ((1 + income_growth / 100) ** p) for p in periods]
    forecast_expense = [total_expense * ((1 + expense_growth / 100) ** p) for p in periods]
    forecast_balance = [i - e for i, e in zip(forecast_income, forecast_expense)]

    forecast_df = pd.DataFrame({
        forecast_trend_type: periods,
        "Projected Income (₹)": [f"{v:,.2f}" for v in forecast_income],
        "Projected Expense (₹)": [f"{v:,.2f}" for v in forecast_expense],
        "Projected Savings (₹)": [f"{v:,.2f}" for v in forecast_balance],
    })
    st.dataframe(forecast_df, use_container_width = True)

    fig2, ax2 = plt.subplots()
    ax2.plot(periods, forecast_income, marker = "o", label = "Income")
    ax2.plot(periods, forecast_expense, marker = "o", label = "Expense")
    ax2.plot(periods, forecast_balance, marker = "o", label = "Savings")
    ax2.set_xlabel(forecast_trend_type)
    ax2.set_ylabel("Amount (₹)")
    ax2.set_title(f"{forecast_trend_type} Forecast for the next {forecast_periods} {forecast_trend_type.lower()}(s)")
    ax2.legend()
    st.pyplot(fig2)

    cumulative_savings = sum(forecast_balance)
    st.info(
        f"Projected cumulative savings over {forecast_periods} {forecast_trend_type.lower()}(s): "
        f"₹{cumulative_savings:,.2f}"
    )
else:
    st.info("Enter income to see forecast")

#=====================STEP 12: AI FINANCIAL ADVISOR (LangChain + Gemini)====================
st.subheader("AI Financial Advisor")

def get_ai_advice(income, expense, balance, savings, expense_summary):
    """This function sends the budget summary
    to gemini model (via langchain) and returns personalized
    financial advice based on given data"""
    prompt = ChatPromptTemplate.from_template(
        """
        You are a friendly personal finance advisor.
        Monthly Income: ₹{income}
        Total Expense: ₹{expense}
        Balance: ₹{balance}
        Savings: ₹{savings}
        Expense Breakdown:
        {expense_summary}

        Give me 3-5 short, practical tips to improve my budget and savings.
        """
    )

    model = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash-lite",
        google_api_key = GEMINI_API_KEY,
        temperature = 0.7
    )

    chain = prompt | model | StrOutputParser()

    response = chain.invoke({
        "income": income,
        "expense": expense,
        "balance": balance,
        "savings": savings,
        "expense_summary": expense_summary
    })

    return response

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

#=====================STEP 13: CHAT WITH YOUR PLANNER====================
st.subheader("💬 Chat with Your Budget Planner")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_msg = st.chat_input("Ask about your budget, savings, or forecast...")

if user_msg:
    if not GEMINI_API_KEY:
        st.error("Give API key first in the sidebar")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        budget_context = f"""
        Monthly Income: ₹{monthly_income}
        Total Expense: ₹{total_expense}
        Balance: ₹{balance}
        Savings Goal: {st.session_state.goal.get('name', 'N/A')} - target ₹{st.session_state.goal.get('amount', 0)} in {st.session_state.goal.get('months', 0)} months
        """

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a friendly, concise personal finance advisor. "
             "Use the budget context below to answer the user's question.\n"
             "Context:\n" + budget_context),
            MessagesPlaceholder(variable_name = "history"),
            ("human", "{input}")
        ])

        chat_model = ChatGoogleGenerativeAI(
            model = "gemini-3.5-flash-lite",
            google_api_key = GEMINI_API_KEY,
            temperature = 0.7
        )

        chat_chain = chat_prompt | chat_model | StrOutputParser()

        history_messages = []
        for m in st.session_state.chat_history[:-1]:
            if m["role"] == "user":
                history_messages.append(HumanMessage(content = m["content"]))
            else:
                history_messages.append(AIMessage(content = m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = chat_chain.invoke({"history": history_messages, "input": user_msg})
                    st.write(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as err:
                    st.error(f"Error Code: {err}")

#=====================STEP 14: EXPORT & SHARE REPORT====================
st.subheader("📤 Export & Share Report")

report_text = (
    "AI Personal Budget Planner Report\n"
    "-----------------------------------\n"
    f"Monthly Income: Rs. {monthly_income:,.2f}\n"
    f"Total Expense: Rs. {total_expense:,.2f}\n"
    f"Balance: Rs. {balance:,.2f}\n"
    f"Savings: Rs. {savings:,.2f}\n\n"
    f"Savings Goal: {st.session_state.goal.get('name', 'N/A')}\n"
    f"Target: Rs. {st.session_state.goal.get('amount', 0):,.2f} in {st.session_state.goal.get('months', 0)} months\n"
)

exp_col1, exp_col2, exp_col3 = st.columns(3)

with exp_col1:
    st.markdown("**Download CSV**")
    if not st.session_state.expenses.empty:
        csv_data = st.session_state.expenses.to_csv(index = False)
        st.download_button("⬇️ Download Expenses CSV", data = csv_data, file_name = "my_expenses.csv", mime = "text/csv")
    else:
        st.info("Add expenses to enable download")

with exp_col2:
    st.markdown("**Send Report via Email**")
    with st.expander("Email settings"):
        sender_email = st.text_input("Your Email (Gmail)", key = "sender_email")
        sender_password = st.text_input("App Password", type = "password", key = "sender_password")
        receiver_email = st.text_input("Send Report To", key = "receiver_email")
        st.caption(
            "Use a Gmail App Password, not your normal password. "
            "Credentials are used only for this session and are not stored."
        )
        if st.button("📧 Send Email Report"):
            if not (sender_email and sender_password and receiver_email):
                st.warning("Fill in all email fields first")
            else:
                try:
                    msg = MIMEMultipart()
                    msg["From"] = sender_email
                    msg["To"] = receiver_email
                    msg["Subject"] = "My Budget Report"
                    msg.attach(MIMEText(report_text, "plain"))
                    with smtplib.SMTP("smtp.gmail.com", 587) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, receiver_email, msg.as_string())
                    st.success("Report sent via email!")
                except Exception as err:
                    st.error(f"Could not send email: {err}")

with exp_col3:
    st.markdown("**Send Report via WhatsApp**")
    whatsapp_number = st.text_input(
        "WhatsApp Number (with country code, no +)", key = "wa_number", placeholder = "919999999999"
    )
    if whatsapp_number:
        encoded_text = urllib.parse.quote(report_text)
        wa_link = f"https://wa.me/{whatsapp_number}?text={encoded_text}"
        st.link_button("💬 Share on WhatsApp", wa_link)
    else:
        st.info("Enter a WhatsApp number to share")

#=====================STEP 15: DISCLAIMER====================
st.divider()
with st.expander("⚠️ Disclaimer"):
    st.caption(
        "This app is for educational purposes only and does not constitute professional "
        "financial, investment, or tax advice. AI-generated advice, the chat responses, and the "
        "forecast are estimates based on the numbers you enter and simple growth assumptions — "
        "they are not guarantees of future performance. Please consult a certified financial "
        "advisor before making major financial decisions. Email credentials you enter above are "
        "used only to send the report during this session and are not stored or transmitted "
        "anywhere else by this app."
    )
