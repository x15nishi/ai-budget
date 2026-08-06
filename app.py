#======Modules=====#
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title = "AI Personal Budget Planner", layout = "wide", page_icon = "💰")

#===============STEP 2 LOAD API-KEY==============
st.title("AI Personal Budget Planner 💰")
st.header("""Track income, expenses, and get AI powered budget advice""")

st.sidebar.title("Give API KEY")

GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type = "password")

ALL_API = [GEMINI_API_KEY]

if not all(ALL_API):
    st.sidebar.error("Must Pass API-KEY")

    url = "https://aistudio.google.com/app/apikey"
    st.sidebar.markdown(f"Get Gemini API Key-{url}")

elif all(ALL_API):
    st.sidebar.success("API KEY LOADED")

    model = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash-lite",
        google_api_key = GEMINI_API_KEY
    )
else:
    st.sidebar.info("Try Valid API-key")

#=========================Step 3 backend============================#
DATA_FILE = "data.csv"
cols = ["Category", "Remark", "Amount"]

# load_expenses
def load_expenses():
  """This function helps to load
  saved expense csv from disk, and
  gives back empty table if file missing
  or columns dont match"""

  if os.path.exists(DATA_FILE):
    try:
      df = pd.read_csv(DATA_FILE)
      if list(df.columns) != cols:
        df = pd.DataFrame(columns = cols)
    except Exception:
      df = pd.DataFrame(columns = cols)
  else:
    df = pd.DataFrame(columns = cols)

  df["Amount"] = pd.to_numeric(df["Amount"], errors = "coerce").fillna(0.0)
  return df


# save_expenses
def save_expenses(df):
  """This function saves given
  expense dataframe back to the
  csv file on disk"""

  df.to_csv(DATA_FILE, index = False)


# url_encode
def url_encode(text):
  """This function does manual
  url encoding for building mail
  and whatsapp share links, no
  need full urllib for this"""

  result = ""
  for ch in text:
    if ch.isalnum() or ch in "-_.~":
      result += ch
    elif ch == " ":
      result += "%20"
    elif ch == "\n":
      result += "%0A"
    else:
      result += "%" + format(ord(ch), "02X")
  return result


# get_ai_reply
def get_ai_reply(prompt):
  """This function calls the
  loaded gemini model with given
  prompt and returns text response"""

  response = model.invoke([HumanMessage(content = prompt)])
  return response.content


if "expenses" not in st.session_state:
    st.session_state.expenses = load_expenses()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

#=====================Step 4 STREAMLIT NAVBARS ===============================#
st.subheader("1. Monthly Income")
income = st.number_input("Enter Monthly Income (₹)", min_value = 0.0, step = 500.0)

tab_add, tab_dash, tab_forecast, tab_advisor, tab_chat, tab_share = st.tabs(
    ["Add Expense", "Dashboard", "Forecast",
     "AI Advisor", "Chat", "Share/Export"])

with tab_add:
    st.subheader("Add Expense")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        category = st.selectbox("Category", ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"])
        custom_cat = ""
        if category == "Other":
            custom_cat = st.text_input("Custom Category Name")
    with c2:
        remark = st.text_input("Remark (optional)")
    with c3:
        amount = st.number_input("Amount (₹)", min_value = 0.0, step = 50.0)
    with c4:
        st.write("")
        st.write("")
        add_btn = st.button("Add Expense", key = "Add-Button")

    if add_btn:
        final_cat = custom_cat.strip() if category == "Other" and custom_cat.strip() else category
        if amount > 0:
            new_row = pd.DataFrame([{"Category": final_cat, "Remark": remark, "Amount": amount}])
            st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index = True)
            st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors = "coerce").fillna(0.0)
            save_expenses(st.session_state.expenses)
            st.success(f"Added {final_cat} : ₹{amount}")
        else:
            st.warning("amount should be more than 0")

    st.write("---")
    st.subheader("Expense Table")

    if st.session_state.expenses.empty:
        st.info("No expenses added yet")
    else:
        edited = st.data_editor(st.session_state.expenses, use_container_width = True, num_rows = "dynamic")
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("Save Changes", key = "Save-Button"):
                edited["Amount"] = pd.to_numeric(edited["Amount"], errors = "coerce").fillna(0.0)
                st.session_state.expenses = edited
                save_expenses(st.session_state.expenses)
                st.success("saved")
        with col_clear:
            if st.button("Clear All", key = "Clear-Button"):
                st.session_state.expenses = pd.DataFrame(columns = cols)
                save_expenses(st.session_state.expenses)
                st.rerun()

    st.write("---")
    st.subheader("Category Budget Limit (optional)")
    st.caption("set a limit for a category, resets when app restarts")

    bcol = st.columns(4)
    limit_cats = ["Food", "Rent", "Transport", "Shopping"]
    for i, cat in enumerate(limit_cats):
        with bcol[i]:
            st.session_state.budgets[cat] = st.number_input(cat + " Limit", min_value = 0.0, step = 500.0, key = "lim_" + cat)

with tab_dash:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp
    savings = balance

    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Income", f"₹{income:,.2f}")
    m2.metric("Total Expense", f"₹{total_exp:,.2f}")
    m3.metric("Balance", f"₹{balance:,.2f}")
    m4.metric("Savings", f"₹{savings:,.2f}")

    if not st.session_state.expenses.empty:
        cat_totals = st.session_state.expenses.groupby("Category")["Amount"].sum()

        colp, colb = st.columns(2)
        with colp:
            st.write("Expense Breakdown")
            fig1, ax1 = plt.subplots()
            ax1.pie(cat_totals.values, labels = cat_totals.index, autopct = "%1.1f%%")
            st.pyplot(fig1)
        with colb:
            st.write("Expense by Category")
            fig2, ax2 = plt.subplots()
            ax2.bar(cat_totals.index, cat_totals.values)
            plt.xticks(rotation = 30)
            st.pyplot(fig2)

        st.write("---")
        st.subheader("Budget Limit Check")
        for cat, lim in st.session_state.budgets.items():
            if lim > 0:
                spent = st.session_state.expenses.loc[st.session_state.expenses["Category"] == cat, "Amount"].sum()
                st.write(cat, ":", spent, "/", lim)
                st.progress(min(spent / lim, 1.0))
                if spent > lim:
                    st.error(cat + " budget over ho gaya!")
    else:
        st.info("add some expenses to see chart")

    st.write("---")
    st.subheader("Saving Tips")
    if income > 0:
        rate = (savings / income) * 100
        if rate < 20:
            st.warning(f"savings is only {rate:.1f}% of income, try to save atleast 20%")
        else:
            st.success(f"nice, you are saving {rate:.1f}% of income")

        if not st.session_state.expenses.empty:
            food_amt = st.session_state.expenses.loc[st.session_state.expenses["Category"] == "Food", "Amount"].sum()
            if food_amt > 0.15 * income:
                st.warning("food expense is quite high, try cooking at home more often")

        if total_exp > income:
            st.error("you are spending more than you earn this month!")

        st.markdown("""
        - try to save money as soon as salary comes  
        - track small expenses too, they add up  
        - cancel subscriptions you dont use  
        - keep some emergency fund
        """)
    else:
        st.info("enter income to see tips")

with tab_forecast:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    st.subheader("Forecast")

    if not st.session_state.expenses.empty and income > 0:
        horizon = st.radio("Forecast for", ["Monthly", "Yearly"], horizontal = True)
        proj_saving = income - total_exp

        if horizon == "Monthly":
            x = list(range(1, 7))
            y = [proj_saving * i for i in x]
            xlabel = "next 6 months"
        else:
            x = list(range(1, 6))
            y = [proj_saving * 12 * i for i in x]
            xlabel = "next 5 years"

        fig4, ax4 = plt.subplots()
        ax4.plot(x, y, marker = "o", color = "green")
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel("Projected Savings")
        st.pyplot(fig4)

        st.write("Current Monthly Expense:", round(total_exp, 2))
        st.write("Projected Yearly Expense:", round(total_exp * 12, 2))
        st.write("Projected Yearly Savings:", round(proj_saving * 12, 2))
    else:
        st.info("enter income and add expenses first")

with tab_advisor:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    st.subheader("AI Financial Advisor")
    if st.button("Get AI Financial Advice", key = "Advisor-Button"):
        if not all(ALL_API):
            st.error("give api key in sidebar first")
        elif income == 0:
            st.warning("enter monthly income first")
        else:
            with st.spinner("Running Agent"):
                try:
                    exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"

                    prompt = """You are a friendly personal finance advisor.
                    Give 3-5 short practical tips to improve budget and savings
                    based on below given numbers:"""

                    prompt = prompt + f"\nMonthly Income: {income}\nTotal Expense: {total_exp}\nBalance: {balance}\nExpense Breakdown:\n{exp_summary}"

                    result = get_ai_reply(prompt)
                    st.markdown("### Gemini's Advice")
                    st.write(result)
                except Exception as err:
                    st.error(f"Error Code: {err}")

    st.write("---")
    st.warning("⚠️ Disclaimer: Yeh AI advice sirf general guidance ke liye hai. Kripya koi bhi bada financial decision lene se pehle ek certified financial adviser se consult karo.")

with tab_chat:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    st.subheader("Chat with your Planner")
    st.caption("ask stuff like 'where can i cut cost' or 'can i afford a 5000 trip'")

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    q = st.chat_input("Ask something...")

    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)

        if not all(ALL_API):
            ans = "give api key in sidebar to use chat"
        else:
            try:
                exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
                chat_prompt = f"""You are a budget planning assistant.
                Income: {income}, Expense: {total_exp}, Balance: {balance}
                Expense breakdown: {exp_summary}
                Question: {q}
                Answer short and clear."""
                ans = get_ai_reply(chat_prompt)
            except Exception as err:
                ans = f"Error Code: {err}"

        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)

with tab_share:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    st.subheader("Export Data")
    if not st.session_state.expenses.empty:
        csv_data = st.session_state.expenses.to_csv(index = False)
        st.download_button("Download CSV", data = csv_data, file_name = "my_expenses.csv", mime = "text/csv")
    else:
        st.info("add expenses first to download")

    st.write("---")
    st.subheader("Share Report")

    exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
    report_text = f"""AI Personal Budget Planner Report

Monthly Income: Rs.{income:,.2f}
Total Expense: Rs.{total_exp:,.2f}
Balance: Rs.{balance:,.2f}

Expense Breakdown:
{exp_summary}

- generated by AI Personal Budget Planner"""

    colA, colB = st.columns(2)

    with colA:
        st.write("Send via Email")
        to_email = st.text_input("Recipient Email")

        subject = "Your Monthly Budget Report"
        mail_subject = url_encode(subject)
        mail_body = url_encode(report_text)

        if st.button("Prepare Mail", key = "Mail-Button"):
            if not to_email:
                st.warning("enter recipient email first")
            else:
                to_encoded = url_encode(to_email)
                gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={to_encoded}&su={mail_subject}&body={mail_body}"
                mailto_url = f"mailto:{to_email}?subject={mail_subject}&body={mail_body}"

                st.success("mail template ready, click below to send")
                st.link_button("Open in Gmail (web)", gmail_url)
                st.link_button("Open in Mail App", mailto_url)
                st.caption("this opens the mail already filled in, using your own logged in gmail / mail app")

    with colB:
        st.write("Send via WhatsApp")
        wa_text = url_encode(report_text)
        wa_url = "https://wa.me/?text=" + wa_text
        st.link_button("Share on WhatsApp", wa_url)
        st.caption("opens whatsapp, pick a contact and send")

# TODO: maybe add monthly comparison graph later
# TODO: pdf export instead of just csv
