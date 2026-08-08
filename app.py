# AI Personal Budget Planner
# Mini Project - Python + Streamlit + Gemini API (via LangChain)
# made this to track monthly budget and get AI tips

import os
import json
import base64
import urllib.parse

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()

DATA_FILE = "data.csv"
cols = ["Category", "Remark", "Amount"]  # columns for the expense table
CATEGORY_OPTIONS = ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"]

st.set_page_config(page_title="AI Personal Budget Planner", layout="wide", page_icon="💰")

# little bit of css just to make metric boxes look nice, took this from streamlit forum
st.markdown("""
<style>
.stMetric {background-color:#F5F7FA; padding:10px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("AI Personal Budget Planner 💰")
st.write("Track your income, expenses, and get AI powered budget advice")

# ---------------- API KEY (sidebar) ----------------
st.sidebar.title("Give API KEY")
env_key = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = st.sidebar.text_input("GEMINI_API_KEY", type="password", value=env_key if env_key else "")

if GEMINI_API_KEY:
    st.sidebar.success("API key Loaded!!")
else:
    st.sidebar.info("Give API key")
    st.sidebar.markdown("Get key from https://aistudio.google.com/app/apikey")


# small helper so we don't repeat the same ChatGoogleGenerativeAI setup everywhere
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )


# loads expense data from csv, if file not there just make empty table
def load_expenses():
    try:
        df = pd.read_csv(DATA_FILE)
        if list(df.columns) != cols:
            df = pd.DataFrame(columns=cols)
    except:
        df = pd.DataFrame(columns=cols)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def save_expenses(df):
    df.to_csv(DATA_FILE, index=False)


# ---------------- OCR BILL SCAN (Gemini Vision) ----------------
# Sends the uploaded/captured bill image straight to Gemini (multimodal) and asks
# it to return structured JSON line items. No separate OCR engine needed since
# Gemini can read the image directly.
def extract_expenses_from_bill(image_bytes, mime_type):
    model = get_gemini_model()
    b64_img = base64.b64encode(image_bytes).decode("utf-8")

    instructions = (
        "You are reading a photo of a shopping receipt / bill / invoice. "
        "Extract each distinct expense line you can clearly identify. "
        "Respond with ONLY valid JSON (no markdown fences, no explanation text), "
        "as a JSON array like this exact shape:\n"
        '[{"category": "Food", "remark": "short item or store name", "amount": 123.45}]\n'
        "Rules:\n"
        f"- category MUST be exactly one of: {', '.join(CATEGORY_OPTIONS)}\n"
        "- amount must be a plain number (no currency symbols, no commas)\n"
        "- if you can only make out one grand total (not itemised), return a single "
        "object using the store/vendor name as remark and the total as amount\n"
        "- if the image is unreadable or not a bill, return an empty array []"
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": instructions},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_img}"},
        ]
    )

    response = model.invoke([message])

    # response.content can be a plain string OR a list of content blocks
    # (e.g. [{"type": "text", "text": "..."}]) depending on the model/response,
    # so normalize it to a string first
    raw_content = response.content
    if isinstance(raw_content, list):
        text_parts = []
        for block in raw_content:
            if isinstance(block, dict):
                text_parts.append(block.get("text", ""))
            else:
                text_parts.append(str(block))
        raw_text = "".join(text_parts).strip()
    else:
        raw_text = str(raw_content).strip()

    if not raw_text:
        return []

    # gemini sometimes wraps json in ```json ... ``` even when told not to, strip that off
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    items = json.loads(raw_text)

    # basic cleanup / validation so bad rows dont break the app
    cleaned = []
    for item in items:
        cat = item.get("category", "Other")
        if cat not in CATEGORY_OPTIONS:
            cat = "Other"
        try:
            amt = float(item.get("amount", 0))
        except (TypeError, ValueError):
            amt = 0.0
        remark = str(item.get("remark", "")).strip()
        if amt > 0:
            cleaned.append({"Category": cat, "Remark": remark, "Amount": amt})

    return cleaned


if "expenses" not in st.session_state:
    st.session_state.expenses = load_expenses()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

if "scanned_items" not in st.session_state:
    st.session_state.scanned_items = pd.DataFrame(columns=cols)

# ---------------- INCOME ----------------
st.subheader("1. Monthly Income")
income = st.number_input("Enter Monthly Income (₹)", min_value=0.0, step=500.0)

# using tabs so everything doesnt look like one giant scrolling page
tab_add, tab_dash, tab_forecast, tab_advisor, tab_chat, tab_share = st.tabs(
    ["Add Expense", "Dashboard", "Forecast", "AI Advisor", "Chat", "Share/Export"]
)

# =========================================================
# TAB 1 - ADD EXPENSE
# =========================================================
with tab_add:
    st.subheader("Add Expense")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        category = st.selectbox("Category", CATEGORY_OPTIONS)
        custom_cat = ""
        if category == "Other":
            custom_cat = st.text_input("Custom Category Name")
    with c2:
        remark = st.text_input("Remark (optional)")
    with c3:
        amount = st.number_input("Amount (₹)", min_value=0.0, step=50.0)
    with c4:
        st.write("")
        st.write("")
        add_btn = st.button("Add Expense")

    if add_btn:
        # agar user ne Other select kiya to custom name use karo
        final_cat = custom_cat.strip() if category == "Other" and custom_cat.strip() != "" else category

        if amount > 0:
            new_row = pd.DataFrame([{"Category": final_cat, "Remark": remark, "Amount": amount}])
            st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
            st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors="coerce").fillna(0.0)
            save_expenses(st.session_state.expenses)
            st.success("Added " + final_cat + " : ₹" + str(amount))
        else:
            st.warning("amount should be more than 0")

    st.write("---")
    st.subheader("📷 Scan a Bill / Receipt (auto add expenses)")
    st.caption("upload a photo or take a picture of your bill, AI will read it and fill in the expenses for you")

    if not GEMINI_API_KEY:
        st.info("give api key in sidebar first to use bill scanning")
    else:
        scan_c1, scan_c2 = st.columns(2)
        with scan_c1:
            uploaded_bill = st.file_uploader("Upload bill image", type=["jpg", "jpeg", "png", "webp"], key="bill_upload")
        with scan_c2:
            camera_bill = st.camera_input("Or take a photo")

        bill_file = camera_bill if camera_bill is not None else uploaded_bill

        if bill_file is not None:
            st.image(bill_file, caption="Bill preview", width=250)

            if st.button("🔍 Scan Bill with AI"):
                with st.spinner("reading your bill.."):
                    try:
                        img_bytes = bill_file.getvalue()
                        mime = bill_file.type if getattr(bill_file, "type", None) else "image/jpeg"
                        extracted = extract_expenses_from_bill(img_bytes, mime)

                        if not extracted:
                            st.warning("couldn't read any expense lines from this image, try a clearer photo")
                        else:
                            st.session_state.scanned_items = pd.DataFrame(extracted)
                            st.success(f"found {len(extracted)} item(s), review below before adding")
                    except json.JSONDecodeError:
                        st.error("AI didn't return clean data, try scanning again or use a clearer photo")
                    except Exception as e:
                        st.error("Error: " + str(e))

        # review + confirm scanned items before they go into the real expense table
        if not st.session_state.scanned_items.empty:
            st.write("Review scanned items (edit if needed):")
            reviewed = st.data_editor(
                st.session_state.scanned_items,
                use_container_width=True,
                num_rows="dynamic",
                key="scanned_editor",
            )

            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("✅ Add Scanned Items to Expenses"):
                    reviewed["Amount"] = pd.to_numeric(reviewed["Amount"], errors="coerce").fillna(0.0)
                    reviewed = reviewed[reviewed["Amount"] > 0]
                    st.session_state.expenses = pd.concat([st.session_state.expenses, reviewed], ignore_index=True)
                    save_expenses(st.session_state.expenses)
                    st.session_state.scanned_items = pd.DataFrame(columns=cols)
                    st.success("added scanned items to your expenses!")
                    st.rerun()
            with rc2:
                if st.button("❌ Discard Scanned Items"):
                    st.session_state.scanned_items = pd.DataFrame(columns=cols)
                    st.rerun()

    st.write("---")
    st.subheader("Expense Table")

    if st.session_state.expenses.empty:
        st.info("No expenses added yet")
    else:
        # data_editor lets user edit/delete rows directly, found this in streamlit docs
        edited = st.data_editor(st.session_state.expenses, use_container_width=True, num_rows="dynamic")

        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("Save Changes"):
                edited["Amount"] = pd.to_numeric(edited["Amount"], errors="coerce").fillna(0.0)
                st.session_state.expenses = edited
                save_expenses(st.session_state.expenses)
                st.success("saved")
        with col_clear:
            if st.button("Clear All"):
                st.session_state.expenses = pd.DataFrame(columns=cols)
                save_expenses(st.session_state.expenses)
                st.rerun()

    st.write("---")
    st.subheader("Category Budget Limit (optional)")
    st.caption("set a limit for a category, resets when you close the app (didnt save this to file yet)")

    bcol = st.columns(4)
    limit_cats = ["Food", "Rent", "Transport", "Shopping"]
    for i in range(len(limit_cats)):
        cat = limit_cats[i]
        with bcol[i]:
            st.session_state.budgets[cat] = st.number_input(cat + " Limit", min_value=0.0, step=500.0, key="lim_" + cat)

# =========================================================
# TAB 2 - DASHBOARD
# =========================================================
with tab_dash:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp
    savings = balance  # savings = balance basically, keeping separate var for readability

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
            st.write("Expense Breakdown (Pie)")
            fig1, ax1 = plt.subplots()
            ax1.pie(cat_totals.values, labels=cat_totals.index, autopct="%1.1f%%")
            st.pyplot(fig1)

        with colb:
            st.write("Expense by Category (Bar)")
            fig2, ax2 = plt.subplots()
            ax2.bar(cat_totals.index, cat_totals.values)
            plt.xticks(rotation=30)
            st.pyplot(fig2)

        st.write("---")
        st.subheader("Budget Limit Check")
        for cat in st.session_state.budgets:
            lim = st.session_state.budgets[cat]
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

# =========================================================
# TAB 3 - FORECAST
# =========================================================
with tab_forecast:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0

    st.subheader("Forecast")
    st.caption("simple projection based on your current numbers")

    if not st.session_state.expenses.empty and income > 0:
        horizon = st.radio("Forecast for", ["Monthly", "Yearly"], horizontal=True)
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
        ax4.plot(x, y, marker="o", color="green")
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel("Projected Savings")
        st.pyplot(fig4)

        st.write("Current Monthly Expense:", round(total_exp, 2))
        st.write("Projected Yearly Expense:", round(total_exp * 12, 2))
        st.write("Projected Yearly Savings:", round(proj_saving * 12, 2))
    else:
        st.info("enter income and add expenses first")

# =========================================================
# TAB 4 - AI ADVISOR
# =========================================================
with tab_advisor:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    # this just builds a text summary of budget to send to gemini
    def get_context():
        exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
        return f"""
        Monthly Income: {income}
        Total Expense: {total_exp}
        Balance: {balance}
        Expense Breakdown:
        {exp_summary}
        """

    st.subheader("AI Financial Advisor")
    if st.button("Get AI Financial Advice"):
        if not GEMINI_API_KEY:
            st.error("give api key in sidebar first")
        elif income == 0:
            st.warning("enter monthly income first")
        else:
            with st.spinner("asking gemini.."):
                try:
                    advisor_prompt = ChatPromptTemplate.from_template(
                        "You are a friendly personal finance advisor. {context} "
                        "Give me 3-5 short practical tips to improve my budget and savings."
                    )
                    chain = advisor_prompt | get_gemini_model() | StrOutputParser()
                    result = chain.invoke({"context": get_context()})
                    st.markdown("### Gemini's Advice")
                    st.write(result)
                except Exception as e:
                    st.error("Error: " + str(e))

    st.write("---")
    st.warning("⚠️ Disclaimer: Yeh AI advice sirf general guidance ke liye hai. Kripya koi bhi bada financial decision lene se pehle ek certified financial adviser se consult karo.")

# =========================================================
# TAB 5 - CHAT
# =========================================================
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

        if not GEMINI_API_KEY:
            ans = "give api key in sidebar to use chat"
        else:
            try:
                exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
                chat_prompt = ChatPromptTemplate.from_template(
                    """You are a budget planning assistant.
                    Income: {income}, Expense: {total_exp}, Balance: {balance}
                    Expense breakdown: {exp_summary}
                    Question: {question}
                    Answer short and clear."""
                )
                chain = chat_prompt | get_gemini_model() | StrOutputParser()
                ans = chain.invoke({
                    "income": income,
                    "total_exp": total_exp,
                    "balance": balance,
                    "exp_summary": exp_summary,
                    "question": q
                })
            except Exception as e:
                ans = "Error: " + str(e)

        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)

# =========================================================
# TAB 6 - SHARE / EXPORT
# =========================================================
with tab_share:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    st.subheader("Export Data")
    if not st.session_state.expenses.empty:
        csv_data = st.session_state.expenses.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="my_expenses.csv", mime="text/csv")
    else:
        st.info("add expenses first to download")

    st.write("---")
    st.subheader("Share Report")

    # builds the text report we send over email/whatsapp
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

        # just building the mail as a template here, not sending it ourself
        # this opens gmail in browser (already logged in) or default mail app
        subject = "Your Monthly Budget Report"
        mail_subject = urllib.parse.quote(subject)
        mail_body = urllib.parse.quote(report_text)

        if st.button("Prepare Mail"):
            if not to_email:
                st.warning("enter recipient email first")
            else:
                to_encoded = urllib.parse.quote(to_email)
                gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={to_encoded}&su={mail_subject}&body={mail_body}"
                mailto_url = f"mailto:{to_email}?subject={mail_subject}&body={mail_body}"

                st.success("mail template ready, click below to send")
                st.link_button("Open in Gmail (web)", gmail_url)
                st.link_button("Open in Mail App", mailto_url)
                st.caption("this just opens the mail already filled in, using your own logged in gmail / mail app. we dont touch your password.")

    with colB:
        st.write("Send via WhatsApp")
        wa_text = urllib.parse.quote(report_text)
        wa_url = "https://wa.me/?text=" + wa_text
        st.link_button("Share on WhatsApp", wa_url)
        st.caption("opens whatsapp, pick a contact and send")

# TODO: maybe add monthly comparison graph later
# TODO: pdf export instead of just csv
