# AI Personal Budget Planner
# Mini Project - Python + Streamlit + Gemini API (via LangChain)
# made this to track monthly budget and get AI tips

import os
import json
import base64
import urllib.parse

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

load_dotenv()

DATA_FILE = "data.csv"
cols = ["Category", "Remark", "Amount"]  # columns for the expense table
CATEGORY_OPTIONS = ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"]

# one consistent colour per category, used across the pie/bar charts and the budget bars
CATEGORY_COLORS = {
    "Food": "#F97316",
    "Rent": "#6366F1",
    "Transport": "#0EA5E9",
    "Shopping": "#EC4899",
    "Entertainment": "#8B5CF6",
    "Utilities": "#14B8A6",
    "Other": "#64748B",
}

st.set_page_config(page_title="AI Personal Budget Planner", layout="wide", page_icon="💸")

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

.main .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }

/* ---- hero header ---- */
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(79,70,229,0.25);
}
.hero h1 { font-family: 'Poppins', sans-serif; font-size: 1.7rem; margin: 0 0 4px 0; font-weight: 700; }
.hero p { margin: 0; opacity: 0.9; font-size: 0.92rem; }

/* ---- metric cards ---- */
.metric-row { display:flex; gap:14px; margin-bottom: 1.3rem; flex-wrap: wrap; }
.metric-card {
    flex:1; min-width: 190px;
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 10px rgba(15,23,42,0.06);
    border-left: 5px solid var(--accent, #4F46E5);
}
.metric-card .m-label { font-size: 0.75rem; color:#64748B; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.metric-card .m-value { font-size: 1.45rem; font-weight:800; color:#0F172A; margin-top:2px; font-family:'Poppins',sans-serif; }
.metric-card .m-sub { font-size:0.76rem; color:#94A3B8; margin-top:3px; }

/* ---- generic section card ---- */
.section-card {
    background:#fff; border-radius:18px; padding:1.3rem 1.4rem;
    box-shadow:0 2px 10px rgba(15,23,42,0.05); margin-bottom:1.1rem;
}
.section-title {
    font-family:'Poppins', sans-serif; font-weight:700; font-size:1.02rem;
    color:#1E293B; margin-bottom: .7rem; display:flex; align-items:center; gap:8px;
}
.section-caption { color:#94A3B8; font-size:0.82rem; margin-top:-8px; margin-bottom:.8rem; }

/* ---- buttons ---- */
.stButton>button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all .15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 14px rgba(79,70,229,0.20); }

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 10px 16px; font-weight:600; }

/* ---- custom budget progress bars ---- */
.budget-bar-wrap { margin-bottom: 14px; }
.budget-bar-label { display:flex; justify-content:space-between; font-size:0.83rem; font-weight:600; color:#334155; margin-bottom:4px; }
.budget-bar-track { background:#EEF2F7; border-radius: 8px; height:10px; overflow:hidden; }
.budget-bar-fill { height:100%; border-radius:8px; }

/* ---- badges ---- */
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; }

/* ---- empty state ---- */
.empty-state { text-align:center; padding: 2.4rem 1rem; color:#94A3B8; }
.empty-state .icon { font-size:2.3rem; margin-bottom: 6px; }
.empty-state .title { color:#475569; font-weight:600; margin-bottom:2px; }

/* ---- sidebar quick stats ---- */
.side-stat { display:flex; justify-content:space-between; font-size:0.85rem; padding:6px 0; border-bottom:1px solid #EEF2F7; }
.side-stat b { color:#0F172A; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SMALL RENDER HELPERS
# =========================================================
def money(v):
    return f"₹{v:,.2f}"


def metric_card_html(label, value, sub, accent):
    return f"""
    <div class="metric-card" style="--accent:{accent}">
        <div class="m-label">{label}</div>
        <div class="m-value">{value}</div>
        <div class="m-sub">{sub}</div>
    </div>
    """


def render_summary_bar(income, total_exp, balance, savings):
    rate = (savings / income * 100) if income > 0 else 0
    cards = [
        metric_card_html("Income", money(income), "this month", "#4F46E5"),
        metric_card_html("Expense", money(total_exp), f"{len(st.session_state.expenses)} entries", "#F97316"),
        metric_card_html("Balance", money(balance), "income − expense", "#0EA5E9" if balance >= 0 else "#EF4444"),
        metric_card_html("Savings", money(savings), f"{rate:.1f}% of income" if income > 0 else "add income", "#22C55E" if rate >= 20 else "#F59E0B"),
    ]
    st.markdown(f'<div class="metric-row">{"".join(cards)}</div>', unsafe_allow_html=True)


def empty_state(icon, title, sub):
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <div class="title">{title}</div>
        <div>{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def budget_bar(cat, spent, lim):
    pct = min(spent / lim, 1.0) * 100 if lim > 0 else 0
    color = "#22C55E" if pct < 80 else ("#F59E0B" if pct <= 100 else "#EF4444")
    over = spent > lim
    st.markdown(f"""
    <div class="budget-bar-wrap">
        <div class="budget-bar-label">
            <span>{cat}</span>
            <span>{money(spent)} / {money(lim)} {"⚠️" if over else ""}</span>
        </div>
        <div class="budget-bar-track">
            <div class="budget-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if over:
        st.markdown(f'<span class="badge" style="background:#FEE2E2;color:#B91C1C;">{cat} is over budget</span>', unsafe_allow_html=True)


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

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    items = json.loads(raw_text)

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


# =========================================================
# SESSION STATE
# =========================================================
if "expenses" not in st.session_state:
    st.session_state.expenses = load_expenses()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

if "scanned_items" not in st.session_state:
    st.session_state.scanned_items = pd.DataFrame(columns=cols)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 💸 Budget Planner")
    st.caption("Track spends, scan bills, get AI tips")
    st.divider()

    st.markdown("**API Key**")
    # checks Streamlit Cloud secrets first (st.secrets), then falls back to .env / local env var
    try:
        env_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        env_key = os.getenv("GEMINI_API_KEY")
    GEMINI_API_KEY = st.text_input("GEMINI_API_KEY", type="password", value=env_key if env_key else "", label_visibility="collapsed")

    if GEMINI_API_KEY:
        st.success("API key loaded ✅")
    else:
        st.info("Paste your Gemini API key to unlock AI features")
        st.caption("Get one free at [aistudio.google.com](https://aistudio.google.com/app/apikey)")

    st.divider()
    st.markdown("**Quick Stats**")
    _total_exp_sb = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    st.markdown(f"""
    <div class="side-stat"><span>Total Expense</span><b>{money(_total_exp_sb)}</b></div>
    <div class="side-stat"><span>Entries</span><b>{len(st.session_state.expenses)}</b></div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Made with Streamlit + Gemini ✨")

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>AI Personal Budget Planner 💰</h1>
    <p>Track your income and expenses, scan bills automatically, and get AI-powered budgeting advice.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- INCOME ----------------
with st.container(border=True):
    st.markdown('<div class="section-title">💵 Monthly Income</div>', unsafe_allow_html=True)
    income = st.number_input("Enter Monthly Income (₹)", min_value=0.0, step=500.0, label_visibility="collapsed")

# persistent summary, visible no matter which tab is open
_total_exp_top = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
render_summary_bar(income, _total_exp_top, income - _total_exp_top, income - _total_exp_top)

# using tabs so everything doesnt look like one giant scrolling page
tab_add, tab_dash, tab_forecast, tab_advisor, tab_chat, tab_share = st.tabs(
    ["➕ Add Expense", "📊 Dashboard", "📈 Forecast", "🤖 AI Advisor", "💬 Chat", "📤 Share/Export"]
)

# =========================================================
# TAB 1 - ADD EXPENSE
# =========================================================
with tab_add:
    with st.container(border=True):
        st.markdown('<div class="section-title">✍️ Add Manually</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns([1.2, 1.4, 1, 0.8])

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
            add_btn = st.button("Add ➕", use_container_width=True)

        if add_btn:
            final_cat = custom_cat.strip() if category == "Other" and custom_cat.strip() != "" else category

            if amount > 0:
                new_row = pd.DataFrame([{"Category": final_cat, "Remark": remark, "Amount": amount}])
                st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
                st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors="coerce").fillna(0.0)
                save_expenses(st.session_state.expenses)
                st.toast(f"Added {final_cat} — {money(amount)}", icon="✅")
            else:
                st.warning("amount should be more than 0")

    with st.container(border=True):
        st.markdown('<div class="section-title">📷 Scan a Bill / Receipt</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Upload or capture a bill photo — AI reads it and fills in the expenses for you.</div>', unsafe_allow_html=True)

        if not GEMINI_API_KEY:
            st.info("Add your Gemini API key in the sidebar to use bill scanning")
        else:
            scan_c1, scan_c2 = st.columns(2)
            with scan_c1:
                uploaded_bill = st.file_uploader("Upload bill image", type=["jpg", "jpeg", "png", "webp"], key="bill_upload")
            with scan_c2:
                camera_bill = st.camera_input("Or take a photo")

            bill_file = camera_bill if camera_bill is not None else uploaded_bill

            if bill_file is not None:
                pc1, pc2 = st.columns([1, 2])
                with pc1:
                    st.image(bill_file, caption="Preview", use_container_width=True)
                with pc2:
                    st.write("")
                    if st.button("🔍 Scan Bill with AI", use_container_width=True):
                        with st.spinner("Reading your bill..."):
                            try:
                                img_bytes = bill_file.getvalue()
                                mime = bill_file.type if getattr(bill_file, "type", None) else "image/jpeg"
                                extracted = extract_expenses_from_bill(img_bytes, mime)

                                if not extracted:
                                    st.warning("Couldn't read any expense lines from this image — try a clearer photo")
                                else:
                                    st.session_state.scanned_items = pd.DataFrame(extracted)
                                    st.success(f"Found {len(extracted)} item(s) — review below before adding")
                            except json.JSONDecodeError:
                                st.error("AI didn't return clean data, try scanning again or use a clearer photo")
                            except Exception as e:
                                st.error("Error: " + str(e))

            if not st.session_state.scanned_items.empty:
                st.write("**Review scanned items** (edit if needed):")
                reviewed = st.data_editor(
                    st.session_state.scanned_items,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="scanned_editor",
                )

                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("✅ Add Scanned Items to Expenses", use_container_width=True):
                        reviewed["Amount"] = pd.to_numeric(reviewed["Amount"], errors="coerce").fillna(0.0)
                        reviewed = reviewed[reviewed["Amount"] > 0]
                        st.session_state.expenses = pd.concat([st.session_state.expenses, reviewed], ignore_index=True)
                        save_expenses(st.session_state.expenses)
                        st.session_state.scanned_items = pd.DataFrame(columns=cols)
                        st.toast("Scanned items added!", icon="🎉")
                        st.rerun()
                with rc2:
                    if st.button("❌ Discard Scanned Items", use_container_width=True):
                        st.session_state.scanned_items = pd.DataFrame(columns=cols)
                        st.rerun()

    with st.container(border=True):
        st.markdown('<div class="section-title">📋 Expense Table</div>', unsafe_allow_html=True)

        if st.session_state.expenses.empty:
            empty_state("🧾", "No expenses yet", "Add one manually or scan a bill above to get started")
        else:
            edited = st.data_editor(st.session_state.expenses, use_container_width=True, num_rows="dynamic")

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("💾 Save Changes", use_container_width=True):
                    edited["Amount"] = pd.to_numeric(edited["Amount"], errors="coerce").fillna(0.0)
                    st.session_state.expenses = edited
                    save_expenses(st.session_state.expenses)
                    st.toast("Saved!", icon="💾")
            with col_clear:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.expenses = pd.DataFrame(columns=cols)
                    save_expenses(st.session_state.expenses)
                    st.rerun()

    with st.container(border=True):
        st.markdown('<div class="section-title">🎯 Category Budget Limits</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Optional — resets when you close the app (not saved to file yet)</div>', unsafe_allow_html=True)

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
    savings = balance

    if not st.session_state.expenses.empty:
        cat_totals = st.session_state.expenses.groupby("Category")["Amount"].sum().reset_index()
        colors = [CATEGORY_COLORS.get(c, "#94A3B8") for c in cat_totals["Category"]]

        colp, colb = st.columns(2)
        with colp:
            with st.container(border=True):
                st.markdown('<div class="section-title">🥧 Expense Breakdown</div>', unsafe_allow_html=True)
                fig1 = px.pie(cat_totals, names="Category", values="Amount", hole=0.55,
                              color="Category", color_discrete_map=CATEGORY_COLORS)
                fig1.update_traces(textinfo="percent+label", textfont_size=12)
                fig1.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320)
                st.plotly_chart(fig1, use_container_width=True)

        with colb:
            with st.container(border=True):
                st.markdown('<div class="section-title">📊 Spend by Category</div>', unsafe_allow_html=True)
                fig2 = px.bar(cat_totals.sort_values("Amount"), x="Amount", y="Category", orientation="h",
                              color="Category", color_discrete_map=CATEGORY_COLORS, text="Amount")
                fig2.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
                fig2.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320,
                                    xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig2, use_container_width=True)

        if st.session_state.budgets and any(v > 0 for v in st.session_state.budgets.values()):
            with st.container(border=True):
                st.markdown('<div class="section-title">🎯 Budget Limit Check</div>', unsafe_allow_html=True)
                for cat in st.session_state.budgets:
                    lim = st.session_state.budgets[cat]
                    if lim > 0:
                        spent = st.session_state.expenses.loc[st.session_state.expenses["Category"] == cat, "Amount"].sum()
                        budget_bar(cat, spent, lim)
    else:
        with st.container(border=True):
            empty_state("📊", "Nothing to chart yet", "Add some expenses in the Add Expense tab first")

    with st.container(border=True):
        st.markdown('<div class="section-title">💡 Saving Tips</div>', unsafe_allow_html=True)
        if income > 0:
            rate = (savings / income) * 100
            if rate < 20:
                st.warning(f"Savings is only {rate:.1f}% of income — try to save at least 20%")
            else:
                st.success(f"Nice! You're saving {rate:.1f}% of income")

            if not st.session_state.expenses.empty:
                food_amt = st.session_state.expenses.loc[st.session_state.expenses["Category"] == "Food", "Amount"].sum()
                if food_amt > 0.15 * income:
                    st.warning("Food expense is quite high — try cooking at home more often")

            if total_exp > income:
                st.error("You are spending more than you earn this month!")

            st.markdown("""
            - Try to save money as soon as salary comes in
            - Track small expenses too — they add up
            - Cancel subscriptions you don't use
            - Keep some emergency fund
            """)
        else:
            st.info("Enter income above to see personalized tips")

# =========================================================
# TAB 3 - FORECAST
# =========================================================
with tab_forecast:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0

    with st.container(border=True):
        st.markdown('<div class="section-title">📈 Forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Simple projection based on your current numbers</div>', unsafe_allow_html=True)

        if not st.session_state.expenses.empty and income > 0:
            horizon = st.radio("Forecast for", ["Monthly", "Yearly"], horizontal=True, label_visibility="collapsed")
            proj_saving = income - total_exp

            if horizon == "Monthly":
                x = list(range(1, 7))
                y = [proj_saving * i for i in x]
                xlabel = "Months ahead"
            else:
                x = list(range(1, 6))
                y = [proj_saving * 12 * i for i in x]
                xlabel = "Years ahead"

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers", line=dict(color="#22C55E", width=3),
                marker=dict(size=8, color="#22C55E"), fill="tozeroy",
                fillcolor="rgba(34,197,94,0.12)"
            ))
            fig4.update_layout(
                xaxis_title=xlabel, yaxis_title="Projected Savings (₹)",
                margin=dict(t=20, b=10, l=10, r=10), height=340,
                plot_bgcolor="white"
            )
            st.plotly_chart(fig4, use_container_width=True)

            f1, f2, f3 = st.columns(3)
            f1.markdown(metric_card_html("Current Monthly Expense", money(total_exp), "", "#F97316"), unsafe_allow_html=True)
            f2.markdown(metric_card_html("Projected Yearly Expense", money(total_exp * 12), "", "#EF4444"), unsafe_allow_html=True)
            f3.markdown(metric_card_html("Projected Yearly Savings", money(proj_saving * 12), "", "#22C55E"), unsafe_allow_html=True)
        else:
            empty_state("📈", "Nothing to forecast yet", "Enter income and add expenses first")

# =========================================================
# TAB 4 - AI ADVISOR
# =========================================================
with tab_advisor:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    def get_context():
        exp_summary = st.session_state.expenses.groupby("Category")["Amount"].sum().to_string() if not st.session_state.expenses.empty else "No expenses recorded"
        return f"""
        Monthly Income: {income}
        Total Expense: {total_exp}
        Balance: {balance}
        Expense Breakdown:
        {exp_summary}
        """

    with st.container(border=True):
        st.markdown('<div class="section-title">🤖 AI Financial Advisor</div>', unsafe_allow_html=True)

        if st.button("✨ Get AI Financial Advice", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("Give API key in sidebar first")
            elif income == 0:
                st.warning("Enter monthly income first")
            else:
                with st.spinner("Asking Gemini..."):
                    try:
                        advisor_prompt = ChatPromptTemplate.from_template(
                            "You are a friendly personal finance advisor. {context} "
                            "Give me 3-5 short practical tips to improve my budget and savings."
                        )
                        chain = advisor_prompt | get_gemini_model() | StrOutputParser()
                        result = chain.invoke({"context": get_context()})
                        with st.chat_message("assistant"):
                            st.markdown(result)
                    except Exception as e:
                        st.error("Error: " + str(e))

    st.warning("⚠️ Disclaimer: Yeh AI advice sirf general guidance ke liye hai. Kripya koi bhi bada financial decision lene se pehle ek certified financial adviser se consult karo.")

# =========================================================
# TAB 5 - CHAT
# =========================================================
with tab_chat:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp

    with st.container(border=True):
        st.markdown('<div class="section-title">💬 Chat with your Planner</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Ask stuff like "where can I cut cost" or "can I afford a ₹5000 trip"</div>', unsafe_allow_html=True)

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        q = st.chat_input("Ask something...")

        if q:
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.write(q)

            if not GEMINI_API_KEY:
                ans = "Give API key in sidebar to use chat"
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

    with st.container(border=True):
        st.markdown('<div class="section-title">⬇️ Export Data</div>', unsafe_allow_html=True)
        if not st.session_state.expenses.empty:
            csv_data = st.session_state.expenses.to_csv(index=False)
            st.download_button("Download CSV", data=csv_data, file_name="my_expenses.csv", mime="text/csv", use_container_width=True)
        else:
            empty_state("⬇️", "Nothing to export yet", "Add expenses first to download")

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
        with st.container(border=True):
            st.markdown('<div class="section-title">📧 Send via Email</div>', unsafe_allow_html=True)
            to_email = st.text_input("Recipient Email")

            subject = "Your Monthly Budget Report"
            mail_subject = urllib.parse.quote(subject)
            mail_body = urllib.parse.quote(report_text)

            if st.button("Prepare Mail", use_container_width=True):
                if not to_email:
                    st.warning("Enter recipient email first")
                else:
                    to_encoded = urllib.parse.quote(to_email)
                    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={to_encoded}&su={mail_subject}&body={mail_body}"
                    mailto_url = f"mailto:{to_email}?subject={mail_subject}&body={mail_body}"

                    st.success("Mail template ready, click below to send")
                    st.link_button("Open in Gmail (web)", gmail_url, use_container_width=True)
                    st.link_button("Open in Mail App", mailto_url, use_container_width=True)
                    st.caption("This just opens the mail already filled in, using your own logged-in Gmail / mail app. We don't touch your password.")

    with colB:
        with st.container(border=True):
            st.markdown('<div class="section-title">🟢 Send via WhatsApp</div>', unsafe_allow_html=True)
            wa_text = urllib.parse.quote(report_text)
            wa_url = "https://wa.me/?text=" + wa_text
            st.link_button("Share on WhatsApp", wa_url, use_container_width=True)
            st.caption("Opens WhatsApp, pick a contact and send")

# TODO: maybe add monthly comparison graph later
# TODO: pdf export instead of just csv
