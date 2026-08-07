# AI Personal Budget Planner
# Mini Project - Python + Streamlit + Gemini API (via LangChain)
# made this to track monthly budget and get AI tips
# v2: added OCR receipt scanning, bank statement import, and a nicer share/export tab

import os
import re
import io
import urllib.parse

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# OCR deps are optional at runtime - if tesseract binary isn't installed on the
# machine, we still want the rest of the app to work, so we fail soft.
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# PDF table extraction - optional, only needed for PDF bank statements
try:
    import pdfplumber
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# DOCX table extraction - optional, only needed for Word bank statements
try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

load_dotenv()

DATA_FILE = "data.csv"
cols = ["Category", "Remark", "Amount"]  # columns for the expense table

st.set_page_config(page_title="AI Personal Budget Planner", layout="wide", page_icon="💰")

# little bit of css just to make metric boxes look nice, took this from streamlit forum
# + a bit more css for the share buttons at the bottom
st.markdown("""
<style>
.stMetric {background-color:#F5F7FA; padding:10px; border-radius:10px;}

.share-btn {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:10px;
    text-decoration:none;
    color:white !important;
    font-weight:600;
    font-size:15px;
    padding:12px 18px;
    border-radius:10px;
    width:100%;
    box-sizing:border-box;
    margin-bottom:8px;
    transition:opacity 0.15s ease-in-out;
}
.share-btn:hover {opacity:0.88;}
.share-btn .logo-circle {
    background:white;
    border-radius:50%;
    width:26px;
    height:26px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:15px;
    flex-shrink:0;
}
.whatsapp-btn {background:#25D366;}
.whatsapp-btn .logo-circle {color:#25D366;}
.gmail-btn {background:#EA4335;}
.gmail-btn .logo-circle {color:#EA4335;}
.mailapp-btn {background:#4A5568;}
.mailapp-btn .logo-circle {color:#4A5568;}
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

if not OCR_AVAILABLE:
    st.sidebar.warning(
        "OCR libraries not found (pillow / pytesseract), or the tesseract "
        "binary isn't installed on this machine. Receipt scanning will be "
        "disabled until that's fixed. See requirements.txt for setup notes."
    )
else:
    # tesseract is a separate binary from the pytesseract python package - on
    # Windows especially, "pip install" succeeding does NOT mean the binary is
    # on PATH. This lets you point pytesseract straight at the .exe as a
    # fallback if "OCR failed: tesseract is not installed or it's not in your
    # PATH" keeps showing up even after installing it.
    with st.sidebar.expander("⚙️ OCR settings (only if receipt scan keeps failing)"):
        st.caption(
            "If you installed Tesseract but still get a 'not in PATH' error, "
            "paste the full path to tesseract.exe here. Typical Windows path: "
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        manual_tess_path = st.text_input("Tesseract binary path (optional)", value="", key="tess_path")
        if manual_tess_path.strip():
            pytesseract.pytesseract.tesseract_cmd = manual_tess_path.strip()


# small helper so we don't repeat the same ChatGoogleGenerativeAI setup everywhere
def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7
    )


# loads expense data from csv, if file not there just make empty table
def load_expenses():
    try:
        df = pd.read_csv(DATA_FILE)
        if list(df.columns) != cols:
            df = pd.DataFrame(columns=cols)
    except Exception:
        df = pd.DataFrame(columns=cols)
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def save_expenses(df):
    df.to_csv(DATA_FILE, index=False)


def add_expense_row(category, remark, amount):
    new_row = pd.DataFrame([{"Category": category, "Remark": remark, "Amount": amount}])
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_row], ignore_index=True)
    st.session_state.expenses["Amount"] = pd.to_numeric(st.session_state.expenses["Amount"], errors="coerce").fillna(0.0)
    save_expenses(st.session_state.expenses)


if "expenses" not in st.session_state:
    st.session_state.expenses = load_expenses()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "budgets" not in st.session_state:
    st.session_state.budgets = {}

if "income_val" not in st.session_state:
    st.session_state.income_val = 0.0

# ---------------- INCOME ----------------
st.subheader("1. Monthly Income")
income = st.number_input("Enter Monthly Income (₹)", min_value=0.0, step=500.0, key="income_val")

# using tabs so everything doesnt look like one giant scrolling page
tab_add, tab_import, tab_dash, tab_forecast, tab_advisor, tab_chat, tab_share = st.tabs(
    ["Add Expense", "Scan / Import", "Dashboard", "Forecast", "AI Advisor", "Chat", "Share/Export"]
)

# =========================================================
# TAB 1 - ADD EXPENSE
# =========================================================
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
        amount = st.number_input("Amount (₹)", min_value=0.0, step=50.0)
    with c4:
        st.write("")
        st.write("")
        add_btn = st.button("Add Expense")

    if add_btn:
        # agar user ne Other select kiya to custom name use karo
        final_cat = custom_cat.strip() if category == "Other" and custom_cat.strip() != "" else category

        if amount > 0:
            add_expense_row(final_cat, remark, amount)
            st.success("Added " + final_cat + " : ₹" + str(amount))
        else:
            st.warning("amount should be more than 0")

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
# TAB 2 - SCAN / IMPORT (OCR receipts + bank statement)
# =========================================================
with tab_import:

    # ---- keyword based auto categorizer, used for both OCR and bank rows ----
    CATEGORY_KEYWORDS = {
        "Food": ["restaurant", "food", "swiggy", "zomato", "cafe", "dine", "kitchen",
                  "hotel", "dominos", "pizza", "bakery"],
        "Rent": ["rent", "landlord", "housing society", "maintenance"],
        "Transport": ["uber", "ola", "fuel", "petrol", "diesel", "metro", "taxi",
                       "irctc", "flight", "indigo", "rapido", "train", "toll"],
        "Shopping": ["amazon", "flipkart", "mall", "store", "myntra", "shop", "mart"],
        "Entertainment": ["netflix", "spotify", "movie", "cinema", "bookmyshow",
                            "prime video", "hotstar", "pvr"],
        "Utilities": ["electricity", "water bill", "recharge", "broadband", "wifi",
                       "gas bill", "dth", "airtel", "jio", "vodafone"],
    }

    def categorize_from_text(text):
        text_l = (text or "").lower()
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(kw in text_l for kw in kws):
                return cat
        return "Other"

    st.subheader("📷 Scan a Bill / Receipt (OCR)")
    st.caption("upload a photo of a receipt and it'll try to read the merchant + total amount off it")

    if not OCR_AVAILABLE:
        st.error(
            "OCR isn't available in this environment. Install the Python packages "
            "`pytesseract` and `pillow`, AND install the Tesseract OCR engine itself "
            "(it's a separate system binary, not just a pip package):\n\n"
            "- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`\n"
            "- Mac: `brew install tesseract`\n"
            "- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki"
        )
    else:
        receipt_img = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"], key="receipt_uploader")

        if receipt_img is not None:
            img = Image.open(receipt_img)
            col_img, col_data = st.columns([1, 1.4])

            with col_img:
                st.image(img, caption="Uploaded receipt", use_container_width=True)

            with st.spinner("reading receipt..."):
                try:
                    ocr_text = pytesseract.image_to_string(img)
                except Exception as e:
                    ocr_text = ""
                    st.error("OCR failed: " + str(e))

            # try to guess the total amount - look for a "total" line first,
            # fall back to the largest rupee-looking number on the receipt
            def guess_amount(text):
                text_l = text.lower()
                patterns = [
                    r'(?:grand\s*total|total\s*amount|net\s*amount|amount\s*payable|total)[:\s₹rs\.]*([\d,]+\.\d{1,2})',
                    r'(?:grand\s*total|total\s*amount|net\s*amount|amount\s*payable|total)[:\s₹rs\.]*([\d,]+)',
                ]
                for pat in patterns:
                    m = re.search(pat, text_l)
                    if m:
                        try:
                            return float(m.group(1).replace(",", ""))
                        except ValueError:
                            pass
                nums = re.findall(r'[\d,]+\.\d{2}', text)
                nums = [float(n.replace(",", "")) for n in nums]
                if nums:
                    return max(nums)
                whole_nums = re.findall(r'\b\d{2,6}\b', text)
                whole_nums = [float(n) for n in whole_nums]
                return max(whole_nums) if whole_nums else 0.0

            def guess_merchant(text):
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                return lines[0] if lines else "Receipt"

            guessed_amount = guess_amount(ocr_text)
            guessed_merchant = guess_merchant(ocr_text)
            guessed_cat = categorize_from_text(ocr_text)

            with col_data:
                with st.expander("Raw OCR text (click to check if something looks off)"):
                    st.text(ocr_text if ocr_text.strip() else "(couldn't read any text from this image)")

                st.write("Review and fix the details below before adding:")
                r_cat = st.selectbox(
                    "Category",
                    ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"],
                    index=["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities", "Other"].index(guessed_cat) if guessed_cat in ["Food", "Rent", "Transport", "Shopping", "Entertainment", "Utilities"] else 6,
                    key="ocr_cat"
                )
                r_remark = st.text_input("Remark / Merchant", value=guessed_merchant, key="ocr_remark")
                r_amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, value=float(guessed_amount), key="ocr_amount")

                if st.button("➕ Add this receipt as an expense"):
                    if r_amount > 0:
                        add_expense_row(r_cat, r_remark, r_amount)
                        st.success(f"Added {r_cat} : ₹{r_amount} from receipt")
                    else:
                        st.warning("amount should be more than 0")

    st.write("---")
    st.subheader("🏦 Import from Bank Statement")
    st.caption("upload your statement as CSV, Excel (.xls/.xlsx), XML, or PDF - map the columns, and bulk-add transactions")

    # ---- robust reader: bank "excel" exports are frequently NOT what their
    # extension claims. Very common in India: a bank labels a file .xls but it's
    # actually an HTML table, or labels it .xlsx but it's actually old OLE .xls.
    # We try several strategies in order rather than trusting the extension.
    def read_bank_statement(uploaded_file):
        """Returns (dataframe_or_None, list_of_extra_tables_or_None, error_message_or_None).
        extra_tables is only populated for PDFs with multiple tables, so the
        caller can let the user pick which table is the real transaction list."""
        name = uploaded_file.name.lower()
        raw = uploaded_file.read()
        buf = io.BytesIO(raw)

        # ---------- CSV ----------
        if name.endswith(".csv"):
            for enc in ["utf-8", "utf-8-sig", "latin-1"]:
                try:
                    buf.seek(0)
                    return pd.read_csv(buf, encoding=enc), None, None
                except Exception:
                    continue
            return None, None, "Could not parse this CSV with common encodings."

        # ---------- XML ----------
        if name.endswith(".xml"):
            try:
                buf.seek(0)
                return pd.read_xml(buf), None, None
            except Exception as e:
                return None, None, f"Could not parse XML: {e}"

        # ---------- XLSX / XLS (try real formats first, then HTML-in-disguise) ----------
        if name.endswith((".xlsx", ".xls")):
            # try as real xlsx (zip-based)
            try:
                buf.seek(0)
                return pd.read_excel(buf, engine="openpyxl"), None, None
            except Exception:
                pass
            # try as real legacy xls (OLE-based) - needs xlrd
            try:
                buf.seek(0)
                return pd.read_excel(buf, engine="xlrd"), None, None
            except ImportError:
                pass
            except Exception:
                pass
            # try as HTML table wearing an Excel extension (very common bank export)
            try:
                buf.seek(0)
                tables = pd.read_html(buf)
                if tables:
                    # usually the largest table is the actual transaction list
                    tables.sort(key=len, reverse=True)
                    return tables[0], tables if len(tables) > 1 else None, None
            except Exception:
                pass
            return None, None, (
                "This file's contents don't match a real .xlsx, .xls, or HTML-based "
                "export. Try re-exporting from your bank as CSV instead - it's the "
                "most reliable format."
            )

        # ---------- PDF ----------
        if name.endswith(".pdf"):
            if not PDF_AVAILABLE:
                return None, None, "PDF support needs `pdfplumber` - run `pip install pdfplumber` and restart."
            try:
                buf.seek(0)
                all_tables = []
                with pdfplumber.open(buf) as pdf:
                    for page in pdf.pages:
                        for tbl in page.extract_tables():
                            if tbl and len(tbl) > 1:
                                df_t = pd.DataFrame(tbl[1:], columns=tbl[0])
                                all_tables.append(df_t)
                if not all_tables:
                    return None, None, (
                        "No tables detected in this PDF. If it's a scanned/image PDF "
                        "rather than a text-based statement, table extraction won't work - "
                        "export as CSV/Excel from your bank instead."
                    )
                all_tables.sort(key=len, reverse=True)
                return all_tables[0], all_tables if len(all_tables) > 1 else None, None
            except Exception as e:
                return None, None, f"Could not parse PDF: {e}"

        # ---------- DOCX ----------
        if name.endswith(".docx"):
            if not DOCX_AVAILABLE:
                return None, None, "DOCX support needs `python-docx` - run `pip install python-docx` and restart."
            try:
                buf.seek(0)
                d = docx.Document(buf)
                all_tables = []
                for t in d.tables:
                    rows = [[cell.text for cell in row.cells] for row in t.rows]
                    if len(rows) > 1:
                        all_tables.append(pd.DataFrame(rows[1:], columns=rows[0]))
                if not all_tables:
                    return None, None, "No tables found in this .docx file."
                all_tables.sort(key=len, reverse=True)
                return all_tables[0], all_tables if len(all_tables) > 1 else None, None
            except Exception as e:
                return None, None, f"Could not parse DOCX: {e}"

        # ---------- old-style .doc (binary, pre-2007 Word) ----------
        if name.endswith(".doc"):
            return None, None, (
                "Old-format .doc files aren't supported directly - they use a binary "
                "format with no reliable pure-Python reader. Please save/export the "
                "statement as .docx, .pdf, or .csv from Word and re-upload."
            )

        return None, None, "Unsupported file type."

    bank_file = st.file_uploader(
        "Upload bank statement (CSV, Excel, XML, PDF, or DOCX)",
        type=["csv", "xlsx", "xls", "xml", "pdf", "docx"],
        key="bank_uploader"
    )

    if bank_file is not None:
        bank_df, extra_tables, err = read_bank_statement(bank_file)
        if err:
            st.error(err)

        if extra_tables:
            st.info(f"Found {len(extra_tables)} tables in this file - showing the largest one below. "
                     "If it's the wrong one, re-export a cleaner file with just the transaction table.")

        if bank_df is not None and not bank_df.empty:
            st.write("Preview:")
            st.dataframe(bank_df.head(), use_container_width=True)

            all_cols = list(bank_df.columns)
            st.write("Map your columns:")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                desc_col = st.selectbox("Description column", all_cols, key="bank_desc_col")
            with mc2:
                amount_mode = st.radio("Amount format", ["Single Amount column (+/-)", "Separate Debit/Credit columns"], key="bank_amt_mode")
            with mc3:
                if amount_mode == "Single Amount column (+/-)":
                    amt_col = st.selectbox("Amount column", all_cols, key="bank_amt_col")
                    debit_col = credit_col = None
                else:
                    debit_col = st.selectbox("Debit column", all_cols, key="bank_debit_col")
                    amt_col = None
            with mc4:
                if amount_mode != "Single Amount column (+/-)":
                    credit_col = st.selectbox("Credit column", all_cols, key="bank_credit_col")

            # build a normalized transactions table: Description, Amount, Type (Debit/Credit)
            work = bank_df.copy()
            if amount_mode == "Single Amount column (+/-)":
                work["_amount_raw"] = pd.to_numeric(work[amt_col], errors="coerce").fillna(0.0)
                work["_type"] = work["_amount_raw"].apply(lambda x: "Credit" if x > 0 else "Debit")
                work["_amount"] = work["_amount_raw"].abs()
            else:
                debit_vals = pd.to_numeric(work[debit_col], errors="coerce").fillna(0.0)
                credit_vals = pd.to_numeric(work[credit_col], errors="coerce").fillna(0.0)
                work["_amount"] = debit_vals.where(debit_vals > 0, credit_vals)
                work["_type"] = debit_vals.apply(lambda x: "Debit" if x > 0 else "Credit")

            work["_desc"] = work[desc_col].astype(str)
            work["_category"] = work["_desc"].apply(categorize_from_text)
            work.loc[work["_type"] == "Credit", "_category"] = "Income"

            preview = work[["_desc", "_type", "_category", "_amount"]].rename(
                columns={"_desc": "Description", "_type": "Type", "_category": "Category", "_amount": "Amount"}
            )
            st.write("Transactions found (edit categories if needed, then import):")
            preview_edited = st.data_editor(preview, use_container_width=True, key="bank_preview_editor")

            total_debit = preview_edited.loc[preview_edited["Type"] == "Debit", "Amount"].sum()
            total_credit = preview_edited.loc[preview_edited["Type"] == "Credit", "Amount"].sum()

            st.write(f"Total spending found: **₹{total_debit:,.2f}**  |  Total income found: **₹{total_credit:,.2f}**")

            bi1, bi2 = st.columns(2)
            with bi1:
                if st.button("➕ Add all Debit rows as Expenses"):
                    debit_rows = preview_edited[preview_edited["Type"] == "Debit"]
                    for _, row in debit_rows.iterrows():
                        if row["Amount"] > 0:
                            add_expense_row(row["Category"], row["Description"], row["Amount"])
                    st.success(f"Added {len(debit_rows)} expense(s) from bank statement")
            with bi2:
                if st.button(f"💰 Set Monthly Income to detected credit total (₹{total_credit:,.2f})"):
                    st.session_state.income_val = float(total_credit)
                    st.rerun()

# =========================================================
# TAB 3 - DASHBOARD
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
# TAB 4 - FORECAST
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
# TAB 5 - AI ADVISOR
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
# TAB 6 - CHAT
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
# TAB 7 - SHARE / EXPORT
# =========================================================
with tab_share:
    total_exp = st.session_state.expenses["Amount"].sum() if not st.session_state.expenses.empty else 0.0
    balance = income - total_exp
    cat_totals = st.session_state.expenses.groupby("Category")["Amount"].sum() if not st.session_state.expenses.empty else pd.Series(dtype=float)

    st.subheader("Export Data")
    if not st.session_state.expenses.empty:
        csv_data = st.session_state.expenses.to_csv(index=False)
        st.download_button("Download CSV", data=csv_data, file_name="my_expenses.csv", mime="text/csv")
    else:
        st.info("add expenses first to download")

    st.write("---")
    st.subheader("Share Report")

    # ---------- nicely formatted markdown report, shown as a live preview ----------
    def build_markdown_report(income, total_exp, balance, cat_totals):
        lines = []
        lines.append("# 📊 AI Personal Budget Planner")
        lines.append("## Monthly Budget Report")
        lines.append("")
        lines.append("**Report Summary**")
        lines.append("")
        lines.append("| Metric | Amount |")
        lines.append("|---|---:|")
        lines.append(f"| 💰 Monthly Income | **Rs. {income:,.2f}** |")
        lines.append(f"| 💸 Total Expenses | **Rs. {total_exp:,.2f}** |")
        bal_icon = "📉" if balance < 0 else "📈"
        lines.append(f"| {bal_icon} Remaining Balance | **Rs. {balance:,.2f}** |")
        lines.append("---")
        lines.append("## Expense Breakdown")
        lines.append("")
        if not cat_totals.empty:
            lines.append("| Category | Amount (Rs.) |")
            lines.append("|---|---:|")
            for cat, amt in cat_totals.items():
                lines.append(f"| {cat} | {amt:,.2f} |")
        else:
            lines.append("_No expenses recorded yet._")
        lines.append("---")
        lines.append("## Financial Insights")
        lines.append("")
        if balance < 0:
            lines.append(f"⚠️ Your expenses exceeded your income this month, resulting in a **negative balance of Rs. {abs(balance):,.2f}**.")
        elif income > 0:
            lines.append(f"✅ You're within budget this month with a **positive balance of Rs. {balance:,.2f}**.")
        else:
            lines.append("ℹ️ No income recorded yet, so balance can't be fully evaluated.")
        lines.append("")
        lines.append("### Recommendations")
        recs = []
        if not cat_totals.empty:
            top_cat = cat_totals.idxmax()
            top_amt = cat_totals.max()
            recs.append(f"Review high-value expenses such as **{top_cat} (Rs. {top_amt:,.2f})**.")
        recs.append("Set a monthly budget limit for discretionary spending.")
        recs.append("Track expenses regularly to avoid overspending.")
        recs.append("Consider building an emergency savings fund.")
        if income == 0:
            recs.append("Record your monthly income to receive more accurate financial insights.")
        for r in recs:
            lines.append(f"* {r}")
        lines.append("---")
        lines.append("*This report was automatically generated by the **AI Personal Budget Planner**.*")
        return "\n".join(lines)

    # ---------- plain-text version for WhatsApp / email body ----------
    # (WhatsApp & most mail clients don't render markdown tables, so this uses
    # simple bullet lines + WhatsApp-style *bold*/_italic_ markers instead)
    def build_plain_report(income, total_exp, balance, cat_totals):
        lines = []
        lines.append("📊 *AI PERSONAL BUDGET PLANNER*")
        lines.append("_Monthly Budget Report_")
        lines.append("")
        lines.append("*Summary*")
        lines.append(f"💰 Income: Rs. {income:,.2f}")
        lines.append(f"💸 Expenses: Rs. {total_exp:,.2f}")
        bal_icon = "📉" if balance < 0 else "📈"
        lines.append(f"{bal_icon} Balance: Rs. {balance:,.2f}")
        lines.append("")
        lines.append("*Expense Breakdown*")
        if not cat_totals.empty:
            for cat, amt in cat_totals.items():
                lines.append(f"• {cat}: Rs. {amt:,.2f}")
        else:
            lines.append("No expenses recorded yet.")
        lines.append("")
        lines.append("*Insights*")
        if balance < 0:
            lines.append(f"⚠️ Overspent by Rs. {abs(balance):,.2f} this month.")
        elif income > 0:
            lines.append(f"✅ Saved Rs. {balance:,.2f} this month.")
        else:
            lines.append("ℹ️ Add your income for a full picture.")
        if not cat_totals.empty:
            lines.append(f"👉 Biggest expense: {cat_totals.idxmax()} (Rs. {cat_totals.max():,.2f})")
        lines.append("")
        lines.append("_Generated by AI Personal Budget Planner_")
        return "\n".join(lines)

    markdown_report = build_markdown_report(income, total_exp, balance, cat_totals)
    plain_report = build_plain_report(income, total_exp, balance, cat_totals)

    with st.expander("📄 Preview report", expanded=True):
        st.markdown(markdown_report)

    st.write("")
    colA, colB = st.columns(2)

    with colA:
        st.write("**Send via Email**")
        to_email = st.text_input("Recipient Email")

        subject = "Your Monthly Budget Report"
        mail_subject = urllib.parse.quote(subject)
        mail_body = urllib.parse.quote(plain_report)

        if st.button("Prepare Mail"):
            if not to_email:
                st.warning("enter recipient email first")
            else:
                to_encoded = urllib.parse.quote(to_email)
                gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={to_encoded}&su={mail_subject}&body={mail_body}"
                mailto_url = f"mailto:{to_email}?subject={mail_subject}&body={mail_body}"

                st.success("mail template ready, click below to send")
                st.markdown(f"""
                <a class="share-btn gmail-btn" href="{gmail_url}" target="_blank">
                    <span class="logo-circle">✉️</span> Open in Gmail (web)
                </a>
                <a class="share-btn mailapp-btn" href="{mailto_url}">
                    <span class="logo-circle">📧</span> Open in Mail App
                </a>
                """, unsafe_allow_html=True)
                st.caption("this just opens the mail already filled in, using your own logged in gmail / mail app. we dont touch your password.")

    with colB:
        st.write("**Send via WhatsApp**")
        wa_text = urllib.parse.quote(plain_report)
        wa_url = "https://wa.me/?text=" + wa_text
        st.markdown(f"""
        <a class="share-btn whatsapp-btn" href="{wa_url}" target="_blank">
            <span class="logo-circle">💬</span> Share on WhatsApp
        </a>
        """, unsafe_allow_html=True)
        st.caption("opens whatsapp, pick a contact and send")

# TODO: maybe add monthly comparison graph later
# TODO: pdf export instead of just csv
# TODO: PDF bank statement parsing (currently CSV/Excel only)
