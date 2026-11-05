# === Importing Libraries ==============================

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# === Page Configuration ==============================

st.set_page_config(
    page_title="SpamLens",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# === Loading Model & Computing Metrics ==============================

@st.cache_resource
def load_model():
    return joblib.load('model/SpamLens.pkl')


@st.cache_data
def compute_metrics():
    mdl = load_model()
    data = pd.read_csv('data/email.csv')
    data.drop_duplicates(keep='first', inplace=True)
    X = data.drop(columns=['Category'])
    y = data['Category']
    y_pred = mdl.predict(X)
    return {
        'accuracy':      round(accuracy_score(y, y_pred) * 100, 2),
        'precision':     round(precision_score(y, y_pred, pos_label='spam') * 100, 2),
        'recall':        round(recall_score(y, y_pred, pos_label='spam') * 100, 2),
        'f1':            round(f1_score(y, y_pred, pos_label='spam') * 100, 2),
        'total_samples': int(len(y)),
    }


model   = load_model()
metrics = compute_metrics()


# === Session State ==============================

if "page" not in st.session_state:
    st.session_state["page"] = "home"

if "theme" not in st.session_state:
    st.session_state["theme"] = "system"


# === Theme Configuration ==============================

theme_override_css = ""
if st.session_state.get("theme") == "dark":
    theme_override_css = """
    :root {
        --bg-main: #0b0f19 !important;
        --bg-sidebar: #0f1422 !important;
        --bg-card: #141a29 !important;
        --bg-card-hover: #1c2438 !important;
        --border-color: rgba(255, 255, 255, 0.09) !important;
        --border-hover: rgba(255, 255, 255, 0.18) !important;
        --text-primary: #f8fafc !important;
        --text-secondary: #94a3b8 !important;
        --text-muted: #64748b !important;
        --btn-bg: #f8fafc !important;
        --btn-color: #0b0f19 !important;
        --btn-hover: #e2e8f0 !important;
        --textarea-bg: #111726 !important;
        --textarea-border: rgba(255, 255, 255, 0.12) !important;
        --textarea-focus: rgba(255, 255, 255, 0.28) !important;
        --sb-btn-color: #94a3b8 !important;
        --sb-btn-hover: rgba(255, 255, 255, 0.06) !important;
        --divider-color: rgba(255, 255, 255, 0.08) !important;
        --footer-color: #64748b !important;
        --accent-line: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), rgba(255,255,255,0.08), transparent) !important;
        --card-blue-bg: rgba(59, 130, 246, 0.16) !important;
        --card-blue-icon: #60a5fa !important;
        --card-blue-val: #93c5fd !important;
        --card-green-bg: rgba(34, 197, 94, 0.16) !important;
        --card-green-icon: #4ade80 !important;
        --card-green-val: #86efac !important;
        --card-amber-bg: rgba(245, 158, 11, 0.16) !important;
        --card-amber-icon: #fbbf24 !important;
        --card-amber-val: #fde047 !important;
        --card-purple-bg: rgba(168, 85, 247, 0.16) !important;
        --card-purple-icon: #c084fc !important;
        --card-purple-val: #d8b4fe !important;
        --card-slate-bg: rgba(255, 255, 255, 0.08) !important;
        --card-slate-icon: #cbd5e1 !important;
        --card-slate-val: #f1f5f9 !important;
        --ham-bg: rgba(34, 197, 94, 0.12) !important;
        --ham-border: rgba(34, 197, 94, 0.3) !important;
        --ham-text: #4ade80 !important;
        --spam-bg: rgba(239, 68, 68, 0.12) !important;
        --spam-border: rgba(239, 68, 68, 0.3) !important;
        --spam-text: #f87171 !important;
        --step-num-bg: #f8fafc !important;
        --step-num-color: #0b0f19 !important;
        --chevron-color: #64748b !important;
        --shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
        --shadow-md: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
    }
    """
elif st.session_state.get("theme") == "light":
    theme_override_css = """
    :root {
        --bg-main: #ffffff !important;
        --bg-sidebar: #ffffff !important;
        --bg-card: #ffffff !important;
        --bg-card-hover: #f8fafc !important;
        --border-color: #e2e8f0 !important;
        --border-hover: #cbd5e1 !important;
        --text-primary: #0f172a !important;
        --text-secondary: #475569 !important;
        --text-muted: #94a3b8 !important;
        --btn-bg: #0f172a !important;
        --btn-color: #ffffff !important;
        --btn-hover: #1e293b !important;
        --textarea-bg: #ffffff !important;
        --textarea-border: #e2e8f0 !important;
        --textarea-focus: #94a3b8 !important;
        --sb-btn-color: #64748b !important;
        --sb-btn-hover: #f1f5f9 !important;
        --divider-color: #e2e8f0 !important;
        --footer-color: #94a3b8 !important;
        --accent-line: linear-gradient(90deg, transparent, #cbd5e1, #94a3b8, transparent) !important;
        --card-blue-bg: rgba(37, 99, 235, 0.08) !important;
        --card-blue-icon: #2563eb !important;
        --card-blue-val: #1e3a8a !important;
        --card-green-bg: rgba(22, 163, 74, 0.08) !important;
        --card-green-icon: #16a34a !important;
        --card-green-val: #14532d !important;
        --card-amber-bg: rgba(217, 119, 6, 0.08) !important;
        --card-amber-icon: #d97706 !important;
        --card-amber-val: #78350f !important;
        --card-purple-bg: rgba(147, 51, 234, 0.08) !important;
        --card-purple-icon: #9333ea !important;
        --card-purple-val: #581c87 !important;
        --card-slate-bg: rgba(15, 23, 42, 0.06) !important;
        --card-slate-icon: #0f172a !important;
        --card-slate-val: #0f172a !important;
        --ham-bg: #f0fdf4 !important;
        --ham-border: #bbf7d0 !important;
        --ham-text: #15803d !important;
        --spam-bg: #fef2f2 !important;
        --spam-border: #fecaca !important;
        --spam-text: #b91c1c !important;
        --step-num-bg: #0f172a !important;
        --step-num-color: #ffffff !important;
        --chevron-color: #94a3b8 !important;
        --shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
        --shadow-md: 0 6px 18px rgba(0, 0, 0, 0.07) !important;
    }
    """


# === Styling ==============================

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&family=Inter:wght@300;400;500;600;700;800;900&family=Cormorant+Garamond:wght@400;500;600&display=swap');

    :root {{
        --bg-main: #ffffff;
        --bg-sidebar: #ffffff;
        --bg-card: #ffffff;
        --bg-card-hover: #f8fafc;
        --border-color: #e2e8f0;
        --border-hover: #cbd5e1;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --btn-bg: #0f172a;
        --btn-color: #ffffff;
        --btn-hover: #1e293b;
        --textarea-bg: #ffffff;
        --textarea-border: #e2e8f0;
        --textarea-focus: #94a3b8;
        --sb-btn-color: #64748b;
        --sb-btn-hover: #f1f5f9;
        --divider-color: #e2e8f0;
        --footer-color: #94a3b8;
        --accent-line: linear-gradient(90deg, transparent, #cbd5e1, #94a3b8, transparent);
        --card-blue-bg: rgba(37, 99, 235, 0.08);
        --card-blue-icon: #2563eb;
        --card-blue-val: #1e3a8a;
        --card-green-bg: rgba(22, 163, 74, 0.08);
        --card-green-icon: #16a34a;
        --card-green-val: #14532d;
        --card-amber-bg: rgba(217, 119, 6, 0.08);
        --card-amber-icon: #d97706;
        --card-amber-val: #78350f;
        --card-purple-bg: rgba(147, 51, 234, 0.08);
        --card-purple-icon: #9333ea;
        --card-purple-val: #581c87;
        --card-slate-bg: rgba(15, 23, 42, 0.06);
        --card-slate-icon: #0f172a;
        --card-slate-val: #0f172a;
        --ham-bg: #f0fdf4;
        --ham-border: #bbf7d0;
        --ham-text: #15803d;
        --spam-bg: #fef2f2;
        --spam-border: #fecaca;
        --spam-text: #b91c1c;
        --step-num-bg: #0f172a;
        --step-num-color: #ffffff;
        --chevron-color: #94a3b8;
        --shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.04);
        --shadow-md: 0 6px 18px rgba(0, 0, 0, 0.07);
    }}

    @media (prefers-color-scheme: dark) {{
        :root {{
            --bg-main: #0b0f19;
            --bg-sidebar: #0f1422;
            --bg-card: #141a29;
            --bg-card-hover: #1c2438;
            --border-color: rgba(255, 255, 255, 0.09);
            --border-hover: rgba(255, 255, 255, 0.18);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --btn-bg: #f8fafc;
            --btn-color: #0b0f19;
            --btn-hover: #e2e8f0;
            --textarea-bg: #111726;
            --textarea-border: rgba(255, 255, 255, 0.12);
            --textarea-focus: rgba(255, 255, 255, 0.28);
            --sb-btn-color: #94a3b8;
            --sb-btn-hover: rgba(255, 255, 255, 0.06);
            --divider-color: rgba(255, 255, 255, 0.08);
            --footer-color: #64748b;
            --accent-line: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), rgba(255,255,255,0.08), transparent);
            --card-blue-bg: rgba(59, 130, 246, 0.16);
            --card-blue-icon: #60a5fa;
            --card-blue-val: #93c5fd;
            --card-green-bg: rgba(34, 197, 94, 0.16);
            --card-green-icon: #4ade80;
            --card-green-val: #86efac;
            --card-amber-bg: rgba(245, 158, 11, 0.16);
            --card-amber-icon: #fbbf24;
            --card-amber-val: #fde047;
            --card-purple-bg: rgba(168, 85, 247, 0.16);
            --card-purple-icon: #c084fc;
            --card-purple-val: #d8b4fe;
            --card-slate-bg: rgba(255, 255, 255, 0.08);
            --card-slate-icon: #cbd5e1;
            --card-slate-val: #f1f5f9;
            --ham-bg: rgba(34, 197, 94, 0.12);
            --ham-border: rgba(34, 197, 94, 0.3);
            --ham-text: #4ade80;
            --spam-bg: rgba(239, 68, 68, 0.12);
            --spam-border: rgba(239, 68, 68, 0.3);
            --spam-text: #f87171;
            --step-num-bg: #f8fafc;
            --step-num-color: #0b0f19;
            --chevron-color: #64748b;
            --shadow-sm: 0 1px 4px rgba(0, 0, 0, 0.4);
            --shadow-md: 0 6px 20px rgba(0, 0, 0, 0.5);
        }}
    }}

    {theme_override_css}

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--text-primary);
        background: var(--bg-main);
    }}

    .stApp {{
        background: var(--bg-main) !important;
    }}

    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], header[data-testid="stHeader"] {{
        background: var(--bg-main) !important;
    }}

    .block-container {{
        padding-top: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 820px;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    div[data-testid="stDecoration"] {{ display: none !important; }}

    section[data-testid="stSidebar"] {{
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding-top: 1rem !important;
        background: var(--bg-sidebar) !important;
    }}

    section[data-testid="stSidebar"] div.stButton > button {{
        background: transparent !important;
        border: none !important;
        border-radius: 10px !important;
        color: var(--sb-btn-color) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-transform: none !important;
        letter-spacing: 0.2px !important;
        box-shadow: none !important;
        padding: 0.6rem 0.75rem !important;
        min-height: 40px !important;
        margin-bottom: 0.15rem !important;
        text-align: left !important;
        justify-content: flex-start !important;
        transition: background 0.15s ease, color 0.15s ease !important;
        width: 100% !important;
    }}

    section[data-testid="stSidebar"] div.stButton > button:hover {{
        background: var(--sb-btn-hover) !important;
        color: var(--text-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    .sb-divider {{
        height: 1px;
        background: var(--divider-color);
        margin: 0.6rem 0.5rem;
    }}

    .sidebar-brand {{
        font-family: 'Great Vibes', cursive;
        font-size: 1.85rem;
        color: var(--text-primary);
        padding: 8px 4px 20px;
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}

    @keyframes shimmer {{
        0%   {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}

    .analyzing-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2.2rem 1rem 1rem;
        animation: fadeUp 0.35s ease forwards;
    }}

    .spin-dot {{
        width: 32px;
        height: 32px;
        border: 2.5px solid rgba(120, 120, 120, 0.15);
        border-top: 2.5px solid var(--btn-bg);
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
        margin-bottom: 0.9rem;
    }}

    .analyzing-text {{
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--text-secondary);
        letter-spacing: 0.3px;
    }}

    .hero-section {{
        position: relative;
        min-height: 270px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2.2rem 1rem 1.8rem;
        background: transparent;
        overflow: hidden;
    }}

    .hero-heading {{
        opacity: 1;
    }}

    .hero-line-1 {{
        font-family: 'Great Vibes', cursive;
        font-size: clamp(3rem, 8vw, 5.5rem);
        font-weight: 400;
        color: var(--text-primary);
        line-height: 1.1;
        margin: 0;
        opacity: 0;
        animation: fadeUp 0.6s ease forwards;
        animation-delay: 0.1s;
    }}

    .hero-line-2 {{
        font-family: 'Inter', sans-serif;
        font-size: clamp(0.95rem, 2.5vw, 1.15rem);
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 5px;
        text-transform: uppercase;
        line-height: 1;
        margin: 0.8rem 0 0 0;
        opacity: 0;
        animation: fadeUp 0.6s ease forwards;
        animation-delay: 0.2s;
    }}

    .hero-subtitle {{
        font-size: clamp(0.85rem, 2vw, 0.95rem);
        color: var(--text-secondary);
        margin-top: 0.8rem;
        max-width: 460px;
        text-align: center;
        line-height: 1.6;
        opacity: 0;
        animation: fadeUp 0.6s ease forwards;
        animation-delay: 0.3s;
    }}

    .hero-accent {{
        width: 120px;
        height: 1.5px;
        margin: 1.8rem auto 0;
        border-radius: 2px;
        background: var(--accent-line);
        background-size: 200% 100%;
        animation: fadeUp 0.6s ease forwards, shimmer 3s ease infinite;
        animation-delay: 0.5s, 0.5s;
        opacity: 0;
    }}

    .stTextArea textarea {{
        border: 1px solid var(--textarea-border) !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        color: var(--text-primary) !important;
        line-height: 1.7 !important;
        background: var(--textarea-bg) !important;
        transition: border-color 0.35s ease, box-shadow 0.35s ease !important;
    }}

    .stTextArea textarea::placeholder {{
        color: var(--text-muted) !important;
        font-weight: 300 !important;
    }}

    .stTextArea textarea:focus {{
        border-color: var(--textarea-focus) !important;
        box-shadow: 0 0 0 3px rgba(120, 120, 120, 0.08) !important;
    }}

    .stTextArea label {{ display: none !important; }}

    .main-content div.stButton > button {{
        background: var(--btn-bg) !important;
        color: var(--btn-color) !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.85rem 3rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: none !important;
        width: auto !important;
    }}

    .main-content div.stButton > button:hover {{
        background: var(--btn-hover) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
        color: var(--btn-color) !important;
    }}

    .main-content div.stButton > button:active {{
        transform: translateY(0) !important;
    }}

    .result-wrap {{
        animation: fadeUp 0.5s ease forwards;
        opacity: 0;
    }}

    .result-card {{
        padding: 22px 26px;
        border-radius: 14px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-top: 24px;
        box-shadow: var(--shadow-sm);
    }}

    .result-card.spam {{
        background: var(--spam-bg);
        border: 1px solid var(--spam-border);
    }}

    .result-card.ham {{
        background: var(--ham-bg);
        border: 1px solid var(--ham-border);
    }}

    .result-icon {{
        font-size: 1.4rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 2px;
    }}

    .result-card.spam .result-icon,
    .result-card.spam .result-label {{
        color: var(--spam-text);
    }}

    .result-card.ham .result-icon,
    .result-card.ham .result-label {{
        color: var(--ham-text);
    }}

    .result-label {{
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }}

    .result-message {{
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.5;
        margin-bottom: 6px;
    }}

    .result-confidence {{
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: var(--text-muted);
        letter-spacing: 0.5px;
    }}

    .result-error {{
        padding: 18px 24px;
        border-radius: 12px;
        background: var(--card-amber-bg);
        border: 1px solid var(--border-color);
        color: var(--card-amber-icon);
        font-size: 0.88rem;
        margin-top: 28px;
        animation: fadeUp 0.5s ease forwards;
        opacity: 0;
    }}

    .section-title-wrap {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 2rem 0 1.2rem;
    }}

    .section-icon {{
        display: flex;
        align-items: center;
        color: var(--text-muted);
    }}

    .section-title-text {{
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
    }}

    .section-divider {{
        flex: 1;
        height: 1px;
        background: var(--divider-color);
    }}

    .sb-stat-new {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.15rem 1rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 142px;
        box-sizing: border-box;
        width: 100%;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }}

    .sb-stat-new:hover {{
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }}

    .sb-stat-top {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }}

    .sb-stat-title {{
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .sb-stat-icon {{
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}

    .sb-stat-icon.blue   {{ background: var(--card-blue-bg);   color: var(--card-blue-icon); }}
    .sb-stat-icon.green  {{ background: var(--card-green-bg);  color: var(--card-green-icon); }}
    .sb-stat-icon.amber  {{ background: var(--card-amber-bg);  color: var(--card-amber-icon); }}
    .sb-stat-icon.purple {{ background: var(--card-purple-bg); color: var(--card-purple-icon); }}
    .sb-stat-icon.slate  {{ background: var(--card-slate-bg);  color: var(--card-slate-icon); }}

    .sb-stat-val-new {{
        font-size: 1.55rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        white-space: nowrap;
        line-height: 1.1;
    }}

    .sb-stat-val-new.blue   {{ color: var(--card-blue-val); }}
    .sb-stat-val-new.green  {{ color: var(--card-green-val); }}
    .sb-stat-val-new.amber  {{ color: var(--card-amber-val); }}
    .sb-stat-val-new.purple {{ color: var(--card-purple-val); }}
    .sb-stat-val-new.slate  {{ color: var(--card-slate-val); }}

    .sb-stat-sub {{
        font-size: 0.72rem;
        color: var(--text-muted);
        letter-spacing: 0.2px;
        line-height: 1.3;
    }}

    .mi-step-card {{
        display: flex;
        align-items: center;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.25s ease;
    }}

    .mi-step-card:hover {{
        background: var(--bg-card-hover);
        border-color: var(--border-hover);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }}

    .mi-step-num {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: var(--step-num-bg);
        color: var(--step-num-color);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
        margin-right: 1.2rem;
    }}

    .mi-step-content {{ flex: 1; }}

    .mi-step-title {{
        font-size: 0.92rem;
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}

    .mi-step-desc {{
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }}

    .mi-step-chevron {{
        flex-shrink: 0;
        margin-left: 1rem;
        display: flex;
        align-items: center;
    }}

    .mi-step-chevron svg {{
        stroke: var(--chevron-color);
    }}

    .footer-note {{
        text-align: center;
        color: var(--footer-color);
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding: 0.8rem 0;
        border-top: 1px solid var(--divider-color);
        font-style: italic;
    }}

    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border-color); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--border-hover); }}

    @media (min-width: 769px) and (max-width: 1024px) {{
        .block-container {{
            max-width: 92% !important;
            padding: 2rem 1.5rem 3rem !important;
        }}

        .hero-line-1 {{
            font-size: clamp(3.2rem, 7vw, 4.5rem) !important;
        }}

        .sb-stat-new {{
            padding: 1rem 0.85rem !important;
        }}

        .sb-stat-val-new {{
            font-size: 1.4rem !important;
        }}
    }}

    @media (max-width: 768px) {{
        section[data-testid="stSidebar"] {{
            width: 250px !important;
            min-width: 250px !important;
            max-width: 250px !important;
        }}

        .block-container {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 100% !important;
        }}

        .hero-section {{
            min-height: 200px !important;
            padding: 1.8rem 0.5rem 1.2rem !important;
        }}

        .hero-line-1 {{
            font-size: clamp(2.6rem, 11vw, 3.8rem) !important;
        }}

        .hero-line-2 {{
            font-size: 0.85rem !important;
            letter-spacing: 3px !important;
        }}

        .hero-subtitle {{
            font-size: 0.85rem !important;
            padding: 0 0.5rem !important;
            line-height: 1.5 !important;
        }}

        .stTextArea textarea {{
            font-size: 16px !important;
            padding: 14px 16px !important;
            min-height: 130px !important;
        }}

        .main-content div.stButton > button {{
            width: 100% !important;
            padding: 0.95rem 1.5rem !important;
            font-size: 0.9rem !important;
            min-height: 48px !important;
        }}

        [data-testid="stHorizontalBlock"] {{
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 10px !important;
        }}

        [data-testid="column"] {{
            flex: 1 1 calc(50% - 10px) !important;
            min-width: calc(50% - 10px) !important;
        }}

        [data-testid="column"]:last-child {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}

        .sb-stat-new {{
            min-height: 122px !important;
            padding: 1rem 0.85rem !important;
        }}

        .sb-stat-val-new {{
            font-size: 1.35rem !important;
        }}

        .mi-step-card {{
            padding: 1rem !important;
        }}

        .mi-step-num {{
            width: 30px !important;
            height: 30px !important;
            font-size: 0.8rem !important;
            margin-right: 0.85rem !important;
        }}

        .mi-step-title {{
            font-size: 0.88rem !important;
        }}

        .mi-step-desc {{
            font-size: 0.78rem !important;
        }}

        .result-card {{
            padding: 18px 20px !important;
            gap: 14px !important;
        }}
    }}

    @media (max-width: 390px) {{
        .hero-line-1 {{
            font-size: 2.3rem !important;
        }}
        .hero-line-2 {{
            font-size: 0.75rem !important;
            letter-spacing: 2px !important;
        }}
        .sb-stat-val-new {{
            font-size: 1.2rem !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)


# === Sidebar Navigation ==============================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">SpamLens</div>', unsafe_allow_html=True)

    nav_items = [
        ("home",       "Home"),
        ("model_info", "Model Info"),
    ]

    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.markdown('<div class="sb-divider">&nbsp;</div>', unsafe_allow_html=True)

    theme_col1, theme_col2 = st.columns(2)
    with theme_col1:
        if st.button("☀ Light", key="btn_theme_light", use_container_width=True):
            st.session_state["theme"] = "light"
            st.rerun()
    with theme_col2:
        if st.button("🌙 Dark", key="btn_theme_dark", use_container_width=True):
            st.session_state["theme"] = "dark"
            st.rerun()

    cur_theme = st.session_state.get("theme", "system")
    theme_status = "Auto (System)" if cur_theme == "system" else cur_theme.capitalize()

    st.markdown(
        f'<p style="color:var(--footer-color);font-size:0.68rem;text-align:center;margin-top:0.4rem;letter-spacing:0.3px;">'
        f'Theme: {theme_status}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:var(--footer-color);font-size:0.7rem;text-align:center;margin-top:0.5rem;">'
        'SpamLens v1.0 · Logistic Regression</p>',
        unsafe_allow_html=True,
    )


# === Page: Home ==============================

if st.session_state["page"] == "home":

    st.markdown("""
        <div class="hero-section">
            <div class="hero-heading">
                <div class="hero-line-1">SpamLens</div>
                <div class="hero-line-2">Spam Ends Here</div>
            </div>
            <div class="hero-subtitle">
                Paste any email or message. Get an instant verdict.
            </div>
            <div class="hero-accent">&nbsp;</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    message = st.text_area(
        "message_input",
        placeholder="Paste your email or message here...",
        height=160,
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze", use_container_width=False)

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not message or not message.strip():
            st.markdown(
                '<div class="result-error">Please enter a message to analyze.</div>',
                unsafe_allow_html=True,
            )
        else:
            placeholder = st.empty()
            placeholder.markdown("""
                <div class="analyzing-wrap">
                    <div class="spin-dot"></div>
                    <div class="analyzing-text">Analyzing message patterns…</div>
                </div>
            """, unsafe_allow_html=True)

            time.sleep(0.9)
            placeholder.empty()

            input_df   = pd.DataFrame({'Message': [message]})
            proba      = model.predict_proba(input_df)[0]
            prediction = model.classes_[proba.argmax()]
            confidence = round(proba.max() * 100, 1)

            is_spam = prediction == 'spam'
            cls     = 'spam' if is_spam else 'ham'
            icon_svg = (
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
                if is_spam else
                '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'
            )
            label   = 'Spam Detected' if is_spam else 'Message is Safe'
            msg     = (
                'This message shows signs of unsolicited content. Exercise caution.'
                if is_spam else
                'This message appears to be legitimate. No threats found.'
            )

            st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-card {cls}">
                        <div class="result-icon">{icon_svg}</div>
                        <div>
                            <div class="result-label">{label}</div>
                            <div class="result-message">{msg}</div>
                            <div class="result-confidence">Confidence: {confidence}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("""
        <div class="footer-note">
            Predictions are based on training data patterns. Results are indicative, not absolute.
        </div>
    """, unsafe_allow_html=True)


# === Page: Model Info ==============================

elif st.session_state["page"] == "model_info":

    st.markdown("""
        <div class="hero-section">
            <div class="hero-heading">
                <div class="hero-line-1">SpamLens</div>
                <div class="hero-line-2">The Model Behind It</div>
            </div>
            <div class="hero-subtitle">
                SpamLens classifies messages as spam or ham using a tuned Logistic Regression pipeline.
            </div>
            <div class="hero-accent">&nbsp;</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title-wrap">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10"></line>
                    <line x1="12" y1="20" x2="12" y2="4"></line>
                    <line x1="6"  y1="20" x2="6"  y2="14"></line>
                </svg>
            </div>
            <div class="section-title-text">Model Performance</div>
            <div class="section-divider"></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""<div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Accuracy</div>
                <div class="sb-stat-icon blue">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                </div>
            </div>
            <div>
                <div class="sb-stat-val-new blue">{metrics['accuracy']}%</div>
                <div class="sb-stat-sub">Overall correct</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Precision</div>
                <div class="sb-stat-icon green">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                </div>
            </div>
            <div>
                <div class="sb-stat-val-new green">{metrics['precision']}%</div>
                <div class="sb-stat-sub">Spam precision</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Recall</div>
                <div class="sb-stat-icon amber">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                </div>
            </div>
            <div>
                <div class="sb-stat-val-new amber">{metrics['recall']}%</div>
                <div class="sb-stat-sub">Detection rate</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">F1 Score</div>
                <div class="sb-stat-icon purple">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                </div>
            </div>
            <div>
                <div class="sb-stat-val-new purple">{metrics['f1']}%</div>
                <div class="sb-stat-sub">Harmonic mean</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""<div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Dataset</div>
                <div class="sb-stat-icon slate">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
                </div>
            </div>
            <div>
                <div class="sb-stat-val-new slate">{metrics['total_samples']:,}</div>
                <div class="sb-stat-sub">Total samples</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title-wrap">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1
                             0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0
                             0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2
                             2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65
                             1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2
                             0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0
                             .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2
                             2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65
                             1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1
                             0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0
                             0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0
                             1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1
                             1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0
                             0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65
                             1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51
                             1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65
                             1.65 0 0 0-1.51 1z"></path>
                </svg>
            </div>
            <div class="section-title-text">How It Works</div>
            <div class="section-divider"></div>
        </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Raw Text Input",
         "An email or SMS message is fed into the pipeline."),
        ("Feature Engineering",
         "12 features will be extracted from the message."),
        ("Logistic Regression",
         "A tuned binary classifier outputs the final spam or ham prediction with confidence."),
    ]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""<div class="mi-step-card">
            <div class="mi-step-num">{i}</div>
            <div class="mi-step-content">
                <div class="mi-step-title">{title}</div>
                <div class="mi-step-desc">{desc}</div>
            </div>
            <div class="mi-step-chevron">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                     stroke="#cbd5e1" stroke-width="2" stroke-linecap="round"
                     stroke-linejoin="round">
                    <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div class="footer-note">
            Built with Logistic Regression · sklearn Pipeline · Streamlit
        </div>
    """, unsafe_allow_html=True)
