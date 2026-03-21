"""
app/streamlit_app.py
--------------------
Interactive Streamlit UI for the Credit Card Fraud Detection system.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Path fix ───────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e17;
    color: #e8eaf0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1520;
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] * {
    color: #c5cfe0 !important;
}

/* Header */
.fraud-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f2744 50%, #0a1628 100%);
    border: 1px solid #1e3a5a;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.fraud-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0,180,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.fraud-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.fraud-header p {
    color: #7a9ab8;
    font-size: 0.95rem;
    margin: 0;
    font-family: 'IBM Plex Mono', monospace;
}
.shield-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00b4ff22, #0066ff22);
    border: 1px solid #00b4ff44;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #00b4ff;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-card {
    background: #0f1a2b;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 20px 22px;
    text-align: center;
}
.metric-card .val {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .lbl {
    font-size: 0.78rem;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Result cards */
.result-safe {
    background: linear-gradient(135deg, #001a0d, #002b15);
    border: 2px solid #00cc6644;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
}
.result-fraud {
    background: linear-gradient(135deg, #1a0000, #2b0000);
    border: 2px solid #ff003344;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
}
.result-icon {
    font-size: 3rem;
    margin-bottom: 10px;
}
.result-label-safe {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00e676;
    letter-spacing: -0.5px;
}
.result-label-fraud {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ff1744;
    letter-spacing: -0.5px;
}
.result-prob {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: #7a9ab8;
    margin-top: 6px;
}

/* Gauge container */
.gauge-wrap {
    background: #0f1a2b;
    border: 1px solid #1e3050;
    border-radius: 10px;
    padding: 18px;
}

/* Section headings */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a7a9a;
    border-bottom: 1px solid #1e3050;
    padding-bottom: 8px;
    margin-bottom: 16px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Sliders, inputs */
[data-testid="stSlider"] > div > div { color: #00b4ff !important; }
.stSlider [data-testid="stThumbValue"] { color: #00b4ff !important; }
label { color: #8aaccc !important; font-size: 0.82rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #0f1520;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #1e2a3a;
}
.stTabs [data-baseweb="tab"] {
    color: #5a7a9a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3050 !important;
    color: #00d4ff !important;
    border-radius: 6px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #003a6e, #005299);
    color: #ffffff;
    border: 1px solid #0066cc44;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 10px 24px;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0055aa, #0077cc);
    border-color: #0099ff88;
}

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e3050; border-radius: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load predictor ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        from src.predict import FraudPredictor
        p = FraudPredictor()
        p.load()
        return p, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


predictor, load_error = load_predictor()


# ── Gauge chart ────────────────────────────────────────────────────────────────
def draw_gauge(probability: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#0f1a2b")
    ax.set_facecolor("#0f1a2b")

    # Track
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#1e3050", linewidth=14, solid_capstyle="round")

    # Filled arc
    fill_end = np.pi - probability * np.pi
    theta_fill = np.linspace(np.pi, fill_end, 300)
    color = "#00e676" if probability < 0.4 else ("#ffab00" if probability < 0.7 else "#ff1744")
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=14, solid_capstyle="round")

    # Needle
    angle = np.pi - probability * np.pi
    ax.annotate(
        "", xy=(0.65 * np.cos(angle), 0.65 * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#ffffff", lw=2, mutation_scale=12),
    )
    ax.plot(0, 0, "o", color="#ffffff", markersize=6, zorder=5)

    # Labels
    ax.text(0, -0.22, f"{probability * 100:.1f}%", ha="center", va="center",
            fontsize=18, fontweight="bold", color=color,
            fontfamily="monospace")
    ax.text(-1.05, -0.15, "0%", ha="center", color="#3a5a7a", fontsize=7)
    ax.text(1.05, -0.15, "100%", ha="center", color="#3a5a7a", fontsize=7)
    ax.text(0, 0.55, "FRAUD RISK", ha="center", color="#3a5a7a", fontsize=7,
            fontfamily="monospace", fontweight="bold")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.45, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ── Feature distribution mini-chart ───────────────────────────────────────────
def draw_feature_bars(feature_values: dict) -> plt.Figure:
    v_features = {k: v for k, v in feature_values.items() if k.startswith("V")}
    keys = list(v_features.keys())
    vals = [v_features[k] for k in keys]
    colors = ["#ff4444" if v < -2 else "#ffaa00" if v < 0 else "#00cc88" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0f1a2b")
    ax.set_facecolor("#0f1a2b")
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.7, edgecolor="none")
    ax.axhline(0, color="#2a4060", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7, color="#5a7a9a", fontfamily="monospace")
    ax.tick_params(axis="y", colors="#3a5a7a", labelsize=7)
    ax.set_title("V1–V28 Feature Values", color="#5a8aaa", fontsize=9,
                 fontfamily="monospace", fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3050")
    ax.set_facecolor("#0f1a2b")
    plt.tight_layout()
    return fig


# ── Default sample values ──────────────────────────────────────────────────────
NORMAL_TX = dict(
    Time=406, Amount=149.62,
    V1=-1.3598071, V2=-0.0727812, V3=2.5363467, V4=1.3781553,
    V5=-0.3383208, V6=0.4623878, V7=0.2395986, V8=0.0986980,
    V9=0.3637870, V10=0.0907942, V11=-0.5515995, V12=-0.6178009,
    V13=-0.9913898, V14=-0.3111694, V15=1.4681770, V16=-0.4704005,
    V17=0.2079708, V18=0.0257906, V19=0.4039936, V20=0.2514121,
    V21=-0.0183067, V22=0.2778376, V23=-0.1104739, V24=0.0669281,
    V25=0.1285394, V26=-0.1891148, V27=0.1335584, V28=-0.0210530,
)

FRAUD_TX = dict(
    Time=0, Amount=0.0,
    V1=-2.3122265, V2=1.9519694, V3=-1.6098228, V4=3.9979055,
    V5=-0.5220732, V6=-1.4265453, V7=-2.5373820, V8=1.3916882,
    V9=-2.7700270, V10=-2.7722813, V11=3.2020302, V12=-2.8996780,
    V13=-0.5952198, V14=-4.2895250, V15=0.3898805, V16=-1.1407429,
    V17=-2.8301056, V18=-0.0168224, V19=0.4169498, V20=0.1267031,
    V21=0.5177640, V22=-0.0354049, V23=-0.4653461, V24=0.1797855,
    V25=-0.3662423, V26=0.0592697, V27=-0.0735274, V28=-0.0684070,
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:1.3rem; font-weight:800; color:#ffffff; letter-spacing:-0.3px;'>🛡️ FraudShield</div>
        <div style='font-size:0.72rem; color:#3a6a8a; font-family: monospace; margin-top:3px;'>ML DETECTION SYSTEM v1.0</div>
    </div>
    <hr style='border-color:#1e2a3a; margin: 8px 0 20px 0;'>
    """, unsafe_allow_html=True)

    # Model status
    if predictor:
        st.markdown(f"""
        <div style='background:#001a0a; border:1px solid #00cc4422; border-radius:8px; padding:12px 14px; margin-bottom:18px;'>
            <div style='font-size:0.7rem; color:#3a9a5a; font-family:monospace; font-weight:700; letter-spacing:1px;'>● MODEL LOADED</div>
            <div style='font-size:0.82rem; color:#a0c8b0; margin-top:4px;'>{predictor.model_name}</div>
            <div style='font-size:0.72rem; color:#3a6a5a; font-family:monospace; margin-top:2px;'>threshold = {predictor.threshold:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#1a0000; border:1px solid #cc000022; border-radius:8px; padding:12px 14px; margin-bottom:18px;'>
            <div style='font-size:0.7rem; color:#cc4444; font-family:monospace; font-weight:700; letter-spacing:1px;'>● MODEL NOT FOUND</div>
            <div style='font-size:0.75rem; color:#7a4444; margin-top:4px;'>Run: python train.py</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">QUICK LOAD</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Normal TX"):
            st.session_state["sample_type"] = "normal"
    with col_b:
        if st.button("⚠️ Fraud TX"):
            st.session_state["sample_type"] = "fraud"

    st.markdown('<br><div class="section-title">THRESHOLD OVERRIDE</div>', unsafe_allow_html=True)
    custom_threshold = st.slider(
        "Decision threshold", min_value=0.05, max_value=0.95,
        value=float(predictor.threshold) if predictor else 0.50,
        step=0.01, help="Lower = more sensitive (catches more fraud, more false alarms)"
    )

    st.markdown('<br><div class="section-title">NAVIGATION</div>', unsafe_allow_html=True)
    page = st.radio(
        "", ["🔍 Single Predict", "📂 Batch Upload", "📊 Model Info"],
        label_visibility="collapsed"
    )

st.markdown("""
<hr style='border-color:#1e2a3a; margin: 20px 0 10px 0;'>

<div style='text-align:center;'>
    <div style='font-size:0.68rem; color:#2a4a6a; font-family:monospace;'>
        Built on FastAPI + XGBoost + SHAP
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fraud-header">
    <div class="shield-badge">REAL-TIME DETECTION</div>
    <h1>🛡️ FraudShield Detection System</h1>
    <p>Credit card fraud analysis powered by ML — enter transaction features to assess risk</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SINGLE PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "🔍 Single Predict" in page:

    sample = FRAUD_TX if st.session_state.get("sample_type") == "fraud" else NORMAL_TX

    tab1, tab2 = st.tabs(["  🎛️  FEATURE INPUTS  ", "  📋  JSON INPUT  "])

    # ── Tab 1: Sliders ──────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">TRANSACTION DETAILS</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            time_val = st.number_input("Time (seconds)", value=float(sample["Time"]), step=1.0)
        with c2:
            amount = st.number_input("Amount ($)", value=float(sample["Amount"]), min_value=0.0, step=0.01)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"💡 Threshold: **{custom_threshold:.2f}**", icon=None)

        st.markdown('<br><div class="section-title">PCA FEATURES  (V1 – V28)</div>', unsafe_allow_html=True)

        v_vals = {}
        cols = st.columns(7)
        for i in range(1, 29):
            col = cols[(i - 1) % 7]
            with col:
                v_vals[f"V{i}"] = st.number_input(
                    f"V{i}", value=float(sample[f"V{i}"]),
                    format="%.4f", step=0.001,
                    key=f"v_{i}",
                )

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍  ANALYSE TRANSACTION", use_container_width=True)

        raw_input = {"Time": time_val, "Amount": amount, **v_vals}

    # ── Tab 2: JSON ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">PASTE JSON</div>', unsafe_allow_html=True)
        json_default = json.dumps({k: round(v, 5) for k, v in sample.items()}, indent=2)
        json_text = st.text_area("Transaction JSON", value=json_default, height=320,
                                  label_visibility="collapsed")
        run_btn_json = st.button("🔍  ANALYSE JSON", use_container_width=True, key="json_btn")

        if run_btn_json:
            try:
                raw_input = json.loads(json_text)
                run_btn = True
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                run_btn = False

    # ── Run prediction ───────────────────────────────────────────────────────
    if run_btn:
        if not predictor:
            st.error("⚠️  No model loaded. Run `python train.py` first to train and save the model.")
        else:
            with st.spinner("Analysing transaction …"):
                time.sleep(0.3)  # brief visual pause for UX
                # Temporarily override threshold
                original_threshold = predictor._meta.get("threshold", 0.5)
                predictor._meta["threshold"] = custom_threshold

                result = predictor.predict(raw_input)

                predictor._meta["threshold"] = original_threshold  # restore

            prob   = result["fraud_probability"]
            is_fraud = result["is_fraud"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">PREDICTION RESULT</div>', unsafe_allow_html=True)

            res_col, gauge_col, detail_col = st.columns([1.2, 1, 1.2])

            with res_col:
                if is_fraud:
                    st.markdown(f"""
                    <div class="result-fraud">
                        <div class="result-icon">🚨</div>
                        <div class="result-label-fraud">FRAUD DETECTED</div>
                        <div class="result-prob">probability: {prob:.4f}</div>
                        <div class="result-prob" style="margin-top:8px; color:#cc3333;">
                            ⬆ Exceeds threshold of {custom_threshold:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <div class="result-icon">✅</div>
                        <div class="result-label-safe">TRANSACTION SAFE</div>
                        <div class="result-prob">probability: {prob:.4f}</div>
                        <div class="result-prob" style="margin-top:8px; color:#336644;">
                            ⬇ Below threshold of {custom_threshold:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with gauge_col:
                st.markdown('<div class="gauge-wrap">', unsafe_allow_html=True)
                fig_gauge = draw_gauge(prob)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close(fig_gauge)
                st.markdown("</div>", unsafe_allow_html=True)

            with detail_col:
                st.markdown("""
                <div style='background:#0f1a2b; border:1px solid #1e3050; border-radius:10px; padding:18px;'>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <table style='width:100%; font-family:monospace; font-size:0.82rem; border-collapse:collapse;'>
                    <tr><td style='color:#3a6a8a; padding:5px 0;'>Model</td>
                        <td style='color:#c0d8f0; text-align:right;'>{result['model_name']}</td></tr>
                    <tr><td style='color:#3a6a8a; padding:5px 0;'>Fraud Prob</td>
                        <td style='color:#{"ff4466" if is_fraud else "00e676"}; text-align:right; font-weight:700;'>{prob:.6f}</td></tr>
                    <tr><td style='color:#3a6a8a; padding:5px 0;'>Threshold</td>
                        <td style='color:#c0d8f0; text-align:right;'>{custom_threshold:.2f}</td></tr>
                    <tr><td style='color:#3a6a8a; padding:5px 0;'>Decision</td>
                        <td style='color:#{"ff4466" if is_fraud else "00e676"}; text-align:right; font-weight:700;'>{"⛔ BLOCK" if is_fraud else "✅ PASS"}</td></tr>
                    <tr><td style='color:#3a6a8a; padding:5px 0;'>Amount</td>
                        <td style='color:#c0d8f0; text-align:right;'>${raw_input.get("Amount", 0):.2f}</td></tr>
                </table>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Feature bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bars = draw_feature_bars(raw_input)
            st.pyplot(fig_bars, use_container_width=True)
            plt.close(fig_bars)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif "📂 Batch Upload" in page:

    st.markdown('<div class="section-title">BATCH TRANSACTION ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#5a7a9a; font-size:0.88rem; margin-bottom:20px;'>
    Upload a CSV file containing multiple transactions. Must include columns V1–V28, Amount, and optionally Time and Class.
    </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"<div style='color:#5a9a7a; font-family:monospace; font-size:0.82rem; margin-bottom:12px;'>✓ Loaded {len(df):,} rows × {len(df.columns)} columns</div>", unsafe_allow_html=True)

        if not predictor:
            st.error("No model loaded. Run `python train.py` first.")
        else:
            required = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            missing_cols = [c for c in required if c not in df.columns]

            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                with st.spinner(f"Scoring {len(df):,} transactions …"):
                    records = df[required + (["Time"] if "Time" in df.columns else [])].to_dict(orient="records")
                    results = predictor.predict_batch(records)

                proba_col   = [r["fraud_probability"] for r in results]
                is_fraud_col = [r["is_fraud"] for r in results]

                # Override with custom threshold
                is_fraud_col = [p >= custom_threshold for p in proba_col]

                df["fraud_probability"] = proba_col
                df["is_fraud"]          = is_fraud_col

                n_fraud = sum(is_fraud_col)
                n_total = len(df)
                pct     = n_fraud / n_total * 100

                # Summary metrics
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                metrics = [
                    ("Total Transactions", f"{n_total:,}"),
                    ("Flagged as Fraud", f"{n_fraud:,}"),
                    ("Fraud Rate", f"{pct:.2f}%"),
                    ("Avg Fraud Prob", f"{np.mean(proba_col):.4f}"),
                ]
                for col, (label, value) in zip([m1, m2, m3, m4], metrics):
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="val">{value}</div>
                            <div class="lbl">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                chart_col, table_col = st.columns([1, 1.5])

                with chart_col:
                    st.markdown('<div class="section-title">SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor("#0f1a2b")
                    ax.set_facecolor("#0f1a2b")
                    ax.hist(proba_col, bins=50, color="#00aaff", edgecolor="none", alpha=0.85)
                    ax.axvline(custom_threshold, color="#ff4466", linewidth=1.5,
                               linestyle="--", label=f"Threshold {custom_threshold:.2f}")
                    ax.set_xlabel("Fraud Probability", color="#5a7a9a", fontsize=8)
                    ax.set_ylabel("Count", color="#5a7a9a", fontsize=8)
                    ax.tick_params(colors="#3a5a7a", labelsize=7)
                    ax.legend(fontsize=7, labelcolor="#aaaaaa",
                              facecolor="#0f1a2b", edgecolor="#1e3050")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#1e3050")
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                with table_col:
                    st.markdown('<div class="section-title">TOP FLAGGED TRANSACTIONS</div>', unsafe_allow_html=True)
                    fraud_df = df[df["is_fraud"]].sort_values("fraud_probability", ascending=False).head(20)
                    show_cols = ["fraud_probability", "Amount"] + (["Class"] if "Class" in df.columns else [])
                    st.dataframe(
                        fraud_df[show_cols].reset_index(drop=True).style.background_gradient(
                            subset=["fraud_probability"], cmap="Reds"
                        ),
                        use_container_width=True,
                        height=260,
                    )

                # Download
                st.markdown("<br>", unsafe_allow_html=True)
                csv_out = df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️  Download Results CSV",
                    data=csv_out,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    else:
        # Placeholder hint
        st.markdown("""
        <div style='background:#0f1a2b; border:1px dashed #1e3050; border-radius:10px;
                    padding:40px; text-align:center; color:#2a4a6a;'>
            <div style='font-size:2.5rem; margin-bottom:12px;'>📂</div>
            <div style='font-family:monospace; font-size:0.85rem;'>
                Upload a CSV with V1–V28 + Amount columns to score in bulk
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Info" in page:

    st.markdown('<div class="section-title">MODEL OVERVIEW</div>', unsafe_allow_html=True)

    info_col, pipeline_col = st.columns([1, 1.3])

    with info_col:
        if predictor:
            st.markdown(f"""
            <div style='background:#0f1a2b; border:1px solid #1e3050; border-radius:10px; padding:20px;'>
            <table style='width:100%; font-family:monospace; font-size:0.85rem; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #1e3050;'>
                    <td style='color:#3a6a8a; padding:10px 0;'>Active Model</td>
                    <td style='color:#00d4ff; text-align:right; font-weight:700;'>{predictor.model_name}</td>
                </tr>
                <tr style='border-bottom:1px solid #1e3050;'>
                    <td style='color:#3a6a8a; padding:10px 0;'>Decision Threshold</td>
                    <td style='color:#c0d8f0; text-align:right;'>{predictor.threshold:.2f}</td>
                </tr>
                <tr style='border-bottom:1px solid #1e3050;'>
                    <td style='color:#3a6a8a; padding:10px 0;'>Custom Threshold</td>
                    <td style='color:#ffaa00; text-align:right;'>{custom_threshold:.2f}</td>
                </tr>
                <tr>
                    <td style='color:#3a6a8a; padding:10px 0;'>Status</td>
                    <td style='color:#00e676; text-align:right;'>● Loaded</td>
                </tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("No model loaded. Train first: `python train.py`")

    with pipeline_col:
        st.markdown('<div class="section-title">PIPELINE STEPS</div>', unsafe_allow_html=True)
        steps = [
            ("01", "Data Loading", "pandas CSV → DataFrame"),
            ("02", "Missing Values", "Drop null rows"),
            ("03", "Time Engineering", "Cyclic hour_sin / hour_cos"),
            ("04", "Amount Scaling", "StandardScaler → saved to disk"),
            ("05", "Class Imbalance", "SMOTE oversampling on train set"),
            ("06", "Model Training", "LR / Random Forest / XGBoost"),
            ("07", "Model Selection", "Best ROC-AUC on test set"),
            ("08", "Threshold Tuning", "Max F2-score (optional)"),
            ("09", "Serialisation", "joblib → models/"),
            ("10", "Inference", "FraudPredictor class → API + UI"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:center; padding:8px 0; border-bottom:1px solid #0f1a2b;'>
                <div style='font-family:monospace; font-size:0.7rem; color:#1e5a8a;
                            background:#0d1f30; border-radius:4px; padding:3px 7px;
                            margin-right:14px; min-width:28px; text-align:center;'>{num}</div>
                <div>
                    <div style='font-size:0.85rem; color:#c0d8f0; font-weight:600;'>{title}</div>
                    <div style='font-size:0.75rem; color:#3a6a8a; font-family:monospace;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">CANDIDATE MODELS</div>', unsafe_allow_html=True)

    model_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Imbalance Strategy": ["class_weight='balanced'", "class_weight='balanced'", "scale_pos_weight=100"],
        "Key Hyperparams": ["max_iter=1000, solver=lbfgs", "n_estimators=300, max_depth=None", "n_estimators=300, lr=0.05, max_depth=6"],
        "Selection Metric": ["ROC-AUC", "ROC-AUC", "ROC-AUC"],
    }
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    # Check if plots exist
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">EVALUATION PLOTS</div>', unsafe_allow_html=True)
    plots_dir = "plots"
    plot_files = []
    if os.path.isdir(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith(".png")]

    if plot_files:
        cols = st.columns(2)
        for i, fname in enumerate(sorted(plot_files)[:6]):
            with cols[i % 2]:
                st.image(os.path.join(plots_dir, fname), caption=fname.replace("_", " ").replace(".png", ""), use_container_width=True)
    else:
        st.markdown("""
        <div style='background:#0f1a2b; border:1px dashed #1e3050; border-radius:10px;
                    padding:30px; text-align:center; color:#2a4a6a;'>
            <div style='font-family:monospace; font-size:0.85rem;'>
                No evaluation plots found.<br>Run: <code style='color:#00aaff;'>python train.py --evaluate</code>
            </div>
        </div>
        """, unsafe_allow_html=True)