"""
🎬 CHURNFLIX — Netflix Customer Churn Prediction Studio
A cinematic, animated Streamlit app for predicting subscriber churn.
"""

import hashlib
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Churnflix | Churn Prediction Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Netflix-inspired dark theme with animation
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    /* ---------- App background ---------- */
    .stApp {
        background: radial-gradient(ellipse at top left, #1a0505 0%, #0d0d0d 45%, #000000 100%);
        color: #f5f5f5;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0b0b 0%, #1a0808 100%);
        border-right: 1px solid #2b2b2b;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Animations ---------- */
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(24px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @keyframes pulseGlow {
        0% {box-shadow: 0 0 6px rgba(229,9,20,0.4);}
        50% {box-shadow: 0 0 26px rgba(229,9,20,0.9);}
        100% {box-shadow: 0 0 6px rgba(229,9,20,0.4);}
    }
    @keyframes pulseGlowGreen {
        0% {box-shadow: 0 0 6px rgba(30,215,96,0.4);}
        50% {box-shadow: 0 0 26px rgba(30,215,96,0.9);}
        100% {box-shadow: 0 0 6px rgba(30,215,96,0.4);}
    }
    @keyframes shake {
        0%, 100% {transform: translateX(0);}
        20% {transform: translateX(-6px);}
        40% {transform: translateX(6px);}
        60% {transform: translateX(-4px);}
        80% {transform: translateX(4px);}
    }
    @keyframes shimmer {
        0% {background-position: -400px 0;}
        100% {background-position: 400px 0;}
    }
    @keyframes typing {
        from {width: 0;}
        to {width: 100%;}
    }
    @keyframes float {
        0%, 100% {transform: translateY(0px);}
        50% {transform: translateY(-8px);}
    }

    .fade-in { animation: fadeInUp 0.7s ease-out both; }
    .fade-in-1 { animation: fadeInUp 0.7s ease-out 0.05s both; }
    .fade-in-2 { animation: fadeInUp 0.7s ease-out 0.15s both; }
    .fade-in-3 { animation: fadeInUp 0.7s ease-out 0.25s both; }
    .fade-in-4 { animation: fadeInUp 0.7s ease-out 0.35s both; }

    /* ---------- Hero header ---------- */
    .hero {
        text-align: center;
        padding: 18px 0 6px 0;
    }
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4.2rem;
        letter-spacing: 4px;
        margin: 0;
        background: linear-gradient(90deg, #E50914, #ff5f6d, #E50914, #b0060c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease infinite;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #b5b5b5;
        overflow: hidden;
        white-space: nowrap;
        border-right: 3px solid #E50914;
        width: fit-content;
        margin: 6px auto 0 auto;
        animation: typing 2.6s steps(40, end);
        letter-spacing: 1px;
    }

    /* ---------- Cards ---------- */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 22px 24px;
        backdrop-filter: blur(6px);
        transition: transform 0.25s ease, border 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(229,9,20,0.5);
    }

    .risk-card-high {
        background: linear-gradient(135deg, rgba(229,9,20,0.18), rgba(80,0,5,0.35));
        border: 1px solid #E50914;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        animation: shake 0.6s ease-in-out, pulseGlow 2s infinite ease-in-out;
    }
    .risk-card-medium {
        background: linear-gradient(135deg, rgba(255,176,32,0.15), rgba(80,55,0,0.3));
        border: 1px solid #FFB020;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 16px rgba(255,176,32,0.35);
    }
    .risk-card-low {
        background: linear-gradient(135deg, rgba(30,215,96,0.15), rgba(0,60,25,0.3));
        border: 1px solid #1ED760;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        animation: pulseGlowGreen 2.4s infinite ease-in-out;
    }
    .risk-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.4rem;
        letter-spacing: 3px;
        margin: 0;
    }
    .risk-emoji { font-size: 3rem; animation: float 3s ease-in-out infinite; display:inline-block; }

    /* ---------- Chips / badges ---------- */
    .chip {
        display: inline-block;
        padding: 5px 14px;
        margin: 3px;
        border-radius: 999px;
        background: rgba(229,9,20,0.15);
        border: 1px solid rgba(229,9,20,0.5);
        font-size: 0.82rem;
        color: #ff8a8f;
    }
    .chip-green {
        background: rgba(30,215,96,0.12);
        border: 1px solid rgba(30,215,96,0.5);
        color: #7CFFB2;
    }

    /* ---------- Section headers ---------- */
    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.9rem;
        letter-spacing: 2px;
        border-left: 5px solid #E50914;
        padding-left: 12px;
        margin: 6px 0 14px 0;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #E50914, #b0060c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.25s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 18px rgba(229,9,20,0.6);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px 10px;
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); }

    /* progress shimmer bar */
    .shimmer-bar {
        height: 10px;
        border-radius: 6px;
        background: linear-gradient(90deg, #2a2a2a 25%, #3d3d3d 37%, #2a2a2a 63%);
        background-size: 400px 100%;
        animation: shimmer 1.4s infinite linear;
    }

    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# LOAD MODEL ARTIFACTS
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open("netflix_churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, features, scaler


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("netflix_customer_churn.csv")


try:
    model, FEATURES, scaler = load_artifacts()
    ARTIFACTS_OK = True
except Exception as e:  # noqa: BLE001
    ARTIFACTS_OK = False
    ARTIFACT_ERROR = str(e)

try:
    df_raw = load_data()
    DATA_OK = True
except Exception:
    DATA_OK = False
    df_raw = pd.DataFrame()

# ============================================================================
# ENCODING MAPS (must match the label encoding used at training time)
# ============================================================================
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
SUB_MAP = {"Basic": 0, "Premium": 1, "Standard": 2}
REGION_MAP = {"Africa": 0, "Asia": 1, "Europe": 2, "North America": 3, "Oceania": 4, "South America": 5}
DEVICE_MAP = {"Desktop": 0, "Laptop": 1, "Mobile": 2, "TV": 3, "Tablet": 4}
PAYMENT_MAP = {"Credit Card": 0, "Crypto": 1, "Debit Card": 2, "Gift Card": 3, "PayPal": 4}
GENRE_MAP = {"Action": 0, "Comedy": 1, "Documentary": 2, "Drama": 3, "Horror": 4, "Romance": 5, "Sci-Fi": 6}
FEE_MAP = {"Basic": 8.99, "Standard": 13.99, "Premium": 17.99}


def encode_customer_ref(ref: str) -> int:
    """Hash an arbitrary customer reference into the 0-4999 id space the
    scaler was fit on. Falls back to the dataset median if left blank."""
    if not ref:
        return 2499
    h = int(hashlib.md5(ref.encode()).hexdigest(), 16)
    return h % 5000


def build_row(inputs: dict) -> pd.DataFrame:
    row = {
        "customer_id": encode_customer_ref(inputs.get("customer_ref", "")),
        "age": inputs["age"],
        "gender": GENDER_MAP[inputs["gender"]],
        "subscription_type": SUB_MAP[inputs["subscription_type"]],
        "watch_hours": inputs["watch_hours"],
        "last_login_days": inputs["last_login_days"],
        "region": REGION_MAP[inputs["region"]],
        "device": DEVICE_MAP[inputs["device"]],
        "monthly_fee": inputs["monthly_fee"],
        "payment_method": PAYMENT_MAP[inputs["payment_method"]],
        "number_of_profiles": inputs["number_of_profiles"],
        "avg_watch_time_per_day": inputs["avg_watch_time_per_day"],
        "favorite_genre": GENRE_MAP[inputs["favorite_genre"]],
    }
    return pd.DataFrame([row])[FEATURES]


def predict_single(inputs: dict):
    df = build_row(inputs)
    scaled = scaler.transform(df)
    proba = float(model.predict_proba(scaled)[0][1])
    pred = int(model.predict(scaled)[0])
    return pred, proba


def risk_bucket(proba: float):
    if proba >= 0.66:
        return "HIGH", "risk-card-high", "🚨"
    elif proba >= 0.4:
        return "MEDIUM", "risk-card-medium", "⚠️"
    else:
        return "LOW", "risk-card-low", "✅"


def make_gauge(proba: float):
    color = "#E50914" if proba >= 0.66 else ("#FFB020" if proba >= 0.4 else "#1ED760")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%", "font": {"size": 44, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#888", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(30,215,96,0.18)"},
                    {"range": [40, 66], "color": "rgba(255,176,32,0.18)"},
                    {"range": [66, 100], "color": "rgba(229,9,20,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": round(proba * 100, 1),
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eee"},
        margin=dict(t=30, b=10, l=30, r=30),
        height=320,
    )
    return fig


# ============================================================================
# SESSION STATE
# ============================================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown(
    """
    <div class="hero fade-in">
        <p class="hero-title">🎬 CHURNFLIX</p>
        <p class="hero-sub">AI-powered subscriber churn prediction studio</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not ARTIFACTS_OK:
    st.error(
        f"Couldn't load model artifacts (netflix_churn_model.pkl / features.pkl / "
        f"scaler.pkl). Make sure they sit next to app.py.\n\nDetails: {ARTIFACT_ERROR}"
    )
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR — Customer profile inputs
# ============================================================================
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Bebas Neue; letter-spacing:2px; color:#E50914;'>"
        "👤 CUSTOMER PROFILE</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Tune the sliders to describe a subscriber, then hit **Predict**.")

    with st.expander("🧬 Demographics", expanded=True):
        age = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        region = st.selectbox("Region", list(REGION_MAP.keys()))

    with st.expander("💳 Subscription & Billing", expanded=True):
        subscription_type = st.selectbox("Plan", list(SUB_MAP.keys()), index=2)
        monthly_fee = FEE_MAP[subscription_type]
        st.metric("Monthly Fee", f"${monthly_fee:.2f}")
        payment_method = st.selectbox("Payment Method", list(PAYMENT_MAP.keys()))
        number_of_profiles = st.slider("Number of Profiles", 1, 5, 3)

    with st.expander("📺 Viewing Behavior", expanded=True):
        watch_hours = st.slider("Total Watch Hours (recent period)", 0.0, 120.0, 12.0, 0.5)
        avg_watch_time_per_day = st.slider("Avg. Watch Time / Day (hrs)", 0.0, 15.0, 1.0, 0.1)
        last_login_days = st.slider("Days Since Last Login", 0, 60, 5)
        device = st.selectbox("Primary Device", list(DEVICE_MAP.keys()))
        favorite_genre = st.selectbox("Favorite Genre", list(GENRE_MAP.keys()))

    with st.expander("🧾 Advanced (optional)"):
        customer_ref = st.text_input("Customer reference / ID", "")
        st.caption("Purely for tracking — hashed internally, doesn't drive the score meaningfully.")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 PREDICT CHURN RISK", use_container_width=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Predict", "📊 Batch Prediction", "📈 Insights Dashboard", "ℹ️ About the Model"]
)

# ----------------------------------------------------------------------------
# TAB 1 — SINGLE PREDICTION
# ----------------------------------------------------------------------------
with tab1:
    inputs = dict(
        age=age,
        gender=gender,
        subscription_type=subscription_type,
        watch_hours=watch_hours,
        last_login_days=last_login_days,
        region=region,
        device=device,
        monthly_fee=monthly_fee,
        payment_method=payment_method,
        number_of_profiles=number_of_profiles,
        avg_watch_time_per_day=avg_watch_time_per_day,
        favorite_genre=favorite_genre,
        customer_ref=customer_ref,
    )

    if predict_clicked:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown('<div class="shimmer-bar"></div>', unsafe_allow_html=True)
            steps = ["🔎 Reading viewing history...", "🧮 Crunching engagement signals...", "🎯 Scoring churn risk..."]
            prog = st.progress(0, text=steps[0])
            for i, s in enumerate(steps):
                time.sleep(0.28)
                prog.progress(int((i + 1) / len(steps) * 100), text=s)
            time.sleep(0.15)
        placeholder.empty()

        pred, proba = predict_single(inputs)
        label, css_class, emoji = risk_bucket(proba)

        st.session_state.history.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "subscription": subscription_type,
                "region": region,
                "probability": round(proba * 100, 1),
                "risk": label,
            },
        )

        col_left, col_right = st.columns([1, 1.15])

        with col_left:
            st.markdown(
                f"""
                <div class="{css_class} fade-in">
                    <div class="risk-emoji">{emoji}</div>
                    <p class="risk-label">{label} RISK</p>
                    <p style="color:#ddd; margin-top:-6px;">
                        Estimated churn probability: <b>{proba*100:.1f}%</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if label == "LOW":
                st.balloons()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-title" style="font-size:1.2rem;">💡 Suggested Action</p>', unsafe_allow_html=True)

            tips = []
            if last_login_days >= 14:
                tips.append("Customer hasn't logged in recently — trigger a **re-engagement email** with personalized picks.")
            if avg_watch_time_per_day < 0.3:
                tips.append("Very low daily watch time — surface a **'trending now'** row or push notification.")
            if subscription_type == "Premium" and avg_watch_time_per_day < 1:
                tips.append("Paying premium but underusing it — consider a **plan-right-sizing offer** or loyalty perk.")
            if number_of_profiles <= 1:
                tips.append("Single-profile account — promote **family/profile sharing** to boost stickiness.")
            if not tips:
                tips.append("Engagement signals look healthy — keep recommending fresh content in their **favorite genre**.")

            for t in tips:
                st.markdown(f'<div class="chip {"chip-green" if label=="LOW" else ""}">🎬 {t}</div>', unsafe_allow_html=True)

        with col_right:
            st.plotly_chart(make_gauge(proba), use_container_width=True, config={"displayModeBar": False})

            c1, c2, c3 = st.columns(3)
            c1.metric("Watch Hrs", f"{watch_hours:.1f}")
            c2.metric("Avg/Day (hrs)", f"{avg_watch_time_per_day:.2f}")
            c3.metric("Last Login", f"{last_login_days}d ago")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="font-size:1.2rem;">🧠 What drove this score?</p>', unsafe_allow_html=True)

        imp_df = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=True
        )
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#3d0a0c", "#E50914"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eee"},
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    else:
        st.info("👈 Set the customer profile in the sidebar, then click **Predict Churn Risk** to see the magic happen.")

    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="font-size:1.2rem;">🕘 Session History</p>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            hist_df.style.background_gradient(subset=["probability"], cmap="Reds"),
            use_container_width=True,
            hide_index=True,
        )

# ----------------------------------------------------------------------------
# TAB 2 — BATCH PREDICTION
# ----------------------------------------------------------------------------
with tab2:
    st.markdown('<p class="section-title">📊 Batch Prediction from CSV</p>', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV with the following columns (raw, human-readable values — "
        "no need to pre-encode anything):"
    )
    st.code(
        "customer_id, age, gender, subscription_type, watch_hours, last_login_days,\n"
        "region, device, monthly_fee, payment_method, number_of_profiles,\n"
        "avg_watch_time_per_day, favorite_genre",
        language="text",
    )

    sample_csv = df_raw.drop(columns=["churned"]).head(5).to_csv(index=False) if DATA_OK else ""
    if sample_csv:
        st.download_button("⬇️ Download sample template", sample_csv, file_name="churn_template.csv", use_container_width=False)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = [c for c in FEATURES]
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                with st.spinner("🎞️ Scoring every subscriber..."):
                    work = batch_df.copy()
                    work["gender"] = work["gender"].map(GENDER_MAP)
                    work["subscription_type"] = work["subscription_type"].map(SUB_MAP)
                    work["region"] = work["region"].map(REGION_MAP)
                    work["device"] = work["device"].map(DEVICE_MAP)
                    work["payment_method"] = work["payment_method"].map(PAYMENT_MAP)
                    work["favorite_genre"] = work["favorite_genre"].map(GENRE_MAP)
                    work["customer_id"] = work["customer_id"].astype(str).apply(encode_customer_ref)

                    if work[FEATURES].isna().any().any():
                        st.warning(
                            "Some categorical values didn't match known categories and were skipped. "
                            "Double-check spelling/casing against the template."
                        )
                        work = work.dropna(subset=FEATURES)
                        batch_df = batch_df.loc[work.index]

                    scaled = scaler.transform(work[FEATURES])
                    probs = model.predict_proba(scaled)[:, 1]
                    preds = model.predict(scaled)

                    time.sleep(0.3)

                result_df = batch_df.copy()
                result_df["churn_probability_%"] = np.round(probs * 100, 1)
                result_df["churn_prediction"] = np.where(preds == 1, "Churn", "Stay")

                st.success(f"Scored {len(result_df)} customers ✅")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Customers", len(result_df))
                m2.metric("Predicted Churners", int((preds == 1).sum()))
                m3.metric("Churn Rate", f"{(preds == 1).mean()*100:.1f}%")
                m4.metric("Avg. Risk Score", f"{probs.mean()*100:.1f}%")

                colA, colB = st.columns(2)
                with colA:
                    fig_hist = px.histogram(
                        result_df, x="churn_probability_%", nbins=25, color_discrete_sequence=["#E50914"]
                    )
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#eee"}, title="Distribution of Churn Probability",
                        margin=dict(t=40, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                with colB:
                    if "subscription_type" in batch_df.columns:
                        sub_summary = result_df.groupby(batch_df["subscription_type"])["churn_prediction"].apply(
                            lambda s: (s == "Churn").mean() * 100
                        ).reset_index(name="churn_rate_%")
                        fig_sub = px.bar(
                            sub_summary, x="subscription_type", y="churn_rate_%",
                            color="churn_rate_%", color_continuous_scale=["#1ED760", "#FFB020", "#E50914"],
                        )
                        fig_sub.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#eee"}, title="Churn Rate by Plan",
                            coloraxis_showscale=False, margin=dict(t=40, b=10, l=10, r=10),
                        )
                        st.plotly_chart(fig_sub, use_container_width=True, config={"displayModeBar": False})

                st.markdown('<p class="section-title" style="font-size:1.2rem;">📋 Scored Results</p>', unsafe_allow_html=True)
                st.dataframe(
                    result_df.style.background_gradient(subset=["churn_probability_%"], cmap="Reds"),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    result_df.to_csv(index=False),
                    file_name="churn_predictions.csv",
                    use_container_width=True,
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't process that file: {e}")
    else:
        st.markdown(
            '<div class="glass-card fade-in">Upload a CSV to score many customers at once — '
            "great for weekly retention reviews. 🍿</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# TAB 3 — INSIGHTS DASHBOARD
# ----------------------------------------------------------------------------
with tab3:
    if not DATA_OK:
        st.warning("netflix_customer_churn.csv not found next to app.py — dashboard unavailable.")
    else:
        st.markdown('<p class="section-title">📈 Subscriber Insights Dashboard</p>', unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            region_filter = st.multiselect("Filter by Region", sorted(df_raw["region"].unique()), default=None)
        with f2:
            sub_filter = st.multiselect("Filter by Plan", sorted(df_raw["subscription_type"].unique()), default=None)

        view = df_raw.copy()
        if region_filter:
            view = view[view["region"].isin(region_filter)]
        if sub_filter:
            view = view[view["subscription_type"].isin(sub_filter)]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{len(view):,}")
        k2.metric("Churn Rate", f"{view['churned'].mean()*100:.1f}%")
        k3.metric("Avg Watch Hrs", f"{view['watch_hours'].mean():.1f}")
        k4.metric("Avg Monthly Fee", f"${view['monthly_fee'].mean():.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        row1c1, row1c2 = st.columns(2)

        with row1c1:
            sub_churn = view.groupby("subscription_type")["churned"].mean().reset_index()
            sub_churn["churned"] *= 100
            fig1 = px.bar(sub_churn, x="subscription_type", y="churned", color="churned",
                          color_continuous_scale=["#1ED760", "#FFB020", "#E50914"], title="Churn Rate by Plan (%)")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": "#eee"}, coloraxis_showscale=False, margin=dict(t=40))
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

        with row1c2:
            region_churn = view.groupby("region")["churned"].mean().reset_index()
            region_churn["churned"] *= 100
            fig2 = px.bar(region_churn, x="region", y="churned", color="churned",
                          color_continuous_scale=["#1ED760", "#FFB020", "#E50914"], title="Churn Rate by Region (%)")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": "#eee"}, coloraxis_showscale=False, margin=dict(t=40))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        row2c1, row2c2 = st.columns(2)
        with row2c1:
            fig3 = px.box(view, x="churned", y="watch_hours", color="churned",
                         color_discrete_map={0: "#1ED760", 1: "#E50914"},
                         title="Watch Hours vs Churn")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": "#eee"}, margin=dict(t=40), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        with row2c2:
            device_churn = view.groupby("device")["churned"].mean().reset_index()
            device_churn["churned"] *= 100
            fig4 = px.bar(device_churn, x="device", y="churned", color="churned",
                          color_continuous_scale=["#1ED760", "#FFB020", "#E50914"], title="Churn Rate by Device (%)")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": "#eee"}, coloraxis_showscale=False, margin=dict(t=40))
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title" style="font-size:1.2rem;">🔥 Correlation Heatmap</p>', unsafe_allow_html=True)
        num_cols = ["age", "watch_hours", "last_login_days", "monthly_fee", "number_of_profiles",
                    "avg_watch_time_per_day", "churned"]
        corr = view[num_cols].corr()
        fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#0d0d0d", "#E50914"], aspect="auto")
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#eee"}, margin=dict(t=10))
        st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# TAB 4 — ABOUT
# ----------------------------------------------------------------------------
with tab4:
    st.markdown('<p class="section-title">ℹ️ About This Model</p>', unsafe_allow_html=True)

    colA, colB = st.columns([1.3, 1])
    with colA:
        st.markdown(
            """
            <div class="glass-card fade-in">
            <b>Model:</b> Random Forest Classifier (150 trees)<br>
            <b>Task:</b> Binary classification — predict whether a subscriber will churn<br>
            <b>Features used:</b> demographics, subscription & billing details, and
            viewing-behavior signals (watch hours, login recency, daily usage, device, genre)<br><br>
            The strongest predictors are <b>daily watch time</b>, <b>total watch hours</b>, and
            <b>days since last login</b> — classic engagement-recency-frequency signals.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <span class="chip">🐍 Python</span>
            <span class="chip">🎈 Streamlit</span>
            <span class="chip">🌲 scikit-learn</span>
            <span class="chip">📊 Plotly</span>
            <span class="chip">🐼 pandas</span>
            """,
            unsafe_allow_html=True,
        )

    with colB:
        imp_df = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=True
        )
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#3d0a0c", "#E50914"],
                     title="Global Feature Importance")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#eee"}, coloraxis_showscale=False, margin=dict(t=40), height=420)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Built with ❤️ using Streamlit · Predictions are estimates, not guarantees — always pair with human judgment.")
