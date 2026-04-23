import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             brier_score_loss)
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Diabetes Early Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body, .main, .block-container {
        background-color: #0d1117 !important;
        color: #e6edf3;
    }
    .block-container { padding: 1.5rem 2rem; }
    .kpi-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        height: 100%;
    }
    .kpi-label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #e6edf3;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1;
    }
    .kpi-sub {
        color: #8b949e;
        font-size: 0.75rem;
        margin-top: 4px;
    }
    .chart-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .chart-title {
        color: #e6edf3;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .warn-box {
        background: #1f1116;
        border-left: 4px solid #f85149;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
        color: #e6edf3;
        font-size: 0.88rem;
    }
    .ok-box {
        background: #0f1f0f;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
        color: #e6edf3;
        font-size: 0.88rem;
    }
    .info-box {
        background: #111a2a;
        border-left: 4px solid #388bfd;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
        color: #e6edf3;
        font-size: 0.88rem;
    }
    .amber-box {
        background: #1a1500;
        border-left: 4px solid #d29922;
        border-radius: 6px;
        padding: 14px 16px;
        margin-top: 10px;
        color: #e6edf3;
        font-size: 0.88rem;
    }
    .section-label {
        color: #8b949e;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 6px;
    }
    .patient-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        background: transparent;
        border-radius: 6px;
        font-size: 0.85rem;
        padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #e6edf3 !important;
    }
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.75rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-size: 1.4rem !important;
    }
    .stButton button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 10px;
        width: 100%;
    }
    .stButton button:hover { background: #2ea043; }
    .stSlider label, .stRadio label, .stNumberInput label {
        color: #c9d1d9 !important;
        font-size: 0.88rem !important;
    }
    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

BG     = '#161b22'
BG2    = '#0d1117'
GRID   = '#21262d'
TEXT   = '#e6edf3'
MUTED  = '#8b949e'
GREEN  = '#3fb950'
AMBER  = '#d29922'
RED    = '#f85149'
BLUE   = '#388bfd'
PURPLE = '#c084fc'
THRESHOLD = 0.15

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

@st.cache_resource
def load_and_train():
    df = pd.read_csv("nhanes_diabetes_clean.csv")
    features = ['age', 'sex', 'bmi', 'sbp', 'dbp']
    X = df[features]
    y = df['diabetes'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb_default = XGBClassifier(
        n_estimators=100, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    xgb_default.fit(X_train, y_train)

    xgb_tuned = XGBClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.01, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        eval_metric='logloss', verbosity=0
    )
    xgb_tuned.fit(X_train, y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        solver='adam', max_iter=1000,
        early_stopping=True, validation_fraction=0.1,
        random_state=42, verbose=False
    )
    mlp.fit(X_train_sc, y_train)

    xgb_platt = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=300, max_depth=3,
            learning_rate=0.01, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        method='sigmoid', cv=5
    )
    xgb_platt.fit(X_train, y_train)

    xgb_iso = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=300, max_depth=3,
            learning_rate=0.01, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
        method='isotonic', cv=5
    )
    xgb_iso.fit(X_train, y_train)

    return (df, scaler, lr, rf, xgb_default, xgb_tuned,
            mlp, xgb_platt, xgb_iso,
            X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc, features)

(df, scaler, lr, rf, xgb_default, xgb_tuned,
 mlp, xgb_platt, xgb_iso,
 X_train, X_test, y_train, y_test,
 X_train_sc, X_test_sc, features) = load_and_train()

@st.cache_resource
def get_shap():
    X_np      = np.array(X_test)
    explainer = shap.TreeExplainer(xgb_tuned)
    sv        = explainer.shap_values(X_np)
    return explainer, sv, X_np

shap_explainer, shap_vals, X_test_np = get_shap()
mean_shap = np.abs(shap_vals).mean(axis=0)

@st.cache_resource
def get_lime():
    return lime.lime_tabular.LimeTabularExplainer(
        training_data         = np.array(X_train),
        feature_names         = features,
        class_names           = ['No Diabetes', 'Diabetes'],
        mode                  = 'classification',
        discretize_continuous = True,
        random_state          = 42
    )

lime_explainer = get_lime()

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum()/n) * abs(
            y_true[mask].mean() - y_prob[mask].mean()
        )
    return ece

def clinical_rule_check(a1c, glucose, ml_prob):
    decision = "High Risk" if ml_prob >= THRESHOLD else "Low Risk"
    flags, reasons = False, []
    if a1c and a1c >= 6.5:
        flags = True
        reasons.append(f"A1C {a1c}% meets ADA threshold of 6.5%")
    if glucose and glucose >= 126:
        flags = True
        reasons.append(f"Glucose {glucose} mg/dL meets ADA threshold of 126")
    if flags and decision == "Low Risk":
        return "contradiction", reasons
    elif flags and decision == "High Risk":
        return "agree_high", reasons
    elif not flags and decision == "High Risk":
        return "model_flag", reasons
    return "agree_low", reasons

st.markdown("""
<div style="background:linear-gradient(135deg,#1a1f35 0%,#0d1117 100%);
            border-bottom:1px solid #21262d;padding:16px 0 12px 0;
            margin-bottom:20px;">
    <h1 style="color:#e6edf3;font-size:1.55rem;font-weight:700;margin:0;">
        Early Detection of Type 2 Diabetes
    </h1>
    <p style="color:#8b949e;font-size:0.82rem;margin:3px 0 0 0;">
        NHANES 2015-2018 &nbsp;|&nbsp; Nottingham Trent University &nbsp;|&nbsp;
        Tuned XGBoost &nbsp;|&nbsp; SHAP &nbsp;|&nbsp; LIME &nbsp;|&nbsp;
        Clinical Rules &nbsp;|&nbsp; FYP 2026
    </p>
</div>
""", unsafe_allow_html=True)

(tab0, tab1, tab2, tab3,
 tab4, tab5, tab6, tab7) = st.tabs([
    "🏠 Home",
    "🔍 Screening",
    "📊 Model Results",
    "⚖️ Calibration",
    "🎯 Threshold",
    "🔁 Robustness",
    "🧠 Explainability",
    "🏥 Clinical Rules"
])

# ── TAB 0 HOME ────────────────────────────────────────────────
with tab0:

    st.markdown(f"""
    <div class="info-box" style="margin-bottom:20px;">
        👋 <strong>Welcome.</strong> This dashboard presents an end-to-end
        machine learning pipeline for early Type 2 diabetes screening,
        built using real health survey data from <strong>10,168 US adults</strong>.
        Use the Screening tab to assess individual patient risk or explore
        the remaining tabs to understand the methodology and results.
    </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        ("👥", "Total Patients",  f"{len(df):,}",
         "NHANES 2015-2018", TEXT),
        ("🩸", "Diabetic Cases",  f"{int(df['diabetes'].sum()):,}",
         "ADA confirmed labels", RED),
        ("📊", "Prevalence",      f"{df['diabetes'].mean()*100:.1f}%",
         "Class imbalance present", AMBER),
        ("🎂", "Mean Age",        f"{df['age'].mean():.1f}",
         f"Std {df['age'].std():.1f} years", GREEN),
        ("⚖️", "Mean BMI",        f"{df['bmi'].mean():.1f}",
         f"Std {df['bmi'].std():.1f}", BLUE),
    ]
    for col, (icon, label, val, sub, color) in zip(
        [k1, k2, k3, k4, k5], kpi_data
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="font-size:1.6rem;">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{color};">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        stats = df[['age','bmi','sbp','dbp','waist']].agg(
            ['mean','median','std','min','max']
        ).T.round(2)
        stats.columns = ['Mean','Median','Std','Min','Max']
        stats.index   = ['Age','BMI','Systolic BP','Diastolic BP','Waist']
        st.dataframe(stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Class Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        counts = df['diabetes'].value_counts().sort_index()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels     = ['No Diabetes','Diabetes'],
            colors     = [GREEN, RED],
            autopct    = '%1.1f%%',
            startangle = 90,
            wedgeprops = {'edgecolor': BG2, 'linewidth': 2}
        )
        for t in texts:     t.set_color(TEXT)
        for t in autotexts: t.set_color(TEXT); t.set_fontsize(11)
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-title">Feature Distributions</div>',
                unsafe_allow_html=True)
    st.caption("White dashed line shows the mean. "
               "Yellow dotted line shows the median.")
    cols_p = ['age','bmi','sbp','dbp','waist']
    ttls_p = ['Age (years)','BMI','Systolic BP (mmHg)',
              'Diastolic BP (mmHg)','Waist (cm)']
    clrs_p = [GREEN, AMBER, RED, BLUE, PURPLE]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    fig.patch.set_facecolor(BG2)
    for ax, col, title, color in zip(axes, cols_p, ttls_p, clrs_p):
        style_ax(ax)
        ax.hist(df[col].dropna(), bins=28, color=color,
                alpha=0.9, edgecolor='none')
        ax.axvline(df[col].mean(),   color='white',  ls='--',
                   lw=1.2, label=f"Mean {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color='yellow', ls=':',
                   lw=1.2, label=f"Med {df[col].median():.1f}")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-title">Feature Distributions by Diabetes Status</div>',
                unsafe_allow_html=True)
    st.caption("Green shows non-diabetic patients. "
               "Red shows diabetic patients. "
               "The difference in medians shows which features separate the groups.")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.patch.set_facecolor(BG2)
    for ax, col, title in zip(axes, cols_p, ttls_p):
        style_ax(ax)
        d0 = df[df['diabetes']==0][col].dropna()
        d1 = df[df['diabetes']==1][col].dropna()
        bp = ax.boxplot(
            [d0, d1],
            labels       = ['No\nDiabetes','Diabetes'],
            patch_artist = True,
            medianprops  = dict(color='white', linewidth=2),
            whiskerprops = dict(color=MUTED),
            capprops     = dict(color=MUTED),
            flierprops   = dict(marker='o', color=MUTED,
                                markersize=2, alpha=0.4)
        )
        bp['boxes'][0].set_facecolor('#1a3a1a')
        bp['boxes'][1].set_facecolor('#3a1a1a')
        bp['boxes'][0].set_edgecolor(GREEN)
        bp['boxes'][1].set_edgecolor(RED)
        ax.set_title(title, fontsize=9)
        diff = d1.mean() - d0.mean()
        ax.set_xlabel(f'Mean difference {diff:+.1f}', fontsize=8, color=MUTED)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-title">Feature Correlation Matrix</div>',
                unsafe_allow_html=True)
    st.caption("Pearson correlation coefficients. "
               "Darker green means stronger positive correlation with diabetes.")
    corr_cols = ['age','sex','bmi','sbp','dbp','waist','diabetes']
    corr_df   = df[corr_cols].copy().astype(float)
    corr_mat  = corr_df.corr()
    fig, ax   = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat, mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', center=0, square=True, ax=ax,
        linewidths=0.5,
        annot_kws={'size': 9, 'color': TEXT},
        cbar_kws={'shrink': 0.8}
    )
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title('Correlation Matrix', color=TEXT, fontsize=11)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Missing Values</div>',
                    unsafe_allow_html=True)
        miss = pd.DataFrame({
            'Column':      df.columns,
            'Missing (n)': df.isnull().sum().values,
            'Missing (%)': (df.isnull().sum().values/len(df)*100).round(1),
            'Status':      ['Complete' if x==0 else 'Has missing'
                            for x in df.isnull().sum().values]
        })
        st.dataframe(miss, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Dashboard Navigation</div>',
                    unsafe_allow_html=True)
        nav_items = [
            ("🔍 Screening",
             "Enter patient details for a personal risk score"),
            ("📊 Model Results",
             "Compare all five models with full performance metrics"),
            ("⚖️ Calibration",
             "Verify probability reliability using Brier score and ECE"),
            ("🎯 Threshold",
             "See why threshold 0.50 fails and how 0.15 was chosen"),
            ("🔁 Robustness",
             "Results validated across ten random seeds"),
            ("🧠 Explainability",
             "SHAP and LIME show why the model made each prediction"),
            ("🏥 Clinical Rules",
             "Contradiction detection catches cases the model misses"),
        ]
        for icon_label, desc in nav_items:
            st.markdown(f"""
            <div style="padding:8px 0;border-bottom:1px solid {GRID};">
                <span style="color:{TEXT};font-weight:600;
                             font-size:0.88rem;">{icon_label}</span>
                <span style="color:{MUTED};font-size:0.82rem;
                             margin-left:8px;">{desc}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 1 SCREENING ───────────────────────────────────────────
with tab1:
    st.markdown("### Patient Risk Screening")
    st.caption("Enter non-invasive measurements only. "
               "No blood test is required for the primary risk assessment.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown('<p class="section-label">Patient Measurements</p>',
                    unsafe_allow_html=True)
        age     = st.slider("Age (years)", 18, 90, 50)
        sex     = st.radio("Sex", ["Male","Female"], horizontal=True)
        sex_val = 1 if sex == "Male" else 2
        bmi_v   = st.slider("BMI", 15.0, 60.0, 27.0, 0.1)
        sbp_v   = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 125)
        dbp_v   = st.slider("Diastolic Blood Pressure (mmHg)", 40, 130, 75)

        st.markdown("---")
        st.markdown('<p class="section-label">Optional Lab Values</p>',
                    unsafe_allow_html=True)
        st.caption("Used only for the clinical rule contradiction check")
        a1c_v   = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, 0.1)
        glc_v   = st.number_input("Fasting Glucose (mg/dL)",
                                   0.0, 500.0, 0.0, 1.0)
        run_btn = st.button("Run Screening", use_container_width=True)

    with col2:
        if run_btn:
            inp  = np.array([[age, sex_val, bmi_v, sbp_v, dbp_v]])
            prob = xgb_tuned.predict_proba(inp)[0][1]
            pct  = prob * 100

            if prob < 0.15:
                level   = "LOW RISK"
                r_color = GREEN
                advice  = ("No immediate action required. "
                           "Maintaining a healthy lifestyle is recommended.")
            elif prob < 0.35:
                level   = "MEDIUM RISK"
                r_color = AMBER
                advice  = ("Lifestyle review is advised. "
                           "A follow-up with a GP is recommended.")
            else:
                level   = "HIGH RISK"
                r_color = RED
                advice  = ("Please consult a healthcare professional "
                           "as soon as possible.")

            st.markdown('<p class="section-label">Risk Assessment</p>',
                        unsafe_allow_html=True)

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': '%', 'font': {'size': 42, 'color': TEXT}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': MUTED,
                             'tickfont': {'color': MUTED}},
                    'bar': {'color': r_color, 'thickness': 0.7},
                    'bgcolor': '#21262d',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 15], 'color': '#0f1f0f'},
                        {'range': [15, 35], 'color': '#1a1500'},
                        {'range': [35, 100], 'color': '#1f1116'}
                    ],
                    'threshold': {
                        'line': {'color': AMBER, 'width': 3},
                        'thickness': 0.8,
                        'value': 15
                    }
                },
                title={'text': f"<b>{level}</b>",
                       'font': {'size': 20, 'color': r_color}}
            ))
            gauge_fig.update_layout(
                paper_bgcolor=BG, font_color=TEXT,
                height=280, margin=dict(t=80, b=20, l=30, r=30)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown(f"""
            <div class="kpi-card" style="margin-top:10px;">
                <div style="color:{MUTED};font-size:0.85rem;margin-top:4px;">
                    Predicted probability {pct:.1f}%
                    at screening threshold {THRESHOLD*100:.0f}%
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Patient Profile</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="patient-card">
                <div style="display:grid;
                             grid-template-columns:repeat(3,1fr);
                             gap:10px;text-align:center;">
                    <div>
                        <div class="kpi-label">Age</div>
                        <div style="color:{TEXT};font-weight:700;">
                            {age} yrs</div>
                    </div>
                    <div>
                        <div class="kpi-label">Sex</div>
                        <div style="color:{TEXT};font-weight:700;">
                            {sex}</div>
                    </div>
                    <div>
                        <div class="kpi-label">BMI</div>
                        <div style="color:{TEXT};font-weight:700;">
                            {bmi_v:.1f}</div>
                    </div>
                    <div>
                        <div class="kpi-label">Systolic BP</div>
                        <div style="color:{TEXT};font-weight:700;">
                            {sbp_v} mmHg</div>
                    </div>
                    <div>
                        <div class="kpi-label">Diastolic BP</div>
                        <div style="color:{TEXT};font-weight:700;">
                            {dbp_v} mmHg</div>
                    </div>
                    <div>
                        <div class="kpi-label">Model</div>
                        <div style="color:{BLUE};font-weight:700;">
                            XGBoost</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<p class="section-label">Feature Contributions</p>',
                        unsafe_allow_html=True)
            pt_shap = shap.TreeExplainer(xgb_tuned).shap_values(inp)[0]
            feat_display = ['Age','Sex','BMI','Systolic BP','Diastolic BP']
            s_idx  = np.argsort(np.abs(pt_shap))
            s_feat = [feat_display[i] for i in s_idx]
            s_vals = pt_shap[s_idx]
            s_data = inp[0][s_idx]

            shap_fig = go.Figure(go.Bar(
                x=s_vals,
                y=[f"{f} = {s_data[i]:.1f}" for i, f in enumerate(s_feat)],
                orientation='h',
                marker_color=[RED if v > 0 else GREEN for v in s_vals],
                text=[f'{"▲" if v>0 else "▼"} {abs(v):.3f}' for v in s_vals],
                textposition='outside',
                textfont=dict(size=11, color=TEXT),
                hovertemplate='%{y}<br>SHAP value: %{x:.4f}<extra></extra>'
            ))
            shap_fig.add_vline(x=0, line_color=MUTED, line_width=1)
            shap_fig.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font_color=TEXT, height=220,
                margin=dict(t=10, b=40, l=10, r=10),
                xaxis=dict(title='Red increases risk  |  Green decreases risk',
                           gridcolor=GRID, zeroline=False, title_font_size=10),
                yaxis=dict(gridcolor=GRID)
            )
            st.plotly_chart(shap_fig, use_container_width=True)

            st.markdown(
                f'<div class="info-box">'
                f'<strong>Recommended Action:</strong> {advice}'
                f'</div>',
                unsafe_allow_html=True
            )

            report_text = f"""DIABETES SCREENING REPORT
Generated: {datetime.now().strftime('%d %B %Y at %H:%M')}
Model: XGBoost Tuned (GridSearchCV, AUC 0.766)
Threshold: {THRESHOLD*100:.0f}%
{'='*50}

PATIENT PROFILE
  Age:             {age} years
  Sex:             {sex}
  BMI:             {bmi_v:.1f}
  Systolic BP:     {sbp_v} mmHg
  Diastolic BP:    {dbp_v} mmHg

RISK ASSESSMENT
  Predicted Risk:  {pct:.1f}%
  Classification:  {level}
  Action:          {advice}

FEATURE CONTRIBUTIONS (SHAP)
"""
            for i, f in enumerate(s_feat):
                direction = "increases" if s_vals[i] > 0 else "decreases"
                report_text += f"  {f} = {s_data[i]:.1f}  ->  {direction} risk by {abs(s_vals[i]):.4f}\n"

            if a1c_v > 0 or glc_v > 0:
                report_text += f"\nCLINICAL LAB VALUES\n"
                if a1c_v > 0: report_text += f"  HbA1c:   {a1c_v}%\n"
                if glc_v > 0: report_text += f"  Glucose:  {glc_v} mg/dL\n"

            report_text += f"""
{'='*50}
DISCLAIMER: This is a screening tool only and does not
constitute a clinical diagnosis. Always consult a qualified
healthcare professional for medical advice.

System: Early Detection of T2DM | NHANES 2015-2018
Nottingham Trent University | BSc Computing FYP 2026
"""
            st.download_button(
                label="Download Screening Report",
                data=report_text,
                file_name=f"diabetes_screening_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

            a1c_in = a1c_v  if a1c_v  > 0 else None
            glc_in = glc_v  if glc_v  > 0 else None
            if a1c_in or glc_in:
                status, reasons = clinical_rule_check(
                    a1c_in, glc_in, prob
                )
                st.markdown(
                    '<p class="section-label" '
                    'style="margin-top:12px;">Clinical Rule Check</p>',
                    unsafe_allow_html=True
                )
                if status == "contradiction":
                    st.markdown(f"""
                    <div class="warn-box">
                        <strong>CONTRADICTION WARNING</strong><br>
                        The model predicts Low Risk but lab values
                        meet ADA diagnostic criteria for diabetes.<br>
                        {' '.join(reasons)}<br>
                        Clinical values take precedence.
                        Refer for full diagnostic evaluation.
                    </div>""", unsafe_allow_html=True)
                elif status == "agree_high":
                    st.markdown(f"""
                    <div class="warn-box">
                        <strong>HIGH RISK CONFIRMED</strong><br>
                        Model prediction and clinical values agree.
                        {' '.join(reasons)}
                    </div>""", unsafe_allow_html=True)
                elif status == "model_flag":
                    st.markdown(f"""
                    <div class="amber-box">
                        <strong>MODEL FLAG</strong><br>
                        Model predicts elevated risk but lab values
                        are within normal range. Close monitoring
                        is recommended.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ok-box">
                        Model prediction and clinical values agree.
                        Low risk confirmed.
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-card"
                 style="padding:60px 20px;margin-top:40px;">
                <div style="font-size:2.5rem;">🔍</div>
                <div style="color:{MUTED};font-size:1rem;margin-top:12px;">
                    Enter patient measurements and click
                    <strong style="color:{TEXT};">Run Screening</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warn-box"
             style="margin-top:16px;font-size:0.8rem;">
            For screening purposes only. Does not constitute a clinical
            diagnosis. Always consult a qualified healthcare professional.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Patient Comparison Mode")
    st.caption("Compare two patients side by side to see how different "
               "profiles affect risk and which features drive the difference.")

    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.markdown(f'<div class="patient-card"><div class="chart-title" '
                    f'style="color:{BLUE};">Patient A</div>',
                    unsafe_allow_html=True)
        cmp_age_a = st.slider("Age", 18, 90, 45, key="cmp_a_age")
        cmp_sex_a = st.radio("Sex", ["Male","Female"], horizontal=True, key="cmp_a_sex")
        cmp_bmi_a = st.slider("BMI", 15.0, 60.0, 24.0, 0.1, key="cmp_a_bmi")
        cmp_sbp_a = st.slider("Systolic BP", 80, 200, 120, key="cmp_a_sbp")
        cmp_dbp_a = st.slider("Diastolic BP", 40, 130, 70, key="cmp_a_dbp")
        st.markdown('</div>', unsafe_allow_html=True)

    with cmp_col2:
        st.markdown(f'<div class="patient-card"><div class="chart-title" '
                    f'style="color:{AMBER};">Patient B</div>',
                    unsafe_allow_html=True)
        cmp_age_b = st.slider("Age", 18, 90, 65, key="cmp_b_age")
        cmp_sex_b = st.radio("Sex", ["Male","Female"], horizontal=True, key="cmp_b_sex")
        cmp_bmi_b = st.slider("BMI", 15.0, 60.0, 34.0, 0.1, key="cmp_b_bmi")
        cmp_sbp_b = st.slider("Systolic BP", 80, 200, 145, key="cmp_b_sbp")
        cmp_dbp_b = st.slider("Diastolic BP", 40, 130, 90, key="cmp_b_dbp")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Compare Patients", use_container_width=True, key="cmp_btn"):
        sex_a = 1 if cmp_sex_a == "Male" else 2
        sex_b = 1 if cmp_sex_b == "Male" else 2
        inp_a = np.array([[cmp_age_a, sex_a, cmp_bmi_a, cmp_sbp_a, cmp_dbp_a]])
        inp_b = np.array([[cmp_age_b, sex_b, cmp_bmi_b, cmp_sbp_b, cmp_dbp_b]])
        prob_a = xgb_tuned.predict_proba(inp_a)[0][1]
        prob_b = xgb_tuned.predict_proba(inp_b)[0][1]
        pct_a = prob_a * 100
        pct_b = prob_b * 100
        level_a = "LOW" if prob_a < 0.15 else ("MEDIUM" if prob_a < 0.35 else "HIGH")
        level_b = "LOW" if prob_b < 0.15 else ("MEDIUM" if prob_b < 0.35 else "HIGH")
        color_a = GREEN if prob_a < 0.15 else (AMBER if prob_a < 0.35 else RED)
        color_b = GREEN if prob_b < 0.15 else (AMBER if prob_b < 0.35 else RED)

        r_col1, r_col2 = st.columns(2)
        with r_col1:
            ga = go.Figure(go.Indicator(
                mode="gauge+number", value=pct_a,
                number={'suffix': '%', 'font': {'size': 36, 'color': TEXT}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
                       'bar': {'color': color_a, 'thickness': 0.7}, 'bgcolor': '#21262d', 'borderwidth': 0,
                       'steps': [{'range': [0, 15], 'color': '#0f1f0f'}, {'range': [15, 35], 'color': '#1a1500'}, {'range': [35, 100], 'color': '#1f1116'}]},
                title={'text': f"<b>Patient A - {level_a} RISK</b>", 'font': {'size': 16, 'color': color_a}}
            ))
            ga.update_layout(paper_bgcolor=BG, font_color=TEXT, height=250, margin=dict(t=70, b=10, l=20, r=20))
            st.plotly_chart(ga, use_container_width=True)

        with r_col2:
            gb = go.Figure(go.Indicator(
                mode="gauge+number", value=pct_b,
                number={'suffix': '%', 'font': {'size': 36, 'color': TEXT}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
                       'bar': {'color': color_b, 'thickness': 0.7}, 'bgcolor': '#21262d', 'borderwidth': 0,
                       'steps': [{'range': [0, 15], 'color': '#0f1f0f'}, {'range': [15, 35], 'color': '#1a1500'}, {'range': [35, 100], 'color': '#1f1116'}]},
                title={'text': f"<b>Patient B - {level_b} RISK</b>", 'font': {'size': 16, 'color': color_b}}
            ))
            gb.update_layout(paper_bgcolor=BG, font_color=TEXT, height=250, margin=dict(t=70, b=10, l=20, r=20))
            st.plotly_chart(gb, use_container_width=True)

        shap_a = shap.TreeExplainer(xgb_tuned).shap_values(inp_a)[0]
        shap_b = shap.TreeExplainer(xgb_tuned).shap_values(inp_b)[0]
        fd = ['Age', 'Sex', 'BMI', 'Systolic BP', 'Diastolic BP']
        cmp_shap_fig = go.Figure()
        cmp_shap_fig.add_trace(go.Bar(name='Patient A', y=fd, x=shap_a, orientation='h', marker_color=BLUE,
            hovertemplate='%{y}: %{x:.4f}<extra>Patient A</extra>'))
        cmp_shap_fig.add_trace(go.Bar(name='Patient B', y=fd, x=shap_b, orientation='h', marker_color=AMBER,
            hovertemplate='%{y}: %{x:.4f}<extra>Patient B</extra>'))
        cmp_shap_fig.update_layout(
            barmode='group', paper_bgcolor=BG, plot_bgcolor=BG,
            font_color=TEXT, height=300, margin=dict(t=40, b=40, l=10, r=10),
            title=dict(text='SHAP Feature Contributions - Side by Side', font=dict(size=14)),
            xaxis=dict(title='SHAP Value (positive = increases risk)', gridcolor=GRID, zeroline=True, zerolinecolor=MUTED),
            yaxis=dict(gridcolor=GRID), legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1)
        )
        st.plotly_chart(cmp_shap_fig, use_container_width=True)

        diff = pct_b - pct_a
        diff_dir = "higher" if diff > 0 else "lower"
        max_diff_idx = np.argmax(np.abs(shap_b - shap_a))
        st.markdown(f"""
        <div class="info-box">
            <strong>Comparison Summary:</strong>
            Patient B's risk is <strong>{abs(diff):.1f} percentage points
            {diff_dir}</strong> than Patient A's.
            The largest contributing difference is in
            <strong>{fd[max_diff_idx]}</strong>,
            which accounts for a SHAP difference of
            {abs(shap_b[max_diff_idx] - shap_a[max_diff_idx]):.4f}.
        </div>""", unsafe_allow_html=True)

# ── TAB 2 MODEL RESULTS ───────────────────────────────────────
with tab2:
    st.markdown("### Model Comparison")
    st.caption("Five classifiers trained on non-invasive features "
               "and evaluated at threshold 0.15.")

    all_models = [
        ("Logistic Regression", lr,          X_test_sc, False),
        ("Random Forest",       rf,          X_test,    False),
        ("XGBoost Default",     xgb_default, X_test,    False),
        ("Neural Network",      mlp,         X_test_sc, False),
        ("XGBoost Tuned",       xgb_tuned,   X_test,    True),
    ]
    all_results = []
    for name, model, X_t, primary in all_models:
        yp  = model.predict_proba(X_t)[:, 1]
        ypb = (yp >= THRESHOLD).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, ypb).ravel()
        all_results.append({
            "Model":     name,
            "AUC":       round(roc_auc_score(y_test, yp), 4),
            "Recall":    round(recall_score(y_test, ypb), 4),
            "Precision": round(precision_score(y_test, ypb), 4),
            "F1":        round(f1_score(y_test, ypb), 4),
            "Brier":     round(brier_score_loss(y_test, yp), 4),
            "Missed":    fn_,
            "Caught":    tp_,
            "Primary":   primary
        })
    res_df = pd.DataFrame(all_results).sort_values(
        "AUC", ascending=False
    ).reset_index(drop=True)

    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    cols_r = st.columns(5)
    for col, (idx, row) in zip(cols_r, res_df.iterrows()):
        medal  = medals[idx]
        border = GREEN if row['Primary'] else GRID
        vc     = GREEN if row['Primary'] else TEXT
        tag    = "PRIMARY MODEL" if row['Primary'] else "Model"
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border:1px solid {border};">
                <div style="font-size:1.5rem;">{medal}</div>
                <div class="kpi-label" style="margin-top:6px;">
                    {tag}</div>
                <div style="color:{vc};font-weight:700;
                             font-size:0.88rem;margin-top:4px;">
                    {row['Model']}</div>
                <div class="kpi-sub" style="margin-top:6px;">
                    AUC {row['AUC']}</div>
                <div class="kpi-sub">Recall {row['Recall']}</div>
                <div class="kpi-sub">Missed {row['Missed']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Full Results")
    display_df = res_df.drop(columns=['Primary']).copy()
    display_df.insert(0, 'Rank',
                      [f"{m} {i+1}" for i, m in enumerate(medals)])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### ROC Curves")
        st.caption("Hover over any curve for exact values. "
                   "Curves closer to the upper left corner "
                   "indicate stronger performance.")
        roc_fig = go.Figure()
        colors5 = [BLUE, GREEN, AMBER, PURPLE, RED]
        for (name, model, X_t, primary), c in zip(all_models, colors5):
            yp = model.predict_proba(X_t)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, yp)
            auc = roc_auc_score(y_test, yp)
            lw  = 3 if primary else 1.5
            dash = 'solid' if primary else 'dash'
            lbl = f"{'* ' if primary else ''}{name} ({auc:.3f})"
            roc_fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=lbl,
                line=dict(color=c, width=lw, dash=dash),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
        roc_fig.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines', name='Random (0.500)',
            line=dict(color=GRID, width=1, dash='dash'), showlegend=True
        ))
        roc_fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font_color=TEXT, height=450,
            margin=dict(t=30, b=40, l=50, r=20),
            xaxis=dict(title='False Positive Rate', gridcolor=GRID, zeroline=False),
            yaxis=dict(title='True Positive Rate', gridcolor=GRID, zeroline=False),
            legend=dict(bgcolor=BG, bordercolor=GRID, borderwidth=1, font=dict(size=10))
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with col_r:
        st.markdown("#### Missed Diabetics at Threshold 0.15")
        st.caption("Fewer missed patients means better screening. "
                   "Hover for exact counts.")
        names_r   = [r['Model'] for _, r in res_df.iterrows()]
        missed_r  = [r['Missed'] for _, r in res_df.iterrows()]
        primary_r = [r['Primary'] for _, r in res_df.iterrows()]
        bar_clrs  = [GREEN if p else BLUE for p in primary_r]
        missed_fig = go.Figure(go.Bar(
            x=missed_r[::-1], y=names_r[::-1], orientation='h',
            marker_color=bar_clrs[::-1],
            text=missed_r[::-1], textposition='outside',
            textfont=dict(color=TEXT, size=12),
            hovertemplate='%{y}: %{x} missed<extra></extra>'
        ))
        missed_fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font_color=TEXT, height=450,
            margin=dict(t=30, b=40, l=10, r=40),
            xaxis=dict(title='Missed Diabetic Patients', gridcolor=GRID, zeroline=False),
            yaxis=dict(gridcolor=GRID)
        )
        st.plotly_chart(missed_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Confusion Matrix")
    st.caption("Patient counts at threshold 0.15. "
               "Bottom left cell shows missed diabetics.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG2)
    for ax, (name, model, X_t, _) in zip(axes, [
        ("Logistic Regression", lr,        X_test_sc, False),
        ("XGBoost Tuned",       xgb_tuned, X_test,    True)
    ]):
        yp  = model.predict_proba(X_t)[:, 1]
        ypb = (yp >= THRESHOLD).astype(int)
        cm  = confusion_matrix(y_test, ypb)
        tn_, fp_, fn_, tp_ = cm.ravel()
        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap='Blues', ax=ax,
            xticklabels=['Predicted No Diabetes',
                         'Predicted Diabetes'],
            yticklabels=['Actual No Diabetes',
                         'Actual Diabetes'],
            annot_kws={'size': 13, 'weight': 'bold'}
        )
        ax.set_title(
            f"{name}   Caught {tp_} of 315   Missed {fn_}",
            color=TEXT, fontsize=10
        )
        ax.tick_params(colors=TEXT)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    best = res_df[res_df['Primary']==True].iloc[0]
    st.markdown(f"""
    <div class="ok-box">
        <strong>Primary Model: XGBoost Tuned</strong><br>
        Selected following GridSearchCV across 192 parameter combinations.
        AUC {best['AUC']}, Recall {best['Recall']},
        {best['Missed']} missed diabetic patients.
        Shallow trees of depth 3 with learning rate 0.01 and 300 estimators
        produced the best generalisation on this dataset.
    </div>""", unsafe_allow_html=True)

# ── TAB 3 CALIBRATION ────────────────────────────────────────
with tab3:
    st.markdown("### Probability Calibration")
    st.caption("Assesses whether predicted risk percentages reflect "
               "true outcome frequencies.")

    st.markdown(f"""
    <div class="info-box">
        <strong>Why this matters:</strong> A model predicting 80% risk
        should correctly identify approximately 80% of such patients as diabetic.
        Without calibration assessment, probability outputs cannot be trusted
        as meaningful risk estimates (Van Calster et al., 2019).
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    y_true_arr   = np.array(y_test)
    y_prob_uncal = xgb_tuned.predict_proba(X_test)[:, 1]
    y_prob_platt = xgb_platt.predict_proba(X_test)[:, 1]
    y_prob_iso   = xgb_iso.predict_proba(X_test)[:, 1]

    b_uncal = brier_score_loss(y_test, y_prob_uncal)
    b_platt = brier_score_loss(y_test, y_prob_platt)
    b_iso   = brier_score_loss(y_test, y_prob_iso)
    e_uncal = compute_ece(y_true_arr, y_prob_uncal)
    e_platt = compute_ece(y_true_arr, y_prob_platt)
    e_iso   = compute_ece(y_true_arr, y_prob_iso)

    m1, m2, m3 = st.columns(3)
    cal_data = [
        ("Uncalibrated",  b_uncal, e_uncal, GREEN,
         "Best ECE result"),
        ("Platt Scaling", b_platt, e_platt, AMBER,
         "Sigmoid calibration"),
        ("Isotonic",      b_iso,   e_iso,   BLUE,
         "Non-parametric calibration"),
    ]
    for col, (method, brier, ece, color, note) in zip(
        [m1, m2, m3], cal_data
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border:1px solid {color};">
                <div class="kpi-label">{method}</div>
                <div class="kpi-value"
                     style="color:{color};font-size:1.5rem;">
                    {brier:.4f}</div>
                <div class="kpi-sub">Brier Score</div>
                <div style="margin-top:8px;color:{MUTED};
                             font-size:0.85rem;">
                    ECE {ece:.4f}</div>
                <div style="margin-top:4px;color:{MUTED};
                             font-size:0.78rem;">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Calibration Curves")
        st.caption("A well calibrated model follows the diagonal line closely.")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        style_ax(ax)
        for probs, label, color in [
            (y_prob_uncal, f"Uncalibrated (ECE {e_uncal:.4f})", GREEN),
            (y_prob_platt, f"Platt Scaling (ECE {e_platt:.4f})", AMBER),
            (y_prob_iso,   f"Isotonic      (ECE {e_iso:.4f})",   BLUE)
        ]:
            fp, mp = calibration_curve(y_test, probs, n_bins=10)
            ax.plot(mp, fp, 's-', color=color, label=label, lw=1.8)
        ax.plot([0,1],[0,1], color=MUTED, ls='--',
                label='Perfect Calibration')
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curves")
        ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("#### Brier Score and ECE")
        st.caption("Lower values are better for both metrics.")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        style_ax(ax)
        methods = ['Uncalibrated','Platt','Isotonic']
        briers  = [b_uncal, b_platt, b_iso]
        eces    = [e_uncal, e_platt, e_iso]
        x       = np.arange(3)
        w       = 0.35
        b1 = ax.bar(x-w/2, briers, w, label='Brier Score',
                    color=GREEN, alpha=0.85, edgecolor='none')
        b2 = ax.bar(x+w/2, eces,   w, label='ECE',
                    color=BLUE,  alpha=0.85, edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Score')
        ax.set_title('Calibration Metrics')
        ax.legend(facecolor=BG, labelcolor=TEXT)
        for bar in list(b1) + list(b2):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{bar.get_height():.4f}',
                ha='center', color=TEXT, fontsize=8
            )
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown(f"""
    <div class="ok-box">
        <strong>Finding:</strong> Tuned XGBoost demonstrates strong inherent
        calibration with ECE of {e_uncal:.4f}, lower than both Platt Scaling
        ({e_platt:.4f}) and Isotonic Regression ({e_iso:.4f}). Calibration
        correction provides no improvement. The model's probability outputs
        can be used directly for clinical risk communication without
        post-hoc adjustment (Van Calster et al., 2019).
    </div>""", unsafe_allow_html=True)

# ── TAB 4 THRESHOLD ───────────────────────────────────────────
with tab4:
    st.markdown("### Threshold Optimisation")
    st.caption("The most impactful decision in the pipeline "
               "for a screening application.")

    st.markdown(f"""
    <div class="info-box">
        <strong>The problem with threshold 0.50:</strong> In a dataset where
        only 15.5% of patients are diabetic, a threshold of 0.50 means the
        model only flags patients it is extremely confident about.
        At this threshold zero diabetic patients are detected despite
        an AUC of 0.766. This confirms that AUC alone is an insufficient
        basis for clinical model evaluation (Van Calster et al., 2019).
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    y_prob_t = xgb_tuned.predict_proba(X_test)[:, 1]
    thresholds = [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]
    rows_t = []
    for t in thresholds:
        yp = (y_prob_t >= t).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, yp).ravel()
        rows_t.append({
            'Threshold':   t,
            'Sensitivity': round(tp_/(tp_+fn_), 3),
            'Specificity': round(tn_/(tn_+fp_), 3),
            'Precision':   round(
                tp_/(tp_+fp_) if tp_+fp_>0 else 0, 3),
            'F1':          round(f1_score(y_test, yp), 3),
            'Missed':      fn_,
            'Caught':      tp_,
            'Selected':    'CHOSEN' if t == 0.15 else ''
        })
    tdf = pd.DataFrame(rows_t)

    ta, tb, tc, td = st.columns(4)
    ta.metric("Chosen Threshold", "0.15")
    tb.metric("Sensitivity",      "81.3%",
              "Catches 256 of 315 diabetics")
    tc.metric("Missed at 0.15",   "59",
              "vs 315 missed at threshold 0.50")
    td.metric("Specificity",      "57.4%",
              "At chosen threshold")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Results at Each Threshold")
    st.dataframe(tdf, use_container_width=True, hide_index=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Sensitivity and Specificity Trade-off")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        style_ax(ax)
        ax.plot(tdf['Threshold'], tdf['Sensitivity'], 'o-',
                color=RED,   lw=2, label='Sensitivity')
        ax.plot(tdf['Threshold'], tdf['Specificity'], 'o-',
                color=GREEN, lw=2, label='Specificity')
        ax.plot(tdf['Threshold'], tdf['Precision'],   'o--',
                color=AMBER, lw=2, label='Precision')
        ax.axvline(0.15, color='white', ls='--',
                   lw=2, label='Chosen threshold 0.15')
        ax.fill_betweenx([0,1], 0, 0.15,
                          alpha=0.06, color=GREEN)
        ax.set_xlabel("Classification Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Sensitivity vs Specificity Trade-off")
        ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("#### Missed Diabetics at Each Threshold")
        st.caption("Red bar shows the chosen threshold. "
                   "The goal is to minimise missed cases.")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(BG)
        style_ax(ax)
        bar_clrs = [RED if t == 0.15 else BLUE
                    for t in tdf['Threshold']]
        bars = ax.bar(
            [str(t) for t in tdf['Threshold']],
            tdf['Missed'],
            color=bar_clrs, edgecolor='none'
        )
        for bar, val in zip(bars, tdf['Missed']):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 3,
                str(val), ha='center',
                color=TEXT, fontsize=9, fontweight='bold'
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Missed Diabetic Patients")
        ax.set_title("False Negatives at Each Threshold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    ba1, ba2 = st.columns(2)
    with ba1:
        st.markdown(f"""
        <div class="warn-box">
            <strong>Default Threshold 0.50</strong><br><br>
            The model catches zero diabetic patients out of 315
            in the test set.<br><br>
            Sensitivity 0.0%<br>
            315 patients missed<br><br>
            Clinically unusable despite AUC of 0.766. This confirms
            that discrimination metrics alone cannot evaluate a
            screening tool's real-world utility.
        </div>""", unsafe_allow_html=True)
    with ba2:
        st.markdown(f"""
        <div class="ok-box">
            <strong>Chosen Threshold 0.15</strong><br><br>
            The model catches 256 diabetic patients out of 315,
            a sensitivity of 81.3%.<br><br>
            59 patients missed<br>
            256 patients correctly referred<br><br>
            An unnecessary referral is always preferable to a
            missed diagnosis in a screening context.
        </div>""", unsafe_allow_html=True)

# ── TAB 5 ROBUSTNESS ─────────────────────────────────────────
with tab5:
    st.markdown("### Robustness Evaluation")
    st.caption("Validates that results are stable and not attributable "
               "to a favourable random data partition.")

    st.markdown(f"""
    <div class="info-box">
        Most published NHANES diabetes prediction studies report performance
        from a single train/test split. If results vary substantially across
        different splits, the reported metrics may be unreliable.
        Tuned XGBoost was retrained across ten random seeds to verify
        reproducibility.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    seeds   = [0,7,13,21,42,55,67,78,89,99]
    r_aucs, r_recs = [], []
    r_rows  = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            df[features], df['diabetes'].astype(int),
            test_size=0.2, random_state=seed,
            stratify=df['diabetes']
        )
        m = XGBClassifier(
            n_estimators=300, max_depth=3,
            learning_rate=0.01, subsample=0.8,
            colsample_bytree=0.8, random_state=seed,
            eval_metric='logloss', verbosity=0
        )
        m.fit(Xtr, ytr)
        yp  = m.predict_proba(Xte)[:, 1]
        ypb = (yp >= THRESHOLD).astype(int)
        a   = roc_auc_score(yte, yp)
        r   = recall_score(yte, ypb)
        r_aucs.append(a)
        r_recs.append(r)
        r_rows.append({
            'Seed':   seed,
            'AUC':    round(a, 4),
            'Recall': round(r, 4)
        })

    auc_mean = np.mean(r_aucs)
    auc_std  = np.std(r_aucs)
    rec_mean = np.mean(r_recs)
    rec_std  = np.std(r_recs)
    verdict  = "STABLE" if auc_std < 0.02 else "UNSTABLE"

    ra, rb, rc, rd = st.columns(4)
    ra.metric("AUC Mean",    f"{auc_mean:.4f}")
    rb.metric("AUC Std",     f"{auc_std:.4f}",
              "Below 0.02 is stable")
    rc.metric("Recall Mean", f"{rec_mean:.4f}")
    rd.metric("Verdict",     verdict)

    st.markdown("<br>", unsafe_allow_html=True)

    rob_df = pd.DataFrame(r_rows)
    st.dataframe(rob_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG2)
    for ax, vals, mean, std, label, color in zip(
        axes,
        [r_aucs, r_recs],
        [auc_mean, rec_mean],
        [auc_std,  rec_std],
        ['AUC', 'Recall at Threshold 0.15'],
        [GREEN, AMBER]
    ):
        style_ax(ax)
        ax.plot(seeds, vals, 'o-', color=color,
                lw=2, markersize=6)
        ax.axhline(mean, color='white', ls='--', lw=1.5,
                   label=f'Mean {mean:.4f}')
        ax.fill_between(
            seeds, mean - std, mean + std,
            alpha=0.15, color=color,
            label=f'Std {std:.4f}'
        )
        ax.set_xlabel('Random Seed')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Across 10 Seeds')
        ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
        ax.set_xticks(seeds)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    box_cls = "ok-box" if verdict == "STABLE" else "warn-box"
    st.markdown(f"""
    <div class="{box_cls}">
        <strong>Verdict: {verdict}.</strong>
        AUC standard deviation of {auc_std:.4f} is
        {'below' if auc_std < 0.02 else 'above'} the 0.02 stability threshold.
        Results are {'consistent and reproducible'
        if verdict == 'STABLE' else 'variable'}
        across all ten random splits.
    </div>""", unsafe_allow_html=True)

# ── TAB 6 EXPLAINABILITY ──────────────────────────────────────
with tab6:
    st.markdown("### Explainable AI")
    st.caption("SHAP and LIME provide two independent explanations "
               "of model predictions. Agreement between methods "
               "strengthens confidence in the explanation.")

    feat_display = ['Age','Sex','BMI','Systolic BP','Diastolic BP']

    st.markdown("#### Feature Importance")
    st.caption("Mean absolute SHAP values across all 2,034 test patients.")

    order = np.argsort(mean_shap)[::-1]
    medals_xai = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    icons_xai  = {
        'age': '🎂', 'sex': '👤',
        'bmi': '⚖️', 'sbp': '💓', 'dbp': '💗'
    }
    descs_xai = {
        'age': 'Strongest predictor. Older age significantly increases risk.',
        'sex': 'Weakest contributor. Males carry marginally higher risk.',
        'bmi': 'Second strongest. High BMI consistently increases risk.',
        'sbp': 'Moderate influence. Elevated blood pressure increases risk.',
        'dbp': 'Weakest blood pressure signal.'
    }
    for rank, i in enumerate(order):
        feat = features[i]
        val  = mean_shap[i]
        pct  = val / mean_shap.sum() * 100
        st.markdown(f"""
        <div class="kpi-card"
             style="margin-bottom:8px;text-align:left;padding:12px 16px;">
            <div style="display:flex;align-items:center;
                        justify-content:space-between;">
                <div>
                    <span style="font-size:1.2rem;">
                        {medals_xai[rank]} {icons_xai[feat]}</span>
                    <span style="color:{TEXT};font-weight:700;
                                 font-size:0.95rem;margin-left:8px;">
                        {feat_display[i]}</span>
                    <span style="color:{MUTED};font-size:0.8rem;
                                 margin-left:10px;">
                        {descs_xai[feat]}</span>
                </div>
                <div style="color:{BLUE};font-weight:700;">
                    {val:.3f} ({pct:.1f}%)</div>
            </div>
            <div style="margin-top:6px;background:{GRID};
                        border-radius:4px;height:6px;">
                <div style="background:{BLUE};border-radius:4px;
                             height:6px;
                             width:{min(int(pct*3),100)}%;">
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Global SHAP View")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Beeswarm Plot**")
        st.caption("Red dots represent patients with high feature values. "
                   "Position on the right side increases predicted risk.")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        fig1.patch.set_facecolor(BG)
        plt.sca(ax1)
        shap.summary_plot(
            shap_vals, X_test_np,
            feature_names=feat_display,
            show=False, plot_size=None
        )
        fig1 = plt.gcf()
        fig1.patch.set_facecolor(BG)
        fig1.axes[0].set_facecolor(BG)
        fig1.axes[0].tick_params(colors=TEXT)
        fig1.axes[0].xaxis.label.set_color(TEXT)
        plt.tight_layout()
        st.pyplot(fig1); plt.clf()

    with col_r:
        st.markdown("**Mean Absolute SHAP Values**")
        st.caption("Longer bar means stronger average influence "
                   "on predictions across all patients.")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor(BG)
        style_ax(ax2)
        feat_ord  = np.argsort(mean_shap)
        bar_clrs2 = [GREEN, AMBER, RED, BLUE, PURPLE]
        bars2 = ax2.barh(
            [feat_display[i] for i in feat_ord],
            mean_shap[feat_ord],
            color=bar_clrs2, edgecolor='none', height=0.5
        )
        for bar, val in zip(bars2, mean_shap[feat_ord]):
            ax2.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}',
                color=TEXT, va='center', fontsize=9
            )
        ax2.set_xlabel("Mean Absolute SHAP Value")
        ax2.set_title("Feature Importance Ranking", color=TEXT)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Individual Patient Explorer")
    st.markdown(f"""
    <div class="info-box">
        Select a patient from the test set. SHAP and LIME charts update
        to show why the model gave that specific risk score.
        Red bars increase risk. Green bars decrease risk.
        When both methods agree on a feature, confidence in the
        explanation is higher.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    p_idx = st.slider("Select patient", 0, len(X_test)-1, 0)

    p_data    = X_test_np[p_idx]
    p_prob    = xgb_tuned.predict_proba(p_data.reshape(1,-1))[0][1]
    p_label   = int(y_test.iloc[p_idx])
    p_shap    = shap_vals[p_idx]
    p_correct = (p_prob >= THRESHOLD) == (p_label == 1)

    r_col  = RED if p_prob>=0.35 else AMBER if p_prob>=0.15 else GREEN
    r_text = ("HIGH RISK" if p_prob>=0.35
              else "MEDIUM RISK" if p_prob>=0.15 else "LOW RISK")

    st.markdown(f"""
    <div class="patient-card">
        <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:16px;">
            <div>
                <div class="kpi-label">Patient {p_idx}</div>
                <div style="color:{r_col};font-size:2.2rem;
                             font-weight:800;">{p_prob*100:.1f}%</div>
                <div style="color:{r_col};font-size:0.95rem;">
                    {r_text}</div>
            </div>
            <div style="text-align:center;">
                <div class="kpi-label">Actual Diagnosis</div>
                <div style="font-size:1.4rem;">
                    {'🩸 Diabetic' if p_label==1
                     else '✅ Not Diabetic'}
                </div>
            </div>
            <div style="text-align:center;">
                <div class="kpi-label">Model Correct</div>
                <div style="font-size:1.4rem;">
                    {'✅ Yes' if p_correct else '❌ No'}
                </div>
            </div>
            <div>
                <div class="kpi-label">Profile</div>
                <div style="color:{TEXT};font-size:0.84rem;">
                    Age {p_data[0]:.0f} &nbsp; BMI {p_data[2]:.1f}<br>
                    SBP {p_data[3]:.0f} &nbsp; DBP {p_data[4]:.0f}<br>
                    {'Male' if p_data[1]==1 else 'Female'}
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    col_shap, col_lime = st.columns(2)

    with col_shap:
        st.markdown("**SHAP Attribution**")
        s_idx_p = np.argsort(np.abs(p_shap))
        s_feats = [feat_display[i] for i in s_idx_p]
        s_vals  = p_shap[s_idx_p]
        s_data  = p_data[s_idx_p]
        fig_s, ax_s = plt.subplots(figsize=(6, 3.5))
        fig_s.patch.set_facecolor(BG)
        style_ax(ax_s)
        s_colors = [RED if v>0 else GREEN for v in s_vals]
        ax_s.barh(
            [f"{f} = {s_data[i]:.1f}"
             for i, f in enumerate(s_feats)],
            s_vals, color=s_colors,
            edgecolor='none', height=0.5
        )
        ax_s.axvline(0, color=MUTED, lw=0.8)
        ax_s.set_xlabel("SHAP Value", color=TEXT, fontsize=8)
        ax_s.set_title("SHAP Feature Attribution",
                       color=TEXT, fontsize=9)
        for bar, val in zip(ax_s.patches, s_vals):
            ax_s.text(
                val + (0.003 if val>=0 else -0.003),
                bar.get_y() + bar.get_height()/2,
                f'{"▲" if val>0 else "▼"} {abs(val):.3f}',
                color=TEXT, va='center', fontsize=7,
                ha='left' if val>=0 else 'right'
            )
        plt.tight_layout()
        st.pyplot(fig_s); plt.close()

    with col_lime:
        st.markdown("**LIME Attribution**")
        lime_exp = lime_explainer.explain_instance(
            data_row   = p_data,
            predict_fn = lambda x: xgb_tuned.predict_proba(x),
            num_features = 5,
            num_samples  = 500,
            labels       = (1,)
        )
        l_list   = lime_exp.as_list(label=1)
        l_feats  = [e[0] for e in l_list]
        l_vals   = [e[1] for e in l_list]
        l_colors = [RED if v>0 else GREEN for v in l_vals]
        fig_l, ax_l = plt.subplots(figsize=(6, 3.5))
        fig_l.patch.set_facecolor(BG)
        style_ax(ax_l)
        ax_l.barh(l_feats, l_vals, color=l_colors,
                  edgecolor='none', height=0.5)
        ax_l.axvline(0, color=MUTED, lw=0.8)
        ax_l.set_xlabel("LIME Weight", color=TEXT, fontsize=8)
        ax_l.set_title("LIME Feature Attribution",
                       color=TEXT, fontsize=9)
        for bar, val in zip(ax_l.patches, l_vals):
            ax_l.text(
                val + (0.001 if val>=0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.4f}',
                color=TEXT, va='center', fontsize=7,
                ha='left' if val>=0 else 'right'
            )
        plt.tight_layout()
        st.pyplot(fig_l); plt.close()

    top_up   = [(s_feats[i], s_vals[i], s_data[i])
                for i in range(len(s_vals)-1,-1,-1)
                if s_vals[i] > 0]
    top_down = [(s_feats[i], s_vals[i], s_data[i])
                for i in range(len(s_vals)-1,-1,-1)
                if s_vals[i] < 0]
    exp_txt  = f"Patient {p_idx}. "
    if top_up:
        exp_txt += ("Features increasing risk: "
                    + ", ".join([
                        f"<strong>{f} ({d:.1f})</strong>"
                        for f, v, d in top_up[:2]
                    ]) + ". ")
    if top_down:
        exp_txt += ("Features reducing risk: "
                    + ", ".join([
                        f"<strong>{f} ({d:.1f})</strong>"
                        for f, v, d in top_down[:2]
                    ]) + ".")
    exp_txt += (f" Model predicted <strong>{r_text}</strong> "
                f"({'correct' if p_correct else 'incorrect'}). "
                f"Actual diagnosis: <strong>"
                f"{'Diabetic' if p_label==1 else 'Not Diabetic'}"
                f"</strong>.")
    box_cls = "ok-box" if p_correct else "warn-box"
    st.markdown(f'<div class="{box_cls}" style="margin-top:12px;">'
                f'{exp_txt}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box" style="margin-top:10px;font-size:0.8rem;">
        SHAP and LIME show statistical associations not medical causation.
        For screening purposes only.
    </div>""", unsafe_allow_html=True)

# ── TAB 7 CLINICAL RULES ──────────────────────────────────────
with tab7:
    st.markdown("### Clinical Rule Layer")
    st.caption("The most novel contribution of this project. "
               "Not identified in any reviewed NHANES-based study.")

    st.markdown(f"""
    <div class="info-box">
        A machine learning model trained on non-invasive features can assign
        low risk to a patient whose laboratory values clearly meet ADA
        diagnostic criteria for diabetes. Without a contradiction detection
        mechanism this clinically inconsistent output would be presented
        without warning. The rule layer acts as a safety net, catching cases
        where the model's blind spot could have real clinical consequences.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Rule Definitions")
    r1c, r2c, r3c, r4c = st.columns(4)
    rule_cards = [
        (RED,   "Contradiction",
         "ML says Low Risk but A1C or Glucose meets ADA threshold",
         "Most critical case. Patient needs urgent referral."),
        (RED,   "High Risk Confirmed",
         "ML says High Risk and lab values confirm diabetes",
         "Both sources agree. Strongest evidence for referral."),
        (GREEN, "Low Risk Confirmed",
         "ML says Low Risk and lab values are normal",
         "Full agreement. Patient can be reassured."),
        (AMBER, "Model Flag Only",
         "ML flags risk but lab values are within normal range",
         "Monitor closely. May indicate pre-diabetic trajectory."),
    ]
    for col, (color, title, rule, meaning) in zip(
        [r1c, r2c, r3c, r4c], rule_cards
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card"
                 style="border:1px solid {color};text-align:left;">
                <div style="color:{color};font-weight:700;
                             font-size:0.9rem;">{title}</div>
                <div style="color:{MUTED};font-size:0.78rem;
                             margin-top:6px;">{rule}</div>
                <div style="color:{TEXT};font-size:0.8rem;
                             margin-top:8px;border-top:1px solid {GRID};
                             padding-top:6px;">{meaning}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Four Real NHANES Patient Cases")
    st.caption("Drawn from the actual dataset. "
               "Real lab values, real ML predictions, "
               "real clinical implications.")

    cases = [
        {
            "title":   "Case 1   Contradiction",
            "profile": "Age 56   Male   BMI 24.4   SBP 114   DBP 81",
            "lab":     "A1C 13.7%   Glucose 397 mg/dL",
            "actual":  "Diabetic",
            "prob":    0.142,
            "status":  "CONTRADICTION WARNING",
            "detail":  ("The model assigns only 14.2% risk because the "
                        "patient appears demographically low-risk. Lean "
                        "build, moderate age, normal blood pressure. "
                        "However A1C of 13.7% and glucose of 397 mg/dL "
                        "clearly meet ADA diagnostic criteria. "
                        "This is the key finding. The rule layer catches "
                        "what the model misses entirely."),
            "box":     "warn-box",
            "border":  RED
        },
        {
            "title":   "Case 2   Agreement High Risk",
            "profile": "Age 67   Male   BMI 28.8   SBP 134   DBP 83",
            "lab":     "A1C 8.6%   Glucose 284 mg/dL",
            "actual":  "Diabetic",
            "prob":    0.262,
            "status":  "HIGH RISK CONFIRMED",
            "detail":  ("Model and clinical rules agree. Older male with "
                        "elevated BMI and blood pressure correctly flagged "
                        "by both the ML model and ADA criteria. "
                        "Consistent evidence from two independent sources."),
            "box":     "warn-box",
            "border":  AMBER
        },
        {
            "title":   "Case 3   Agreement Low Risk",
            "profile": "Age 42   Female   BMI 20.3   SBP 107   DBP 62",
            "lab":     "A1C 5.6%   Glucose 84 mg/dL",
            "actual":  "Not Diabetic",
            "prob":    0.034,
            "status":  "AGREEMENT LOW RISK",
            "detail":  ("Young female patient with healthy BMI and normal "
                        "blood pressure. Both model and lab values confirm "
                        "low risk. Patient can be reassured with confidence "
                        "that two independent assessment methods agree."),
            "box":     "ok-box",
            "border":  GREEN
        },
        {
            "title":   "Case 4   Model Flag",
            "profile": "Age 53   Male   BMI 30.8   SBP 143   DBP 88",
            "lab":     "A1C 5.5%   Glucose 101 mg/dL",
            "actual":  "Not Diabetic",
            "prob":    0.222,
            "status":  "MODEL FLAG   LABS NORMAL",
            "detail":  ("Model flags elevated risk based on age, BMI and "
                        "blood pressure, a profile consistent with metabolic "
                        "syndrome. Lab values are currently within normal "
                        "range. This patient may represent a pre-diabetic "
                        "trajectory warranting monitoring rather than "
                        "immediate referral."),
            "box":     "amber-box",
            "border":  AMBER
        }
    ]

    for case in cases:
        decision    = "High Risk" if case['prob'] >= THRESHOLD else "Low Risk"
        prob_color  = (RED if case['prob'] >= 0.35
                       else AMBER if case['prob'] >= 0.15
                       else GREEN)
        st.markdown(f"""
        <div class="patient-card"
             style="border:1px solid {case['border']};
                    margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;
                        flex-wrap:wrap;gap:12px;">
                <div style="flex:1;min-width:200px;">
                    <div style="color:{case['border']};font-weight:700;
                                 font-size:1rem;margin-bottom:8px;">
                        {case['title']}</div>
                    <div style="color:{MUTED};font-size:0.82rem;">
                        {case['profile']}</div>
                    <div style="color:{MUTED};font-size:0.82rem;
                                 margin-top:4px;">
                        Lab values: {case['lab']}</div>
                    <div style="color:{TEXT};font-size:0.82rem;
                                 margin-top:4px;">
                        Actual diagnosis:
                        <strong>{case['actual']}</strong></div>
                </div>
                <div style="text-align:center;min-width:100px;">
                    <div class="kpi-label">ML Probability</div>
                    <div style="color:{prob_color};font-size:1.8rem;
                                 font-weight:800;">
                        {case['prob']*100:.1f}%</div>
                    <div style="color:{prob_color};font-size:0.82rem;">
                        {decision}</div>
                </div>
                <div style="text-align:center;min-width:160px;">
                    <div class="kpi-label">Rule Layer Output</div>
                    <div style="color:{case['border']};font-weight:700;
                                 font-size:0.85rem;margin-top:4px;">
                        {case['status']}</div>
                </div>
            </div>
            <div style="margin-top:12px;padding-top:12px;
                        border-top:1px solid {GRID};
                        color:{MUTED};font-size:0.82rem;">
                {case['detail']}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warn-box">
        <strong>Key Finding from Case 1.</strong><br>
        A lean diabetic patient with BMI 24.4 and normal blood pressure
        receives only 14.2% predicted risk from the ML model, below the
        screening threshold. Without the contradiction layer this patient
        would leave the screening process undetected. An A1C of 13.7%
        and glucose of 397 mg/dL are not borderline values. They represent
        severe uncontrolled diabetes. The rule layer catches this case and
        raises an immediate warning.<br><br>
        No equivalent mechanism was identified in any reviewed NHANES-based
        diabetes prediction study (ADA, 2023).
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Interactive Contradiction Checker")
    st.caption("Enter a predicted probability and lab values "
               "to see what the rule layer outputs.")

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        test_prob = st.slider("ML Predicted Probability",
                              0.0, 1.0, 0.12, 0.01)
    with ic2:
        test_a1c  = st.number_input("A1C (%)", 0.0, 20.0, 7.5, 0.1,
                                     key="rule_a1c")
    with ic3:
        test_glc  = st.number_input("Glucose (mg/dL)",
                                     0.0, 500.0, 150.0, 1.0,
                                     key="rule_glc")

    t_status, t_reasons = clinical_rule_check(
        test_a1c if test_a1c > 0 else None,
        test_glc if test_glc > 0 else None,
        test_prob
    )
    decision_txt = "High Risk" if test_prob >= THRESHOLD else "Low Risk"

    if t_status == "contradiction":
        box_t = "warn-box"
        msg   = (f"<strong>CONTRADICTION WARNING.</strong> "
                 f"Model predicts {decision_txt} "
                 f"({test_prob*100:.1f}%) but lab values indicate "
                 f"diabetes. {' '.join(t_reasons)}")
    elif t_status == "agree_high":
        box_t = "warn-box"
        msg   = (f"<strong>HIGH RISK CONFIRMED.</strong> "
                 f"Model and clinical rules agree. "
                 f"{' '.join(t_reasons)}")
    elif t_status == "model_flag":
        box_t = "amber-box"
        msg   = (f"<strong>MODEL FLAG.</strong> "
                 f"Model predicts {decision_txt} "
                 f"({test_prob*100:.1f}%) but lab values "
                 f"are within normal range.")
    else:
        box_t = "ok-box"
        msg   = (f"<strong>AGREEMENT LOW RISK.</strong> "
                 f"Model ({test_prob*100:.1f}%) and lab values agree. "
                 f"No diabetes indicators present.")

    st.markdown(
        f'<div class="{box_t}" style="margin-top:10px;">'
        f'{msg}</div>',
        unsafe_allow_html=True
    )

# ── FOOTER ─────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("For screening purposes only. Not a clinical diagnosis.")
with col_f2:
    st.caption("NHANES 2015-2018   Tuned XGBoost   AUC 0.7658")
with col_f3:
    st.caption("Nottingham Trent University   FYP 2026")
