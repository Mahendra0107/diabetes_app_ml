import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI · Predictive Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

/* ── Root & Background ── */
:root {
    --neon-cyan: #00f5ff;
    --neon-purple: #bf00ff;
    --neon-green: #00ff88;
    --neon-orange: #ff6b35;
    --dark-bg: #020408;
    --panel-bg: rgba(0, 245, 255, 0.03);
    --border: rgba(0, 245, 255, 0.15);
    --text-primary: #e8f4f8;
    --text-muted: rgba(232,244,248,0.5);
}

.stApp {
    background: var(--dark-bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,245,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(191,0,255,0.06) 0%, transparent 60%),
        linear-gradient(180deg, #020408 0%, #040a10 50%, #020408 100%);
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-primary);
}

/* Grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(2, 4, 8, 0.95) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(180deg, rgba(0,245,255,0.05) 0%, transparent 40%);
    pointer-events: none;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.08em;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.2rem 1.5rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
}
[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--text-primary) !important;
    font-size: 2rem !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em;
    padding: 0.6rem 1.5rem !important;
    border-radius: 2px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.1) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3), inset 0 0 20px rgba(0,245,255,0.05) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple)) !important;
}
.stSlider label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em;
}

/* ── Select Boxes ── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    letter-spacing: 0.08em;
    font-size: 0.78rem !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text-primary) !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0;
    border-bottom: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em;
    color: var(--text-muted) !important;
    background: transparent !important;
    padding: 0.7rem 1.5rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    background: rgba(0,245,255,0.05) !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--neon-cyan) !important;
    background: var(--panel-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

/* ── Info / Warning boxes ── */
.stInfo, .stSuccess, .stWarning {
    font-family: 'Rajdhani', sans-serif !important;
    border-radius: 2px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--panel-bg) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* ── Glow panels (custom) ── */
.glow-panel {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.5rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.glow-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
}
.glow-panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--neon-cyan);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.glow-panel-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e8f4f8;
}
.glow-panel-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Prediction result ── */
.prediction-box {
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.prediction-box.high-risk {
    background: rgba(255, 50, 50, 0.06);
    border: 1px solid rgba(255, 50, 50, 0.4);
}
.prediction-box.low-risk {
    background: rgba(0, 255, 136, 0.06);
    border: 1px solid rgba(0, 255, 136, 0.4);
}
.prediction-box.pre-diabetic {
    background: rgba(255, 165, 0, 0.06);
    border: 1px solid rgba(255, 165, 0, 0.4);
}
.prediction-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.prediction-result {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
}
.tag {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    padding: 0.2rem 0.8rem;
    border-radius: 2px;
    text-transform: uppercase;
    margin: 0.2rem;
}
.tag-cyan { background: rgba(0,245,255,0.1); border: 1px solid rgba(0,245,255,0.3); color: var(--neon-cyan); }
.tag-purple { background: rgba(191,0,255,0.1); border: 1px solid rgba(191,0,255,0.3); color: #bf00ff; }
.tag-green { background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: var(--neon-green); }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.5rem 0;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:rgba(0,245,255,0.6); letter-spacing:0.3em; text-transform:uppercase; margin-bottom:0.5rem;">
        ◈ NEURAL DIAGNOSTICS SYSTEM v2.4
    </div>
    <h1 style="font-family:'Orbitron',monospace; font-size:clamp(1.8rem,4vw,3rem); font-weight:900; color:#e8f4f8; margin:0; letter-spacing:0.05em; line-height:1.1;">
        DIABETES<span style="color:#00f5ff;">AI</span>
    </h1>
    <div style="font-family:'Rajdhani',sans-serif; font-size:1.05rem; color:rgba(232,244,248,0.55); margin-top:0.4rem; letter-spacing:0.08em;">
        Predictive Analytics Platform &nbsp;·&nbsp; Multi-Model Classification Engine
    </div>
    <div style="margin-top:1rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
        <span class="tag tag-cyan">Logistic Regression</span>
        <span class="tag tag-purple">Decision Tree</span>
        <span class="tag tag-green">Random Forest</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'Orbitron',monospace; font-size:1.2rem; color:#00f5ff; letter-spacing:0.15em;">⬡ CONTROLS</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:rgba(0,245,255,0.4); letter-spacing:0.2em; margin-top:0.3rem;">SYSTEM PARAMETERS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📂 Data Source")
    data_source = st.radio(
        "Select dataset",
        ["Use sample diabetes.csv", "Upload your own CSV"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "Upload your own CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("---")
    st.markdown("#### 🔬 Active Models")
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    use_rf = st.checkbox("Random Forest", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:rgba(0,245,255,0.25); letter-spacing:0.15em; line-height:2;">
    DATASET · 253,680 RECORDS<br>
    FEATURES · 21 VARIABLES<br>
    TARGET · DIABETES_012<br>
    MODELS · PRE-TRAINED<br>
    FRAMEWORK · SKLEARN 1.x
    </div>
    """, unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
NEON_CYAN = "#00f5ff"
NEON_PURPLE = "#bf00ff"
NEON_GREEN = "#00ff88"
NEON_ORANGE = "#ff6b35"
DARK_BG = "#020408"
PANEL_BG = "#050d14"

def style_matplotlib():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": "#0a2030",
        "axes.labelcolor": "#7ab8cc",
        "axes.titlecolor": NEON_CYAN,
        "xtick.color": "#7ab8cc",
        "ytick.color": "#7ab8cc",
        "text.color": "#e8f4f8",
        "grid.color": "#0a1a25",
        "grid.linewidth": 0.6,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 5000
    data = pd.DataFrame({
        'Diabetes_012': np.random.choice([0, 1, 2], n, p=[0.73, 0.02, 0.25]),
        'HighBP': np.random.choice([0, 1], n, p=[0.57, 0.43]),
        'HighChol': np.random.choice([0, 1], n, p=[0.56, 0.44]),
        'CholCheck': np.random.choice([0, 1], n, p=[0.07, 0.93]),
        'BMI': np.clip(np.random.normal(28, 7, n), 12, 98).round(1),
        'Smoker': np.random.choice([0, 1], n, p=[0.56, 0.44]),
        'Stroke': np.random.choice([0, 1], n, p=[0.96, 0.04]),
        'HeartDiseaseorAttack': np.random.choice([0, 1], n, p=[0.91, 0.09]),
        'PhysActivity': np.random.choice([0, 1], n, p=[0.25, 0.75]),
        'Fruits': np.random.choice([0, 1], n, p=[0.37, 0.63]),
        'Veggies': np.random.choice([0, 1], n, p=[0.19, 0.81]),
        'HvyAlcoholConsump': np.random.choice([0, 1], n, p=[0.94, 0.06]),
        'AnyHealthcare': np.random.choice([0, 1], n, p=[0.05, 0.95]),
        'NoDocbcCost': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'GenHlth': np.random.choice([1, 2, 3, 4, 5], n),
        'MentHlth': np.clip(np.random.exponential(3, n), 0, 30).round(),
        'PhysHlth': np.clip(np.random.exponential(4, n), 0, 30).round(),
        'DiffWalk': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'Sex': np.random.choice([0, 1], n),
        'Age': np.random.choice(range(1, 14), n),
        'Education': np.random.choice(range(1, 7), n),
        'Income': np.random.choice(range(1, 9), n),
    })
    return data

@st.cache_resource
def load_models():
    """Load pre-trained models and metadata from the models/ directory."""
    import pickle, os
    models_dir = "models"
    model_files = {
        'Logistic Regression': ('logistic_regression.pkl', NEON_CYAN),
        'Decision Tree':       ('decision_tree.pkl',       NEON_ORANGE),
        'Random Forest':       ('random_forest.pkl',       NEON_GREEN),
    }
    meta_path = os.path.join(models_dir, "metadata.pkl")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    results = {}
    for display_name, (fname, color) in model_files.items():
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            model = pickle.load(f)
        res_meta = meta["results"].get(fname.replace(".pkl", ""), {})
        results[display_name] = {
            "model":    model,
            "accuracy": res_meta.get("accuracy", 0),
            "report":   res_meta.get("report", {}),
            "cm":       np.array(res_meta.get("cm", [[0]])),
            "color":    color,
        }
    return results, meta

# ─── Load Models ────────────────────────────────────────────────────────────────
all_models, meta = load_models()

if all_models is None:
    st.error("❌ No pre-trained models found. Run `python train_and_save.py` locally first, then push the `models/` folder to GitHub.")
    st.stop()

# Filter by sidebar checkboxes
MODEL_DISPLAY = {
    'Logistic Regression': use_lr,
    'Decision Tree':       use_dt,
    'Random Forest':       use_rf,
}
results = {k: v for k, v in all_models.items() if MODEL_DISPLAY.get(k, False)}
feature_names = meta.get("feature_names", [])

# ─── Load Data ──────────────────────────────────────────────────────────────────
if data_source == "Upload your own CSV" and uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Diabetes_012' not in df.columns:
        st.error("❌ CSV must contain 'Diabetes_012' column as target.")
        st.stop()
    data = df
elif data_source == "Use sample diabetes.csv":
    data = load_sample_data()
else:
    st.info("⬆ Upload a CSV file in the sidebar to get started.")
    st.stop()

# ─── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  OVERVIEW",
    "🧠  MODEL TRAINING",
    "📈  PERFORMANCE",
    "🔮  PREDICT"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### DATASET OVERVIEW")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    with col2:
        st.metric("Features", f"{data.shape[1]-1}")
    with col3:
        st.metric("Target Classes", "3")
    with col4:
        missing = data.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Target Distribution")
        style_matplotlib()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = data['Diabetes_012'].value_counts().sort_index()
        labels_map = {0: "No Diabetes", 1: "Pre-Diabetic", 2: "Diabetic"}
        colors = [NEON_CYAN, NEON_ORANGE, NEON_PURPLE]
        labels = [labels_map.get(i, str(i)) for i in counts.index]
        bars = ax.bar(labels, counts.values, color=colors, width=0.5, zorder=3)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9, color='#e8f4f8', family='monospace')
        ax.set_title("CLASS DISTRIBUTION", fontsize=9, pad=12)
        ax.grid(axis='y', zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Quick Stats")
        st.markdown(f"""
        <div class="glow-panel">
            <div class="glow-panel-title">BMI Range</div>
            <div class="glow-panel-value">{data['BMI'].min():.0f} – {data['BMI'].max():.0f}</div>
            <div class="glow-panel-sub">Mean: {data['BMI'].mean():.1f}</div>
        </div>
        <div class="glow-panel">
            <div class="glow-panel-title">High Blood Pressure</div>
            <div class="glow-panel-value">{data['HighBP'].mean()*100:.1f}%</div>
            <div class="glow-panel-sub">Of total population</div>
        </div>
        <div class="glow-panel">
            <div class="glow-panel-title">Physically Active</div>
            <div class="glow-panel-value">{data['PhysActivity'].mean()*100:.1f}%</div>
            <div class="glow-panel-sub">Engaged in physical activity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Raw Data Preview")
    st.dataframe(
        data.head(20).style.background_gradient(cmap='Blues', axis=0),
        use_container_width=True, height=280
    )

    st.markdown("---")
    st.markdown("#### Feature Correlation Heatmap")
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    cols_to_show = corr.columns[:12]
    corr_subset = corr.loc[cols_to_show, cols_to_show]
    im = ax.imshow(corr_subset.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols_to_show)))
    ax.set_yticks(range(len(cols_to_show)))
    ax.set_xticklabels(cols_to_show, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(cols_to_show, fontsize=7)
    for i in range(len(cols_to_show)):
        for j in range(len(cols_to_show)):
            val = corr_subset.values[i, j]
            if abs(val) > 0.3:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color='white' if abs(val) > 0.5 else '#aaa')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("FEATURE CORRELATION MATRIX", fontsize=9, pad=14)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### MODEL TRAINING")

    if not any([use_lr, use_dt, use_rf]):
        st.warning("⚠ Select at least one model in the sidebar.")
    else:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:rgba(0,255,136,0.7); letter-spacing:0.15em; margin-bottom:1rem;">
        ⚡ MODELS LOADED FROM PRE-TRAINED FILES — NO TRAINING REQUIRED
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", f"{meta.get('train_rows', 0):,}")
        with col2:
            st.metric("Test Samples", f"{meta.get('test_rows', 0):,}")
        with col3:
            st.metric("Models Loaded", f"{len(results)}")

        st.markdown("---")
        for name, res in results.items():
            with st.expander(f"🔬 {name.upper()} — Accuracy: {res['accuracy']:.4f}", expanded=True):
                r = res['report']
                cols = st.columns(4)
                with cols[0]:
                    st.metric("Accuracy", f"{res['accuracy']:.4f}")
                with cols[1]:
                    wf1 = r.get('weighted avg', {}).get('f1-score', 0)
                    st.metric("Weighted F1", f"{wf1:.4f}")
                with cols[2]:
                    wp = r.get('weighted avg', {}).get('precision', 0)
                    st.metric("Precision", f"{wp:.4f}")
                with cols[3]:
                    wr = r.get('weighted avg', {}).get('recall', 0)
                    st.metric("Recall", f"{wr:.4f}")

        # Feature Importances
        st.markdown("---")
        st.markdown("#### Feature Importances")
        if 'Random Forest' in results:
            style_matplotlib()
            rf_model = results['Random Forest']['model']
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[-15:]
            fig, ax = plt.subplots(figsize=(10, 5))
            colors_fi = [NEON_CYAN if v > np.median(importances[indices]) else NEON_PURPLE for v in importances[indices]]
            bars = ax.barh([feature_names[i] for i in indices], importances[indices], color=colors_fi, height=0.6)
            ax.set_title("TOP 15 FEATURES · RANDOM FOREST", fontsize=9, pad=12)
            ax.grid(axis='x', zorder=0)
            ax.set_axisbelow(True)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
        elif 'Decision Tree' in results:
            style_matplotlib()
            dt_model = results['Decision Tree']['model']
            importances = dt_model.feature_importances_
            indices = np.argsort(importances)[-15:]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh([feature_names[i] for i in indices], importances[indices], color=NEON_ORANGE, height=0.6)
            ax.set_title("TOP 15 FEATURES · DECISION TREE", fontsize=9, pad=12)
            ax.grid(axis='x', zorder=0)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Feature importances available for tree-based models. Enable Decision Tree or Random Forest.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### PERFORMANCE ANALYSIS")

    if not results:
        st.warning("⚠ Select at least one model in the sidebar.")
    else:
        # Accuracy comparison across test sizes
        st.markdown("#### Accuracy vs Test Split Size")
        test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
        model_configs = []
        if use_lr: model_configs.append(('Logistic Regression', NEON_CYAN))
        if use_dt: model_configs.append(('Decision Tree', NEON_ORANGE))
        if use_rf: model_configs.append(('Random Forest', NEON_GREEN))

        # Use precomputed values from PDF
        precomputed = {
            'Logistic Regression': [0.8438, 0.8430, 0.8434, 0.8426, 0.8423],
            'Decision Tree':       [0.7701, 0.7659, 0.7662, 0.7674, 0.7673],
            'Random Forest':       [0.8432, 0.8416, 0.8415, 0.8421, 0.8419],
        }

        style_matplotlib()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Line chart
        for name, color in model_configs:
            if name in precomputed:
                ax1.plot(test_sizes, precomputed[name], color=color, linewidth=2, marker='o', markersize=6, label=name)
                ax1.fill_between(test_sizes, precomputed[name], alpha=0.05, color=color)
        ax1.set_xlabel("Test Size", fontsize=9)
        ax1.set_ylabel("Accuracy", fontsize=9)
        ax1.set_title("ACCURACY TREND", fontsize=9, pad=12)
        ax1.legend(fontsize=8, framealpha=0.2, facecolor='#040a10', edgecolor='#0a2030')
        ax1.grid(True, zorder=0)

        # Bar chart
        x = np.arange(len(test_sizes))
        width = 0.25
        bars_list = []
        for i, (name, color) in enumerate(model_configs):
            if name in precomputed:
                offset = (i - (len(model_configs)-1)/2) * width
                b = ax2.bar(x + offset, precomputed[name], width*0.9, color=color, label=name, alpha=0.85)
                bars_list.append(b)
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_sizes, fontsize=8)
        ax2.set_title("ACCURACY COMPARISON", fontsize=9, pad=12)
        ax2.set_ylabel("Accuracy", fontsize=9)
        ax2.legend(fontsize=8, framealpha=0.2, facecolor='#040a10', edgecolor='#0a2030')
        ax2.set_ylim(0.72, 0.88)
        ax2.grid(axis='y', zorder=0)
        ax2.set_axisbelow(True)

        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        # Confusion Matrices
        st.markdown("---")
        st.markdown("#### Confusion Matrices")
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, results.items()):
            cm = res['cm']
            color = res['color']
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(f"{name.upper()}\nAcc: {res['accuracy']:.3f}", fontsize=8, pad=10, color=color)
            tick_marks = np.arange(cm.shape[0])
            labels = ['No DM', 'Pre-DM', 'DM'] if cm.shape[0] == 3 else [str(i) for i in range(cm.shape[0])]
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels[:cm.shape[0]], fontsize=8)
            ax.set_yticklabels(labels[:cm.shape[0]], fontsize=8)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                            color='white' if cm[i, j] > cm.max()/2 else '#aaa')
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)

        fig.suptitle("CONFUSION MATRICES", fontsize=10, color=NEON_CYAN, y=1.02)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary table
        st.markdown("---")
        st.markdown("#### Model Comparison Table")
        table_data = []
        for name, res in results.items():
            r = res['report']
            table_data.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']:.4f}",
                "Weighted F1": f"{r.get('weighted avg',{}).get('f1-score',0):.4f}",
                "Precision": f"{r.get('weighted avg',{}).get('precision',0):.4f}",
                "Recall": f"{r.get('weighted avg',{}).get('recall',0):.4f}",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### REAL-TIME PREDICTION")

    if not results:
        st.warning("⚠ Select at least one model in the sidebar.")
    else:

        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:rgba(0,245,255,0.5); letter-spacing:0.15em; margin-bottom:1rem;">
        ◈ ENTER PATIENT PARAMETERS BELOW
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        input_data = {}
        feature_inputs = {
            'HighBP': ('High Blood Pressure', [0, 1], col1),
            'HighChol': ('High Cholesterol', [0, 1], col1),
            'CholCheck': ('Cholesterol Check (5yr)', [0, 1], col1),
            'BMI': ('BMI', None, col1),
            'Smoker': ('Smoker', [0, 1], col2),
            'Stroke': ('History of Stroke', [0, 1], col2),
            'HeartDiseaseorAttack': ('Heart Disease/Attack', [0, 1], col2),
            'PhysActivity': ('Physical Activity', [0, 1], col2),
            'Fruits': ('Fruits (daily)', [0, 1], col3),
            'Veggies': ('Vegetables (daily)', [0, 1], col3),
            'HvyAlcoholConsump': ('Heavy Alcohol Use', [0, 1], col3),
            'AnyHealthcare': ('Has Healthcare', [0, 1], col3),
            'NoDocbcCost': ('No Doc due to Cost', [0, 1], col1),
            'GenHlth': ('General Health (1-5)', None, col1),
            'MentHlth': ('Mental Health Days (0-30)', None, col2),
            'PhysHlth': ('Physical Health Days (0-30)', None, col2),
            'DiffWalk': ('Difficulty Walking', [0, 1], col3),
            'Sex': ('Sex (0=F, 1=M)', [0, 1], col3),
            'Age': ('Age Category (1-13)', None, col1),
            'Education': ('Education Level (1-6)', None, col2),
            'Income': ('Income Level (1-8)', None, col3),
        }

        for feat in feature_names:
            if feat in feature_inputs:
                label, options, column = feature_inputs[feat]
                with column:
                    if options is not None:
                        val = st.selectbox(label, options, key=f"feat_{feat}")
                    else:
                        if feat == 'BMI':
                            val = st.slider(label, 10.0, 100.0, 27.0, 0.5, key=f"feat_{feat}")
                        elif feat in ['MentHlth', 'PhysHlth']:
                            val = st.slider(label, 0, 30, 0, key=f"feat_{feat}")
                        elif feat == 'GenHlth':
                            val = st.slider(label, 1, 5, 3, key=f"feat_{feat}")
                        elif feat == 'Age':
                            val = st.slider(label, 1, 13, 7, key=f"feat_{feat}")
                        elif feat == 'Education':
                            val = st.slider(label, 1, 6, 4, key=f"feat_{feat}")
                        elif feat == 'Income':
                            val = st.slider(label, 1, 8, 5, key=f"feat_{feat}")
                        else:
                            val = st.number_input(label, value=0.0, key=f"feat_{feat}")
                input_data[feat] = val

        st.markdown("---")
        predict_btn = st.button("🔮  RUN PREDICTION", use_container_width=False)

        if predict_btn:
            input_df = pd.DataFrame([input_data])
            label_map = {0: "NO DIABETES", 1: "PRE-DIABETIC", 2: "DIABETIC"}
            class_style = {0: "low-risk", 1: "pre-diabetic", 2: "high-risk"}
            class_color = {0: "#00ff88", 1: "#ff9900", 2: "#ff3333"}

            pred_cols = st.columns(len(results))
            for col, (name, res) in zip(pred_cols, results.items()):
                pred = int(res['model'].predict(input_df)[0])
                prob = res['model'].predict_proba(input_df)[0]
                with col:
                    st.markdown(f"""
                    <div class="prediction-box {class_style[pred]}">
                        <div class="prediction-title">{name}</div>
                        <div class="prediction-result" style="color:{class_color[pred]};">
                            {label_map[pred]}
                        </div>
                        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:rgba(232,244,248,0.5); margin-top:1rem;">
                            Confidence: {prob[pred]*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("---")
            st.markdown("#### Prediction Probabilities")
            style_matplotlib()
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
            if len(results) == 1:
                axes = [axes]
            for ax, (name, res) in zip(axes, results.items()):
                prob = res['model'].predict_proba(input_df)[0]
                classes = [f"Class {i}" for i in range(len(prob))]
                class_labels = ['No DM', 'Pre-DM', 'DM'] if len(prob) == 3 else classes
                colors_p = [NEON_GREEN, NEON_ORANGE, NEON_PURPLE][:len(prob)]
                bars = ax.bar(class_labels, prob, color=colors_p[:len(prob)], width=0.5)
                for bar, p in zip(bars, prob):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{p:.3f}', ha='center', va='bottom', fontsize=9, color='#e8f4f8')
                ax.set_ylim(0, 1.15)
                ax.set_title(name.upper(), fontsize=8, color=res['color'], pad=10)
                ax.grid(axis='y', zorder=0)
                ax.set_axisbelow(True)
            fig.suptitle("PREDICTION PROBABILITY DISTRIBUTION", fontsize=9, color=NEON_CYAN)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
