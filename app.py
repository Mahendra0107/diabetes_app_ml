import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

:root {
    --cream:      #f5f0e8;
    --cream-dark: #ede7d9;
    --ink:        #1c1c1a;
    --ink-light:  #5a5a56;
    --ink-faint:  #9a9a94;
    --forest:     #2d4a3e;
    --forest-mid: #3d6b5a;
    --terra:      #c4622d;
    --terra-light:#e8845a;
    --sage:       #7a9e8e;
    --gold:       #c9973a;
    --rule:       rgba(28,28,26,0.12);
}

.stApp {
    background-color: var(--cream);
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
}
.block-container { padding: 2.5rem 3rem 3rem 3rem !important; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background-color: var(--forest) !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #7a9e8e !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(122,158,142,0.25) !important; }

h1 { font-family: 'Playfair Display', serif !important; }
h2, h3 { font-family: 'Playfair Display', serif !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid var(--rule) !important; gap: 0 !important; }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--ink-faint) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 1.8rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color: var(--forest) !important; border-bottom: 2px solid var(--terra) !important; }

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--rule);
    border-top: 3px solid var(--forest);
    padding: 1.2rem 1.4rem !important;
    border-radius: 2px;
}
[data-testid="stMetric"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--ink-faint) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    color: var(--ink) !important;
}

.stButton > button {
    background: var(--forest) !important;
    color: var(--cream) !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.8rem !important;
    border-radius: 2px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--forest-mid) !important; }

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] > label,
[data-testid="stRadio"] > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--ink-light) !important;
}

hr { border: none !important; border-top: 1px solid var(--rule) !important; margin: 1.8rem 0 !important; }

.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: white !important;
    border: 1px solid var(--rule) !important;
    color: var(--ink-light) !important;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c4622d;
    margin-bottom: 0.3rem;
}
.stat-card {
    background: white;
    border: 1px solid rgba(28,28,26,0.12);
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
}
.stat-card-label { font-family: 'DM Mono', monospace; font-size: 0.58rem; letter-spacing: 0.16em; text-transform: uppercase; color: #9a9a94; margin-bottom: 0.3rem; }
.stat-card-value { font-family: 'Playfair Display', serif; font-size: 1.9rem; color: #1c1c1a; line-height: 1; }
.stat-card-sub   { font-size: 0.78rem; color: #9a9a94; margin-top: 0.3rem; }
.result-block    { padding: 2rem; text-align: center; border: 1px solid rgba(28,28,26,0.12); background: white; }
.result-block-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: #9a9a94; margin-bottom: 0.8rem; }
.result-block-value { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 600; }
.badge { display: inline-block; font-family: 'DM Mono', monospace; font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase; padding: 0.25rem 0.7rem; border-radius: 1px; margin: 0.15rem; }
.badge-forest { background: rgba(45,74,62,0.08);  color: #2d4a3e; border: 1px solid rgba(45,74,62,0.2);  }
.badge-terra  { background: rgba(196,98,45,0.08); color: #c4622d; border: 1px solid rgba(196,98,45,0.2); }
.badge-gold   { background: rgba(201,151,58,0.1); color: #c9973a; border: 1px solid rgba(201,151,58,0.25);}
</style>
""", unsafe_allow_html=True)

C_FOREST = "#2d4a3e"
C_TERRA  = "#c4622d"
C_GOLD   = "#c9973a"
C_SAGE   = "#7a9e8e"
C_CREAM  = "#f5f0e8"
C_INK    = "#1c1c1a"
C_RULE   = "#ddd8ce"

def mpl_style():
    plt.rcParams.update({
        "figure.facecolor":   C_CREAM,
        "axes.facecolor":     "white",
        "axes.edgecolor":     C_RULE,
        "axes.labelcolor":    "#5a5a56",
        "axes.titlecolor":    C_INK,
        "xtick.color":        "#9a9a94",
        "ytick.color":        "#9a9a94",
        "text.color":         C_INK,
        "grid.color":         C_RULE,
        "grid.linewidth":     0.8,
        "font.family":        "serif",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlesize":     10,
        "axes.titleweight":   "normal",
        "axes.titlelocation": "left",
    })

@st.cache_resource
def load_models():
    models_dir = "models"
    mapping = {
        "Logistic Regression": ("logistic_regression.pkl", C_FOREST),
        "Decision Tree":       ("decision_tree.pkl",       C_TERRA),
        "Random Forest":       ("random_forest.pkl",       C_GOLD),
    }
    meta_path = os.path.join(models_dir, "metadata.pkl")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    results = {}
    for name, (fname, color) in mapping.items():
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            model = pickle.load(f)
        key = fname.replace(".pkl", "")
        rm = meta.get("results", {}).get(key, {})
        results[name] = {
            "model":    model,
            "accuracy": rm.get("accuracy", 0),
            "report":   rm.get("report", {}),
            "cm":       np.array(rm.get("cm", [[0]])),
            "color":    color,
        }
    return results, meta

@st.cache_data
def make_sample_data():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'Diabetes_012':        np.random.choice([0,1,2], n, p=[0.73,0.02,0.25]),
        'HighBP':              np.random.choice([0,1], n, p=[0.57,0.43]),
        'HighChol':            np.random.choice([0,1], n, p=[0.56,0.44]),
        'CholCheck':           np.random.choice([0,1], n, p=[0.07,0.93]),
        'BMI':                 np.clip(np.random.normal(28,7,n),12,98).round(1),
        'Smoker':              np.random.choice([0,1], n, p=[0.56,0.44]),
        'Stroke':              np.random.choice([0,1], n, p=[0.96,0.04]),
        'HeartDiseaseorAttack':np.random.choice([0,1], n, p=[0.91,0.09]),
        'PhysActivity':        np.random.choice([0,1], n, p=[0.25,0.75]),
        'Fruits':              np.random.choice([0,1], n, p=[0.37,0.63]),
        'Veggies':             np.random.choice([0,1], n, p=[0.19,0.81]),
        'HvyAlcoholConsump':   np.random.choice([0,1], n, p=[0.94,0.06]),
        'AnyHealthcare':       np.random.choice([0,1], n, p=[0.05,0.95]),
        'NoDocbcCost':         np.random.choice([0,1], n, p=[0.84,0.16]),
        'GenHlth':             np.random.choice([1,2,3,4,5], n),
        'MentHlth':            np.clip(np.random.exponential(3,n),0,30).round(),
        'PhysHlth':            np.clip(np.random.exponential(4,n),0,30).round(),
        'DiffWalk':            np.random.choice([0,1], n, p=[0.85,0.15]),
        'Sex':                 np.random.choice([0,1], n),
        'Age':                 np.random.choice(range(1,14), n),
        'Education':           np.random.choice(range(1,7), n),
        'Income':              np.random.choice(range(1,9), n),
    })

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem 0; border-bottom:1px solid rgba(122,158,142,0.25); margin-bottom:1rem;">
        <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.22em; text-transform:uppercase; color:#7a9e8e; margin-bottom:0.4rem;">CDC · BRFSS 2015</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.15rem; color:#f5f0e8; line-height:1.3;">Glycemic Risk<br>Assessment</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Data source")
    data_source = st.radio("", ["Sample data", "Upload CSV"], label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("#### Models")
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    use_rf = st.checkbox("Random Forest", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.14em; line-height:2.2; color:rgba(245,240,232,0.35);">
    253,680 RECORDS<br>21 FEATURES<br>3 TARGET CLASSES<br>PRE-TRAINED MODELS
    </div>
    """, unsafe_allow_html=True)

# ── Load ─────────────────────────────────────────────────────────────────────
all_models, meta = load_models()
if all_models is None:
    st.error("No pre-trained models found. Run `python train_and_save.py` locally, then push the `models/` folder to GitHub.")
    st.stop()

visibility = {"Logistic Regression": use_lr, "Decision Tree": use_dt, "Random Forest": use_rf}
results       = {k: v for k, v in all_models.items() if visibility.get(k)}
feature_names = meta.get("feature_names", [])

if data_source == "Upload CSV" and uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'Diabetes_012' not in data.columns:
        st.error("CSV must contain a 'Diabetes_012' column.")
        st.stop()
else:
    data = make_sample_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid rgba(28,28,26,0.12); padding-bottom:2rem; margin-bottom:2.5rem;">
    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; letter-spacing:0.22em; text-transform:uppercase; color:#c4622d; margin-bottom:0.8rem;">
        Predictive Health Analytics · Multi-Model Classification
    </div>
    <h1 style="font-family:'Playfair Display',serif; font-size:clamp(2rem,5vw,3.4rem); font-weight:400; color:#1c1c1a; margin:0; line-height:1.1;">
        Glycemic Risk<br><em style="font-weight:400; color:#2d4a3e;">Assessment Tool</em>
    </h1>
    <div style="margin-top:1.2rem; display:flex; gap:0.4rem; flex-wrap:wrap; align-items:center;">
        <span class="badge badge-forest">Logistic Regression</span>
        <span class="badge badge-terra">Decision Tree</span>
        <span class="badge badge-gold">Random Forest</span>
        <span style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#9a9a94; margin-left:0.5rem;">CDC BRFSS 2015 Dataset</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Models", "Performance", "Predict"])

# TAB 1
with tab1:
    st.markdown("<div class='section-label'>Dataset summary</div>", unsafe_allow_html=True)
    st.markdown("### Exploring the Data")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Records",       f"{len(data):,}")
    with c2: st.metric("Features",      f"{data.shape[1]-1}")
    with c3: st.metric("Target classes","3")
    with c4: st.metric("Missing values",f"{data.isnull().sum().sum():,}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("<div class='section-label'>Class distribution</div>", unsafe_allow_html=True)
        mpl_style()
        fig, ax = plt.subplots(figsize=(8, 3.8))
        counts = data['Diabetes_012'].value_counts().sort_index()
        lmap   = {0:"No Diabetes", 1:"Pre-Diabetic", 2:"Diabetic"}
        clrs   = [C_SAGE, C_GOLD, C_TERRA]
        lbl    = [lmap.get(i, str(i)) for i in counts.index]
        bars   = ax.bar(lbl, counts.values, color=clrs[:len(counts)], width=0.45, zorder=3)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.012,
                    f'{val:,}', ha='center', va='bottom', fontsize=8.5, color=C_INK)
        ax.set_title("Distribution of target classes", pad=14)
        ax.grid(axis='y', zorder=0, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close()

    with col_r:
        st.markdown("<div class='section-label'>Key indicators</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-label">Average BMI</div>
            <div class="stat-card-value">{data['BMI'].mean():.1f}</div>
            <div class="stat-card-sub">Range: {data['BMI'].min():.0f} – {data['BMI'].max():.0f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">High Blood Pressure</div>
            <div class="stat-card-value">{data['HighBP'].mean()*100:.1f}%</div>
            <div class="stat-card-sub">of surveyed population</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-label">Physically Active</div>
            <div class="stat-card-value">{data['PhysActivity'].mean()*100:.1f}%</div>
            <div class="stat-card-sub">report regular activity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Raw records</div>", unsafe_allow_html=True)
    st.markdown("### Data Preview")
    st.dataframe(data.head(15), use_container_width=True, height=260)

    st.markdown("---")
    st.markdown("<div class='section-label'>Correlation analysis</div>", unsafe_allow_html=True)
    st.markdown("### Feature Correlations")
    mpl_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    numeric = data.select_dtypes(include=[np.number])
    corr    = numeric.corr()
    cols12  = corr.columns[:12]
    sub     = corr.loc[cols12, cols12]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("c", [C_TERRA, "white", C_FOREST], N=256)
    im = ax.imshow(sub.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols12))); ax.set_yticks(range(len(cols12)))
    ax.set_xticklabels(cols12, rotation=40, ha='right', fontsize=7)
    ax.set_yticklabels(cols12, fontsize=7)
    for i in range(len(cols12)):
        for j in range(len(cols12)):
            v = sub.values[i,j]
            if abs(v) > 0.25:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if abs(v) > 0.55 else C_INK)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    ax.set_title("Pearson correlation — first 12 features", pad=14)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

# TAB 2
with tab2:
    st.markdown("<div class='section-label'>Pre-trained classifiers</div>", unsafe_allow_html=True)
    st.markdown("### Model Summary")

    if not results:
        st.warning("Select at least one model in the sidebar.")
    else:
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif; font-size:0.88rem; color:#5a5a56;
                    background:white; border:1px solid rgba(28,28,26,0.1);
                    border-left:3px solid #2d4a3e; padding:1rem 1.4rem; margin-bottom:1.5rem;">
            Models were trained on the full 253,680-record CDC BRFSS 2015 dataset
            (80/20 train-test split, random state 42) and saved before deployment.
            No training occurs at runtime.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Training records", f"{meta.get('train_rows',0):,}")
        with c2: st.metric("Test records",     f"{meta.get('test_rows',0):,}")
        with c3: st.metric("Models active",    f"{len(results)}")

        st.markdown("---")
        for name, res in results.items():
            r   = res['report']
            wf1 = r.get('weighted avg',{}).get('f1-score',0)
            wp  = r.get('weighted avg',{}).get('precision',0)
            wr  = r.get('weighted avg',{}).get('recall',0)
            with st.expander(f"{name}   —   Accuracy {res['accuracy']:.4f}", expanded=True):
                mc1,mc2,mc3,mc4 = st.columns(4)
                with mc1: st.metric("Accuracy",    f"{res['accuracy']:.4f}")
                with mc2: st.metric("Weighted F1", f"{wf1:.4f}")
                with mc3: st.metric("Precision",   f"{wp:.4f}")
                with mc4: st.metric("Recall",      f"{wr:.4f}")

        st.markdown("---")
        st.markdown("<div class='section-label'>Variable importance</div>", unsafe_allow_html=True)
        st.markdown("### Feature Importances")
        tree_model, tree_name = None, None
        for n in ["Random Forest","Decision Tree"]:
            if n in results:
                tree_model = results[n]['model']
                tree_name  = n
                break
        if tree_model and feature_names:
            imps    = tree_model.feature_importances_
            indices = np.argsort(imps)[-15:]
            mpl_style()
            fig, ax = plt.subplots(figsize=(9, 4.5))
            vals    = imps[indices]
            bclrs   = [C_FOREST if v >= np.median(vals) else C_SAGE for v in vals]
            ax.barh([feature_names[i] for i in indices], vals, color=bclrs, height=0.55)
            ax.set_title(f"Top 15 features by importance · {tree_name}", pad=14)
            ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis='y', labelsize=8.5)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Enable Decision Tree or Random Forest to see feature importances.")

# TAB 3
with tab3:
    st.markdown("<div class='section-label'>Evaluation metrics</div>", unsafe_allow_html=True)
    st.markdown("### Performance Analysis")

    if not results:
        st.warning("Select at least one model in the sidebar.")
    else:
        precomputed = {
            'Logistic Regression': [0.8438, 0.8430, 0.8434, 0.8426, 0.8423],
            'Decision Tree':       [0.7701, 0.7659, 0.7662, 0.7674, 0.7673],
            'Random Forest':       [0.8432, 0.8416, 0.8415, 0.8421, 0.8419],
        }
        test_sizes   = [0.1, 0.2, 0.3, 0.4, 0.5]
        model_colors = {'Logistic Regression': C_FOREST, 'Decision Tree': C_TERRA, 'Random Forest': C_GOLD}

        mpl_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for name in results:
            if name in precomputed:
                ax1.plot(test_sizes, precomputed[name], color=model_colors[name],
                         linewidth=1.8, marker='o', markersize=5, label=name)
                ax1.fill_between(test_sizes, precomputed[name], alpha=0.06, color=model_colors[name])
        ax1.set_xlabel("Test split ratio", fontsize=9)
        ax1.set_ylabel("Accuracy", fontsize=9)
        ax1.set_title("Accuracy across test split sizes", pad=14)
        ax1.legend(fontsize=8, framealpha=0.6, facecolor='white', edgecolor=C_RULE)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        x = np.arange(len(test_sizes))
        w = 0.22
        active = [n for n in results if n in precomputed]
        for i, name in enumerate(active):
            offset = (i - (len(active)-1)/2) * w
            ax2.bar(x + offset, precomputed[name], w*0.88, color=model_colors[name], label=name, alpha=0.88)
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_sizes, fontsize=8)
        ax2.set_title("Side-by-side accuracy comparison", pad=14)
        ax2.set_ylabel("Accuracy", fontsize=9)
        ax2.set_ylim(0.72, 0.88)
        ax2.legend(fontsize=8, framealpha=0.6, facecolor='white', edgecolor=C_RULE)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)
        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("<div class='section-label'>Confusion matrices</div>", unsafe_allow_html=True)
        st.markdown("### Prediction vs. Actual")
        n_m  = len(results)
        fig, axes = plt.subplots(1, n_m, figsize=(5.5*n_m, 4.5))
        if n_m == 1: axes = [axes]
        for ax, (name, res) in zip(axes, results.items()):
            cm  = res['cm']
            ax.imshow(cm, cmap=plt.get_cmap('YlGn'), aspect='auto')
            lbl = ['No DM','Pre-DM','DM'] if cm.shape[0]==3 else [str(i) for i in range(cm.shape[0])]
            tks = np.arange(cm.shape[0])
            ax.set_xticks(tks); ax.set_xticklabels(lbl[:cm.shape[0]], fontsize=8)
            ax.set_yticks(tks); ax.set_yticklabels(lbl[:cm.shape[0]], fontsize=8)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=9,
                            color='white' if cm[i,j] > cm.max()*0.55 else C_INK)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual",    fontsize=8)
            ax.set_title(f"{name}\nAccuracy {res['accuracy']:.3f}", pad=10, color=res['color'])
        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("<div class='section-label'>Comparison table</div>", unsafe_allow_html=True)
        st.markdown("### All Models at a Glance")
        rows = []
        for name, res in results.items():
            r = res['report']
            rows.append({
                "Model":       name,
                "Accuracy":    f"{res['accuracy']:.4f}",
                "Weighted F1": f"{r.get('weighted avg',{}).get('f1-score',0):.4f}",
                "Precision":   f"{r.get('weighted avg',{}).get('precision',0):.4f}",
                "Recall":      f"{r.get('weighted avg',{}).get('recall',0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# TAB 4
with tab4:
    st.markdown("<div class='section-label'>Individual risk assessment</div>", unsafe_allow_html=True)
    st.markdown("### Enter Patient Parameters")
    st.markdown("<p style='font-size:0.88rem; color:#5a5a56; max-width:600px; margin-bottom:1.5rem;'>Adjust the indicators below. Each model will independently calculate a risk classification.</p>", unsafe_allow_html=True)

    if not results:
        st.warning("Select at least one model in the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)
        input_data = {}
        fields = {
            'HighBP':               ('High Blood Pressure',       [0,1],  c1),
            'HighChol':             ('High Cholesterol',           [0,1],  c1),
            'CholCheck':            ('Cholesterol Check (5 yr)',   [0,1],  c1),
            'BMI':                  ('Body Mass Index',            None,   c1),
            'Smoker':               ('Smoker',                     [0,1],  c2),
            'Stroke':               ('History of Stroke',          [0,1],  c2),
            'HeartDiseaseorAttack': ('Heart Disease / Attack',     [0,1],  c2),
            'PhysActivity':         ('Physically Active',          [0,1],  c2),
            'Fruits':               ('Daily Fruit Intake',         [0,1],  c3),
            'Veggies':              ('Daily Vegetable Intake',     [0,1],  c3),
            'HvyAlcoholConsump':    ('Heavy Alcohol Use',          [0,1],  c3),
            'AnyHealthcare':        ('Has Healthcare Coverage',    [0,1],  c3),
            'NoDocbcCost':          ('Skipped Doctor (Cost)',      [0,1],  c1),
            'GenHlth':              ('General Health (1-5)',       None,   c1),
            'MentHlth':             ('Mental Health Days (0-30)',  None,   c2),
            'PhysHlth':             ('Physical Health Days (0-30)',None,   c2),
            'DiffWalk':             ('Difficulty Walking',         [0,1],  c3),
            'Sex':                  ('Sex (0=Female, 1=Male)',     [0,1],  c3),
            'Age':                  ('Age Category (1-13)',        None,   c1),
            'Education':            ('Education Level (1-6)',      None,   c2),
            'Income':               ('Income Level (1-8)',         None,   c3),
        }
        slider_ranges = {
            'BMI':      (10.0,98.0,27.0,0.5),
            'MentHlth': (0,30,0,1),
            'PhysHlth': (0,30,0,1),
            'GenHlth':  (1,5,3,1),
            'Age':      (1,13,7,1),
            'Education':(1,6,4,1),
            'Income':   (1,8,5,1),
        }
        for feat in feature_names:
            if feat not in fields: continue
            label, opts, col = fields[feat]
            with col:
                if opts is not None:
                    val = st.selectbox(label, opts, key=f"p_{feat}")
                elif feat in slider_ranges:
                    mn,mx,dv,st_ = slider_ranges[feat]
                    val = st.slider(label, mn, mx, dv, st_, key=f"p_{feat}")
                else:
                    val = st.number_input(label, value=0.0, key=f"p_{feat}")
            input_data[feat] = val

        st.markdown("---")
        go = st.button("Calculate Risk")

        if go:
            inp_df    = pd.DataFrame([input_data])
            label_map = {0:"No Diabetes", 1:"Pre-Diabetic", 2:"Diabetic"}
            color_map = {0:C_FOREST, 1:C_GOLD, 2:C_TERRA}

            st.markdown("<div class='section-label' style='margin-top:1rem;'>Classification results</div>", unsafe_allow_html=True)
            st.markdown("### Model Predictions")
            rcols = st.columns(len(results))
            for col, (name, res) in zip(rcols, results.items()):
                pred  = int(res['model'].predict(inp_df)[0])
                probs = res['model'].predict_proba(inp_df)[0]
                clr   = color_map[pred]
                with col:
                    st.markdown(f"""
                    <div class="result-block" style="border-top:3px solid {clr};">
                        <div class="result-block-label">{name}</div>
                        <div class="result-block-value" style="color:{clr};">{label_map[pred]}</div>
                        <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#9a9a94; margin-top:0.7rem; letter-spacing:0.1em;">
                            {probs[pred]*100:.1f}% confidence
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>Probability breakdown</div>", unsafe_allow_html=True)
            mpl_style()
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 3.8))
            if len(results) == 1: axes = [axes]
            for ax, (name, res) in zip(axes, results.items()):
                probs  = res['model'].predict_proba(inp_df)[0]
                clbls  = ['No Diabetes','Pre-Diabetic','Diabetic'][:len(probs)]
                bclrs  = [C_SAGE, C_GOLD, C_TERRA][:len(probs)]
                bars   = ax.bar(clbls, probs, color=bclrs, width=0.45)
                for bar, p in zip(bars, probs):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                            f'{p:.3f}', ha='center', va='bottom', fontsize=8.5)
                ax.set_ylim(0, 1.15)
                ax.set_title(name, pad=12, color=res['color'])
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_axisbelow(True)
                ax.tick_params(axis='x', labelsize=8)
            fig.suptitle("Predicted class probabilities", fontsize=9, y=1.02, color=C_INK)
            fig.tight_layout(pad=1.8)
            st.pyplot(fig)
            plt.close()
