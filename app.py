import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Mining · UAS 2025/2026",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
    color: #8b949e !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Main background ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    color: #39d353 !important;
}

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid #21262d !important;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Buttons ── */
.stButton > button {
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    transition: background 0.15s;
}
.stButton > button:hover {
    background: #2ea043;
}

/* ── Sliders ── */
.stSlider [data-testid="stTickBar"] { color: #8b949e; }

/* ── Section headers ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #39d353;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 0;
}

/* ── Info/warning/error overrides ── */
.stAlert {
    border-radius: 8px;
    border-left: 3px solid;
}

/* ── Number inputs ── */
input[type=number] {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
}

/* ── Divider ── */
hr { border-color: #21262d; }

/* ── Plot backgrounds via matplotlib rcparams (set in code) ── */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB THEME (dark)
# ─────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
BORDER    = "#21262d"
GREEN     = "#39d353"
BLUE      = "#58a6ff"
PURPLE    = "#bc8cff"
ORANGE    = "#e3b341"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
PALETTE   = [GREEN, BLUE, PURPLE, ORANGE, "#ff7b72", "#ffa657"]

def apply_dark_style(ax, title=""):
    ax.set_facecolor(CARD_BG)
    ax.figure.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight='600', pad=12)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 8px 0;'>
        <div style='font-family: JetBrains Mono, monospace; font-size:0.65rem;
                    color:#39d353; letter-spacing:0.12em; text-transform:uppercase;'>
            Data Mining · UAS
        </div>
        <div style='font-size:1.1rem; font-weight:600; color:#e6edf3; margin-top:4px;'>
            SIF304 · 2025/2026
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    menu = st.selectbox(
        "NAVIGASI",
        ["🏠  Home", "🩺  Klasifikasi Diabetes", "☕  Clustering Gerai Kopi"]
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; line-height:1.7;'>
        <b style='color:#e6edf3;'>Algoritma yang digunakan</b><br>
        · KNN Classifier<br>
        · Naive Bayes<br>
        · Decision Tree<br>
        · K-Means Clustering
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if "Home" in menu:
    st.markdown("""
    <div style='padding: 48px 0 24px 0;'>
        <div style='font-family: JetBrains Mono, monospace; font-size:0.72rem;
                    color:#39d353; letter-spacing:0.14em; text-transform:uppercase;
                    margin-bottom:10px;'>
            Ujian Akhir Semester · Data Mining
        </div>
        <h1 style='font-size:2.4rem; font-weight:700; color:#e6edf3;
                   margin:0 0 12px 0; line-height:1.2;'>
            Supervised &amp;<br>Unsupervised Learning
        </h1>
        <p style='color:#8b949e; max-width:520px; font-size:1rem; line-height:1.7;'>
            Implementasi algoritma klasifikasi dan clustering pada dua studi kasus nyata —
            prediksi risiko diabetes dan analisis persebaran gerai kopi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                    padding:24px; height:100%;'>
            <div style='font-family: JetBrains Mono, monospace; font-size:0.65rem;
                        color:#39d353; letter-spacing:0.1em; text-transform:uppercase;
                        margin-bottom:8px;'>Bagian A</div>
            <div style='font-size:1.15rem; font-weight:600; color:#e6edf3;
                        margin-bottom:10px;'>🩺 Klasifikasi Diabetes</div>
            <p style='color:#8b949e; font-size:0.88rem; line-height:1.6; margin:0;'>
                Prediksi risiko diabetes menggunakan tiga algoritma — KNN, Naive Bayes,
                dan Decision Tree — pada dataset Pima Indians Diabetes.
            </p>
            <div style='margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#1f2937; color:#58a6ff; font-size:0.72rem;
                             padding:3px 10px; border-radius:20px;'>KNN</span>
                <span style='background:#1f2937; color:#bc8cff; font-size:0.72rem;
                             padding:3px 10px; border-radius:20px;'>Naive Bayes</span>
                <span style='background:#1f2937; color:#e3b341; font-size:0.72rem;
                             padding:3px 10px; border-radius:20px;'>Decision Tree</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #21262d; border-radius:10px;
                    padding:24px; height:100%;'>
            <div style='font-family: JetBrains Mono, monospace; font-size:0.65rem;
                        color:#39d353; letter-spacing:0.1em; text-transform:uppercase;
                        margin-bottom:8px;'>Bagian B</div>
            <div style='font-size:1.15rem; font-weight:600; color:#e6edf3;
                        margin-bottom:10px;'>☕ Clustering Gerai Kopi</div>
            <p style='color:#8b949e; font-size:0.88rem; line-height:1.6; margin:0;'>
                Analisis klaster lokasi gerai kopi dan deteksi zona sepi berdasarkan
                koordinat geografis dan parameter lingkungan menggunakan K-Means.
            </p>
            <div style='margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;'>
                <span style='background:#1f2937; color:#39d353; font-size:0.72rem;
                             padding:3px 10px; border-radius:20px;'>K-Means</span>
                <span style='background:#1f2937; color:#ff7b72; font-size:0.72rem;
                             padding:3px 10px; border-radius:20px;'>Deteksi Zona Sepi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#8b949e; line-height:1.8;'>
        <b style='color:#e6edf3;'>Informasi Proyek</b><br>
        Mata Kuliah: Data Mining (SIF304) &nbsp;·&nbsp;
        Dosen: Teuku Rizky Noviandy, S.Kom., M.Kom. &nbsp;·&nbsp;
        Tahun Ajaran: Genap 2025/2026
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BAGIAN A: KLASIFIKASI DIABETES
# ─────────────────────────────────────────────
elif "Diabetes" in menu:
    st.markdown("""
    <div style='padding: 32px 0 8px 0;'>
        <div class='section-label'>Bagian A · Supervised Learning</div>
        <h2 class='section-title'>🩺 Klasifikasi Diabetes</h2>
        <p style='color:#8b949e; margin-top:6px; font-size:0.9rem;'>
            Prediksi risiko diabetes berdasarkan data diagnostik menggunakan tiga algoritma klasifikasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset loading (FIX: file_uploader OUTSIDE cache) ──────────────
    uploaded_file = st.file_uploader("Upload dataset diabetes (CSV)", type=['csv'],
                                      key="diabetes_upload")

    @st.cache_data
    def load_default_diabetes():
        url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
               "pima-indians-diabetes.data.csv")
        cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        return pd.read_csv(url, names=cols)

    if uploaded_file is not None:
        try:
            df_diabetes = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()
    else:
        try:
            df_diabetes = load_default_diabetes()
        except Exception:
            st.error("Gagal memuat dataset default. Upload file CSV secara manual.")
            st.stop()

    # ── Dataset preview ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sampel", df_diabetes.shape[0])
    m2.metric("Fitur", df_diabetes.shape[1] - 1)
    m3.metric("Positif Diabetes", int(df_diabetes['Outcome'].sum()))
    m4.metric("Negatif Diabetes", int((df_diabetes['Outcome'] == 0).sum()))

    with st.expander("Lihat dataset & statistik deskriptif", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_diabetes.head(10), use_container_width=True)
        with c2:
            st.dataframe(df_diabetes.describe().T.round(2), use_container_width=True)

    # ── Model building ───────────────────────────────────────────────────
    X = df_diabetes.drop('Outcome', axis=1)
    y = df_diabetes['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    st.markdown("---")
    st.markdown('<div class="section-label">Konfigurasi Model</div>', unsafe_allow_html=True)
    k_value = st.slider("Nilai K untuk KNN", 1, 21, 5, step=2)

    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train_sc, y_train)
    y_pred_knn = knn.predict(X_test_sc)

    nb = GaussianNB()
    nb.fit(X_train_sc, y_train)
    y_pred_nb = nb.predict(X_test_sc)

    dt = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    all_preds  = {'KNN': y_pred_knn, 'Naive Bayes': y_pred_nb, 'Decision Tree': y_pred_dt}
    all_models = {'KNN': knn,        'Naive Bayes': nb,         'Decision Tree': dt}

    metrics = {}
    for name, yp in all_preds.items():
        metrics[name] = {
            'Accuracy':  round(accuracy_score(y_test, yp),  4),
            'Precision': round(precision_score(y_test, yp), 4),
            'Recall':    round(recall_score(y_test, yp),    4),
            'F1-Score':  round(f1_score(y_test, yp),        4),
        }
    metrics_df = pd.DataFrame(metrics).T

    # ── Metrics display ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Evaluasi Model</div>', unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
            .format("{:.4f}")
            .highlight_max(axis=0, color="#1a3a2a")
            .set_properties(**{'font-family': 'JetBrains Mono, monospace',
                               'font-size': '0.85rem'}),
        use_container_width=True
    )

    # ── Chart: Accuracy + F1 comparison ─────────────────────────────────
    c1, c2 = st.columns(2)
    bar_colors = [GREEN, BLUE, PURPLE]

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(metrics_df.index, metrics_df['Accuracy'], color=bar_colors,
                      width=0.5, zorder=3)
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    color=TEXT, fontsize=8.5,
                    fontfamily='JetBrains Mono')
        apply_dark_style(ax, "Accuracy")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(metrics_df.index, metrics_df['F1-Score'], color=bar_colors,
                      width=0.5, zorder=3)
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    color=TEXT, fontsize=8.5,
                    fontfamily='JetBrains Mono')
        apply_dark_style(ax, "F1-Score")
        plt.tight_layout()
        st.pyplot(fig)

    # ── Confusion Matrix ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_choice = st.radio("Tampilkan:", ['Semua Model', 'KNN', 'Naive Bayes', 'Decision Tree'],
                          horizontal=True)

    if cm_choice == 'Semua Model':
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.set_facecolor(DARK_BG)
        for idx, (name, yp) in enumerate(all_preds.items()):
            cm = confusion_matrix(y_test, yp)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                        linewidths=0.5, linecolor=BORDER,
                        cbar=False, annot_kws={'size': 13, 'color': TEXT})
            axes[idx].set_facecolor(CARD_BG)
            axes[idx].set_title(name, color=TEXT, fontsize=11, fontweight='600')
            axes[idx].set_xlabel('Predicted', color=MUTED, fontsize=9)
            axes[idx].set_ylabel('Actual', color=MUTED, fontsize=9)
            axes[idx].tick_params(colors=MUTED)
            axes[idx].set_xticklabels(['Negatif', 'Positif'], color=MUTED)
            axes[idx].set_yticklabels(['Negatif', 'Positif'], color=MUTED, rotation=0)
        plt.tight_layout(pad=2)
        st.pyplot(fig)
    else:
        yp = all_preds[cm_choice]
        cm = confusion_matrix(y_test, yp)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                    linewidths=0.5, linecolor=BORDER, cbar=False,
                    annot_kws={'size': 16, 'color': TEXT})
        ax.set_title(cm_choice, color=TEXT, fontsize=12, fontweight='600')
        ax.set_xlabel('Predicted', color=MUTED)
        ax.set_ylabel('Actual', color=MUTED)
        ax.tick_params(colors=MUTED)
        ax.set_xticklabels(['Negatif', 'Positif'], color=MUTED)
        ax.set_yticklabels(['Negatif', 'Positif'], color=MUTED, rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Prediksi pasien baru ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Prediksi Pasien Baru</div>', unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:#8b949e; font-size:0.85rem;'>"
        f"Model terbaik berdasarkan Accuracy: "
        f"<b style='color:{GREEN};'>{metrics_df['Accuracy'].idxmax()}</b> "
        f"({metrics_df['Accuracy'].max():.4f})</p>",
        unsafe_allow_html=True
    )

    with st.form("predict_diabetes"):
        c1, c2, c3, c4 = st.columns(4)
        pregnancies    = c1.number_input("Pregnancies",                0,  20, 1)
        glucose        = c2.number_input("Glucose",                    0, 200, 100)
        blood_pressure = c3.number_input("Blood Pressure",             0, 150, 70)
        skin_thickness = c4.number_input("Skin Thickness",             0, 100, 20)
        insulin        = c1.number_input("Insulin",                    0, 900, 50)
        bmi            = c2.number_input("BMI",                      0.0, 70.0, 25.0)
        pedigree       = c3.number_input("Diabetes Pedigree Function",0.0,  3.0,  0.5)
        age            = c4.number_input("Age",                        0, 100, 30)
        submitted = st.form_submit_button("▶ Prediksi")

    if submitted:
        best_name  = metrics_df['Accuracy'].idxmax()
        best_model = all_models[best_name]
        inp = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, pedigree, age]])
        if best_name in ['KNN', 'Naive Bayes']:
            pred = best_model.predict(scaler.transform(inp))[0]
        else:
            pred = best_model.predict(inp)[0]

        if pred == 1:
            st.error("⚠️ **Pasien diprediksi MENGIDAP DIABETES** — Disarankan untuk konsultasi lebih lanjut.")
        else:
            st.success("✅ **Pasien diprediksi TIDAK mengidap diabetes** — Tetap jaga pola hidup sehat.")
        st.caption(f"Model: {best_name} · Accuracy: {metrics_df.loc[best_name, 'Accuracy']:.4f}")


# ─────────────────────────────────────────────
# BAGIAN B: CLUSTERING GERAI KOPI
# ─────────────────────────────────────────────
elif "Kopi" in menu:
    st.markdown("""
    <div style='padding: 32px 0 8px 0;'>
        <div class='section-label'>Bagian B · Unsupervised Learning</div>
        <h2 class='section-title'>☕ Clustering Gerai Kopi</h2>
        <p style='color:#8b949e; margin-top:6px; font-size:0.9rem;'>
            Analisis persebaran gerai kopi dan deteksi zona sepi menggunakan K-Means Clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load/generate data ────────────────────────────────────────────────
    uploaded_coffee = st.file_uploader("Upload dataset gerai kopi (CSV) — opsional", type=['csv'],
                                        key="coffee_upload")

    @st.cache_data
    def generate_coffee_data():
        np.random.seed(42)
        n_per = 67          # 67 × 3 = 201, semua kolom pakai len(lat)
        lat = np.concatenate([
            np.random.normal(-6.200, 0.05, n_per),
            np.random.normal(-6.150, 0.05, n_per),
            np.random.normal(-6.250, 0.05, n_per),
        ])
        lon = np.concatenate([
            np.random.normal(106.800, 0.05, n_per),
            np.random.normal(106.850, 0.05, n_per),
            np.random.normal(106.750, 0.05, n_per),
        ])
        n = len(lat)        # pakai panjang aktual, bukan angka hardcode
        return pd.DataFrame({
            'Latitude':          lat,
            'Longitude':         lon,
            'Customer_Density':  np.random.uniform(10, 100, n),
            'Competition_Level': np.random.uniform(1,  10,  n),
            'Distance_to_Center':np.random.uniform(0.5, 10, n),
        })

    df_coffee = pd.read_csv(uploaded_coffee) if uploaded_coffee else generate_coffee_data()

    # ── Dataset info ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Gerai", df_coffee.shape[0])
    m2.metric("Fitur",       df_coffee.shape[1])
    if 'Customer_Density' in df_coffee.columns:
        m3.metric("Rata-rata Customer Density", f"{df_coffee['Customer_Density'].mean():.1f}")

    with st.expander("Lihat data mentah", expanded=False):
        st.dataframe(df_coffee.head(10), use_container_width=True)

    # ── Feature selection + K ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Konfigurasi Clustering</div>', unsafe_allow_html=True)

    feat_opts = ['Latitude', 'Longitude', 'Customer_Density',
                 'Competition_Level', 'Distance_to_Center']
    avail = [f for f in feat_opts if f in df_coffee.columns]
    default_feats = ['Latitude', 'Longitude'] if 'Latitude' in avail else avail[:2]

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_features = st.multiselect("Fitur untuk clustering", avail,
                                            default=default_feats)
    with c2:
        n_clusters = st.slider("Jumlah Klaster (K)", 2, 8, 3)

    if len(selected_features) < 2:
        st.warning("Pilih minimal 2 fitur untuk melakukan clustering.")
        st.stop()

    # ── Run K-Means ──────────────────────────────────────────────────────
    X_c = df_coffee[selected_features].copy()
    sc  = StandardScaler()
    X_scaled = sc.fit_transform(X_c)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_coffee['Cluster'] = kmeans.fit_predict(X_scaled)

    # ── Visualisasi scatter ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Visualisasi Klaster</div>', unsafe_allow_html=True)

    use_geo = 'Latitude' in selected_features and 'Longitude' in selected_features
    x_col = 'Longitude' if use_geo else selected_features[0]
    y_col = 'Latitude'  if use_geo else selected_features[1]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)

    for k in range(n_clusters):
        mask = df_coffee['Cluster'] == k
        ax.scatter(df_coffee.loc[mask, x_col], df_coffee.loc[mask, y_col],
                   color=PALETTE[k % len(PALETTE)], alpha=0.7, s=45,
                   edgecolors='none', label=f'Klaster {k}', zorder=3)

    # centroids
    if use_geo:
        cents_orig = sc.inverse_transform(kmeans.cluster_centers_)
        feat_list  = selected_features
        lon_idx    = feat_list.index('Longitude')
        lat_idx    = feat_list.index('Latitude')
        ax.scatter(cents_orig[:, lon_idx], cents_orig[:, lat_idx],
                   color='white', marker='X', s=150, zorder=5,
                   edgecolors=BORDER, linewidth=0.8, label='Centroid')

    ax.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.set_xlabel(x_col, color=MUTED, fontsize=10)
    ax.set_ylabel(y_col, color=MUTED, fontsize=10)
    title = "Peta Sebaran Klaster Gerai Kopi" if use_geo else "Klaster Berdasarkan Fitur"
    apply_dark_style(ax, title)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.4, alpha=0.5)
    ax.xaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Deteksi zona sepi ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Deteksi Zona Sepi</div>', unsafe_allow_html=True)

    quiet_zones = []  # FIX: always initialise before use

    if 'Customer_Density' in df_coffee.columns and 'Competition_Level' in df_coffee.columns:
        cluster_stats = df_coffee.groupby('Cluster')[
            ['Customer_Density', 'Competition_Level']].mean().round(2)
        cluster_stats['Status'] = '✅ Zona Ramai'

        cd_med = cluster_stats['Customer_Density'].median()
        cl_med = cluster_stats['Competition_Level'].median()
        mask_sepi = ((cluster_stats['Customer_Density']  < cd_med) &
                     (cluster_stats['Competition_Level'] > cl_med))
        cluster_stats.loc[mask_sepi, 'Status'] = '⚠️ Zona Sepi'
        quiet_zones = cluster_stats[mask_sepi].index.tolist()

        st.dataframe(
            cluster_stats.style
                .apply(lambda row: [
                    'color: #ff7b72' if row['Status'] == '⚠️ Zona Sepi' else 'color: #39d353'
                ] * len(row), axis=1)
                .set_properties(**{'font-family': 'JetBrains Mono, monospace',
                                   'font-size': '0.85rem'}),
            use_container_width=True
        )

        if quiet_zones:
            st.error(f"⚠️ **Zona Sepi terdeteksi di Klaster:** {quiet_zones} — "
                      "Kepadatan pelanggan rendah dengan kompetisi tinggi.")
        else:
            st.success("✅ Tidak ada zona sepi yang terdeteksi.")
    else:
        st.info("Tambahkan kolom `Customer_Density` dan `Competition_Level` "
                "pada dataset untuk deteksi zona sepi.")

    # ── Distribusi + boxplot ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Distribusi Klaster</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        counts = df_coffee['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(counts.index.astype(str),
                      counts.values,
                      color=[PALETTE[k % len(PALETTE)] for k in counts.index],
                      width=0.55, zorder=3)
        ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Klaster", color=MUTED)
        ax.set_ylabel("Jumlah Gerai", color=MUTED)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    int(bar.get_height()), ha='center', color=TEXT, fontsize=9,
                    fontfamily='JetBrains Mono')
        apply_dark_style(ax, "Jumlah Gerai per Klaster")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        if 'Customer_Density' in df_coffee.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.set_facecolor(DARK_BG)
            ax.set_facecolor(CARD_BG)
            groups = [df_coffee.loc[df_coffee['Cluster'] == k, 'Customer_Density'].values
                      for k in sorted(df_coffee['Cluster'].unique())]
            bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                            medianprops=dict(color=GREEN, linewidth=2),
                            boxprops=dict(linewidth=0),
                            whiskerprops=dict(color=MUTED),
                            capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', color=MUTED, markersize=3))
            for patch, color in zip(bp['boxes'], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xticklabels([f"K{k}" for k in sorted(df_coffee['Cluster'].unique())])
            apply_dark_style(ax, "Customer Density per Klaster")
            ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)

    # ── Prediksi klaster baru ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">Prediksi Lokasi Baru</div>', unsafe_allow_html=True)

    with st.form("predict_cluster"):
        input_values = {}
        cols = st.columns(len(selected_features))
        defaults = {
            'Latitude': -6.2, 'Longitude': 106.8,
            'Customer_Density': 50.0, 'Competition_Level': 5.0,
            'Distance_to_Center': 5.0
        }
        mins = {'Latitude': -10.0, 'Longitude': 100.0,
                'Customer_Density': 0.0, 'Competition_Level': 1.0, 'Distance_to_Center': 0.0}
        maxs = {'Latitude': 10.0, 'Longitude': 115.0,
                'Customer_Density': 100.0, 'Competition_Level': 10.0, 'Distance_to_Center': 20.0}
        for col_ui, feat in zip(cols, selected_features):
            input_values[feat] = col_ui.number_input(feat,
                                                      min_value=float(mins.get(feat, 0)),
                                                      max_value=float(maxs.get(feat, 100)),
                                                      value=float(defaults.get(feat, 0)))
        submitted_c = st.form_submit_button("▶ Prediksi Klaster")

    if submitted_c:
        inp_df     = pd.DataFrame([input_values])
        inp_scaled = sc.transform(inp_df[selected_features])
        pred_k     = kmeans.predict(inp_scaled)[0]
        st.info(f"📍 Lokasi ini masuk ke **Klaster {pred_k}**")
        if quiet_zones and pred_k in quiet_zones:
            st.error("⚠️ **Zona Sepi** — Tidak direkomendasikan untuk membuka gerai baru.")
        elif quiet_zones:
            st.success("✅ **Zona Ramai** — Potensi baik untuk membuka gerai baru.")
        else:
            st.info("Data tidak cukup untuk menentukan status zona (butuh kolom Customer_Density & Competition_Level).")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-top:40px; padding:20px 0; border-top:1px solid #21262d;
            text-align:center;'>
    <span style='font-family: JetBrains Mono, monospace; font-size:0.7rem;
                 color:#8b949e; letter-spacing:0.08em;'>
        DATA MINING · SIF304 · 2025/2026 · Built with Streamlit
    </span>
</div>
""", unsafe_allow_html=True)