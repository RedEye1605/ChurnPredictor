"""Streamlit app for Telco customer churn prediction."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.base import BaseEstimator, TransformerMixin


# Custom transformer registry
class FixedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, support_mask):
        self.support_mask = np.asarray(support_mask, dtype=bool)

    def fit(self, X, y=None):
        self.n_features_in_ = getattr(X, "shape", (None, None))[1]
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X[:, self.support_mask]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_mask)[0]
        return self.support_mask


# Register class in multiple modules to ensure unpickling works
sys.modules.setdefault("__main__", sys.modules[__name__])
if "__main__" in sys.modules:
    setattr(sys.modules["__main__"], "FixedFeatureSelector", FixedFeatureSelector)

# Also register in current module
globals()["FixedFeatureSelector"] = FixedFeatureSelector


# Paths & constants
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "final_churn_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

DERIVED_COLUMNS = {
    "tenure_years",
    "tenure_bucket",
    "long_term_contract",
    "revenue_per_month",
    "total_streaming_services",
    "total_addon_services",
}

# Page config
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# Utilities
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained model and metadata"""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    
    # Ensure FixedFeatureSelector is available before unpickling
    import __main__
    if not hasattr(__main__, "FixedFeatureSelector"):
        __main__.FixedFeatureSelector = FixedFeatureSelector
    
    model = load(MODEL_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to match training"""
    df_fe = df.copy()
    streaming_cols = [c for c in ["streaming_tv", "streaming_movies", "streaming_music"] if c in df_fe.columns]
    service_cols = [
        c for c in [
            "phone_service", "online_security", "online_backup", "device_protection",
            "premium_tech_support", "streaming_tv", "streaming_movies", "streaming_music", "unlimited_data"
        ] if c in df_fe.columns
    ]

    if "tenure" in df_fe.columns:
        df_fe["tenure_years"] = df_fe["tenure"] / 12.0
        df_fe["tenure_bucket"] = pd.cut(
            df_fe["tenure"], bins=[-1, 12, 24, 48, 999], labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
        ).astype(str)

    if {"contract", "total_revenue", "tenure"}.issubset(df_fe.columns):
        df_fe["long_term_contract"] = (df_fe["contract"] != "Month-to-Month").astype(int)
        mask = df_fe["tenure"] > 0
        df_fe["revenue_per_month"] = np.where(mask, df_fe["total_revenue"] / df_fe["tenure"], 0)

    if streaming_cols:
        df_fe["total_streaming_services"] = df_fe[streaming_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
    if service_cols:
        df_fe["total_addon_services"] = df_fe[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)
    
    return df_fe


def prepare_prediction_dataframe(raw_inputs: Dict[str, object], metadata: Dict[str, object]) -> pd.DataFrame:
    """Prepare input data for model prediction"""
    feature_columns: List[str] = metadata["feature_columns"]
    numeric_features: List[str] = metadata["numeric_features"]
    categorical_features: List[str] = metadata["categorical_features"]

    df = pd.DataFrame([raw_inputs])
    df_engineered = engineer_features(df)

    # Add missing features
    for col in feature_columns:
        if col not in df_engineered.columns:
            if col in numeric_features or col in DERIVED_COLUMNS:
                df_engineered[col] = 0
            else:
                df_engineered[col] = "Unknown"

    df_engineered = df_engineered[feature_columns]

    # Type conversion
    for col in numeric_features:
        if col in df_engineered.columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors="coerce").fillna(0)
    for col in categorical_features:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].astype(str)

    return df_engineered.fillna(0)


def format_risk_level(prob: float) -> str:
    """Format risk level based on probability"""
    if prob >= 0.7:
        return "Sangat Tinggi"
    elif prob >= 0.5:
        return "Tinggi"
    elif prob >= 0.3:
        return "Sedang"
    else:
        return "Rendah"


def build_recommendations(input_df: pd.DataFrame, raw_inputs: Dict[str, object]) -> List[str]:
    """Generate retention recommendations"""
    recs: List[str] = []
    row = input_df.iloc[0]

    if raw_inputs.get("contract") == "Month-to-Month":
        recs.append("Tawarkan kontrak jangka panjang dengan diskon loyalitas")
    if raw_inputs.get("payment_method") == "Electronic check":
        recs.append("Migrasi ke metode pembayaran otomatis (kartu kredit/bank transfer)")
    if row.get("total_addon_services", 0) <= 2:
        recs.append("Cross-sell layanan tambahan (streaming, security, backup)")
    if float(raw_inputs.get("monthly_charges", 0)) > 90:
        recs.append("Review paket layanan untuk optimasi biaya")
    if row.get("tenure_years", 0) < 1:
        recs.append("Proactive engagement: survey kepuasan pada bulan ke-3 dan ke-6")

    if not recs:
        recs.append("Pertahankan kualitas layanan dan monitor kepuasan pelanggan")
    
    return recs


def create_gauge_chart(prob: float) -> alt.Chart:
    """Create donut gauge for probability"""
    pct = float(np.clip(prob, 0, 1))
    data = pd.DataFrame({
        "category": ["Risk", "Safe"],
        "value": [pct, 1 - pct],
        "color": ["#ef4444", "#e5e7eb"]
    })
    
    base = alt.Chart(data).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("color:N", scale=None, legend=None)
    )
    
    arc = base.mark_arc(innerRadius=70, outerRadius=110)
    
    text = alt.Chart(pd.DataFrame({"prob": [pct]})).mark_text(
        align="center",
        baseline="middle",
        fontSize=32,
        fontWeight="bold",
        color="#1f2937"
    ).encode(
        text=alt.Text("prob:Q", format=".1%")
    )
    
    return (arc + text).properties(width=250, height=250)


def create_sensitivity_chart(model, metadata, raw_inputs: Dict[str, object], feature: str) -> alt.Chart:
    """Create sensitivity analysis chart"""
    numeric_summary = metadata.get("numeric_summary", {})
    summary = numeric_summary.get(feature, {"min": 0.0, "max": 100.0})
    lo, hi = float(summary.get("min", 0.0)), float(summary.get("max", 100.0))
    
    if lo == hi:
        hi = lo + 1.0

    xs = np.linspace(lo, hi, 25)
    results = []
    
    for x in xs:
        temp_inputs = {**raw_inputs, feature: float(x)}
        df_temp = prepare_prediction_dataframe(temp_inputs, metadata)
        prob = float(model.predict_proba(df_temp)[0, 1])
        results.append({"value": float(x), "probability": prob})

    df_chart = pd.DataFrame(results)
    
    chart = alt.Chart(df_chart).mark_line(
        point=alt.OverlayMarkDef(filled=True, size=60),
        color="#3b82f6",
        strokeWidth=3
    ).encode(
        x=alt.X("value:Q", title=feature.replace("_", " ").title()),
        y=alt.Y("probability:Q", title="Churn Probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("value:Q", format=".2f", title=feature.replace("_", " ").title()),
            alt.Tooltip("probability:Q", format=".1%", title="Probability")
        ]
    ).properties(height=300)
    
    return chart


# Sidebar
with st.sidebar:
    st.title("Telco Churn Prediction")
    st.caption("Prediksi risiko churn pelanggan")
    
    st.write("")
    st.write("")
    
    st.subheader("Pengaturan")
    threshold = st.slider(
        "Threshold Keputusan",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probabilitas di atas nilai ini diklasifikasikan sebagai Churn"
    )
    
    st.write("")
    
    quick_demo = st.toggle("Mode Demo", value=False, help="Isi form dengan nilai default")
    
    st.divider()
    
    st.subheader("Batch Upload")
    uploaded_csv = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload file CSV untuk prediksi batch"
    )
    
    if uploaded_csv:
        st.success(f"File: {uploaded_csv.name}")
        st.caption(f"Size: {uploaded_csv.size/1024:.1f} KB")
    
    st.divider()
    
    st.caption("Model: XGBoost Pipeline")
    st.caption("Author: Rhendy Saragih")


# Main app
def render_prediction_tab(model, metadata):
    """Single customer prediction"""
    st.header("Prediksi Pelanggan")
    st.write("Masukkan data pelanggan untuk memprediksi risiko churn")
    
    st.write("")
    
    feature_columns = metadata["feature_columns"]
    numeric_summary = metadata.get("numeric_summary", {})
    category_options = metadata.get("category_options", {})
    
    # Separate columns
    left_cols = [
        "gender", "senior_citizen", "partner", "dependents", "number_of_dependents",
        "phone_service", "multiple_lines", "internet_service", "online_security",
        "online_backup", "device_protection", "premium_tech_support", "streaming_tv",
        "streaming_movies", "streaming_music", "unlimited_data", "referred_a_friend"
    ]
    right_cols = [
        "country", "state", "city", "zip_code", "total_population", "latitude", "longitude",
        "contract", "paperless_billing", "payment_method", "monthly_charges",
        "avg_monthly_long_distance_charges", "total_charges", "total_refunds",
        "total_extra_data_charges", "total_long_distance_charges", "total_revenue",
        "tenure", "avg_monthly_gb_download", "number_of_referrals"
    ]
    
    base_columns = [c for c in feature_columns if c not in DERIVED_COLUMNS and c in left_cols + right_cols]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        raw_inputs: Dict[str, object] = {}

        def num_input(colname: str, parent):
            summary = numeric_summary.get(colname, {})
            min_val = float(summary.get("min", 0.0))
            max_val = float(summary.get("max", min_val + 100))
            mean_val = float(summary.get("mean", min_val))
            
            if colname in {"number_of_dependents", "zip_code", "total_population", "tenure",
                           "avg_monthly_gb_download", "number_of_referrals"}:
                min_int = int(np.floor(min_val))
                max_int = int(np.ceil(max_val) if max_val >= min_val else min_val + 100)
                default_int = int(np.clip(round(mean_val), min_int, max_int))
                return parent.number_input(
                    colname.replace("_", " ").title(),
                    min_value=min_int,
                    max_value=max_int,
                    value=default_int,
                    step=1
                )
            else:
                min_float = float(min_val)
                max_float = float(max_val if max_val > min_val else min_val + 100)
                default_float = float(np.clip(mean_val, min_float, max_float))
                step = 0.1 if max_float - min_float > 1 else 0.01
                return parent.number_input(
                    colname.replace("_", " ").title(),
                    min_value=min_float,
                    max_value=max_float,
                    value=default_float,
                    step=step
                )

        # Left column
        for col in left_cols:
            if col not in base_columns:
                continue
            with col1:
                if col in category_options:
                    options = category_options.get(col, ["No", "Yes"])
                    raw_inputs[col] = options[0] if quick_demo else st.selectbox(
                        col.replace("_", " ").title(), options
                    )
                else:
                    raw_inputs[col] = num_input(col, col1) if not quick_demo else numeric_summary.get(col, {}).get("mean", 0.0)

        # Right column
        for col in right_cols:
            if col not in base_columns:
                continue
            with col2:
                if col in category_options:
                    options = category_options.get(col, ["No", "Yes"])
                    raw_inputs[col] = options[0] if quick_demo else st.selectbox(
                        col.replace("_", " ").title(), options
                    )
                else:
                    raw_inputs[col] = num_input(col, col2) if not quick_demo else numeric_summary.get(col, {}).get("mean", 0.0)

        st.write("")
        submitted = st.form_submit_button("Prediksi Risiko Churn", use_container_width=True, type="primary")

    if not submitted:
        st.info("Lengkapi formulir dan klik tombol Prediksi")
        return

    # Make prediction
    input_prepared = prepare_prediction_dataframe(raw_inputs, metadata)
    prob = float(model.predict_proba(input_prepared)[0, 1])
    risk_level = format_risk_level(prob)
    decision = "Churn" if prob >= threshold else "Tidak Churn"

    st.write("")
    st.write("")
    
    # Results
    st.subheader("Hasil Prediksi")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="large")
    
    with metric_col1:
        st.metric("Probabilitas Churn", f"{prob:.1%}")
    with metric_col2:
        st.metric("Tingkat Risiko", risk_level)
    with metric_col3:
        st.metric("Keputusan", decision)
    with metric_col4:
        addon = int(input_prepared.get("total_addon_services", pd.Series([0])).iloc[0])
        st.metric("Layanan Tambahan", f"{addon}")

    st.write("")
    st.write("")
    
    # Gauge chart
    col_gauge, col_info = st.columns([1, 2], gap="large")
    
    with col_gauge:
        st.altair_chart(create_gauge_chart(prob), use_container_width=True)
    
    with col_info:
        st.write("**Detail Risiko**")
        st.progress(prob)
        st.caption(f"Threshold: {threshold:.0%}")
        
        st.write("")
        
        tenure_val = float(raw_inputs.get("tenure", 0))
        monthly_val = float(raw_inputs.get("monthly_charges", 0))
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Tenure", f"{tenure_val:.0f} bulan")
        with info_col2:
            st.metric("Biaya Bulanan", f"${monthly_val:.2f}")

    st.write("")
    st.divider()
    
    # Recommendations
    st.subheader("Rekomendasi Retensi")
    recommendations = build_recommendations(input_prepared, raw_inputs)
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    st.write("")
    st.divider()
    
    # Sensitivity analysis
    st.subheader("Analisis Sensitivitas")
    st.write("Lihat bagaimana perubahan fitur mempengaruhi probabilitas churn")
    
    st.write("")
    
    key_features = ["monthly_charges", "tenure", "total_revenue", "avg_monthly_gb_download"]
    available_features = [f for f in key_features if f in raw_inputs or f in input_prepared.columns]
    
    if available_features:
        selected_feature = st.selectbox("Pilih Fitur", available_features)
        st.write("")
        st.altair_chart(
            create_sensitivity_chart(model, metadata, raw_inputs, selected_feature),
            use_container_width=True
        )
    else:
        st.info("Tidak ada fitur yang tersedia untuk analisis")
    
    st.write("")
    
    with st.expander("Lihat Data Fitur"):
        feature_view = input_prepared.T.reset_index()
        feature_view.columns = ["Fitur", "Nilai"]
        st.dataframe(feature_view, use_container_width=True, height=400)


def render_insights_tab(metadata):
    """Model performance insights"""
    st.header("Performa Model")
    st.write("Metrik evaluasi dan feature importance")
    
    st.write("")
    st.write("")
    
    tuned = metadata.get("tuned_metrics", {}) or {}
    comparison = pd.DataFrame(metadata.get("comparison_metrics", []))
    feat_imp = pd.DataFrame(metadata.get("top_feature_importances", []))

    # Metrics
    st.subheader("Metrik Model")
    
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")
    col1.metric("Accuracy", f"{tuned.get('Accuracy', 0):.3f}")
    col2.metric("Precision", f"{tuned.get('Precision', 0):.3f}")
    col3.metric("Recall", f"{tuned.get('Recall', 0):.3f}")
    col4.metric("F1-Score", f"{tuned.get('F1-Score', 0):.3f}")
    col5.metric("ROC AUC", f"{tuned.get('ROC AUC', 0):.3f}")

    st.write("")
    st.divider()

    # Comparison
    st.subheader("Perbandingan Baseline vs Tuned")
    
    if not comparison.empty:
        comparison_display = comparison.copy()
        comparison_display.columns = ["Metrik", "Baseline", "Tuned", "Delta"]
        
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)
        
        st.write("")
        
        # Bar chart comparison
        comp_melted = comparison.melt(
            id_vars=["Metric"],
            value_vars=["Baseline", "Tuned"],
            var_name="Model",
            value_name="Score"
        )
        
        chart = alt.Chart(comp_melted).mark_bar().encode(
            x=alt.X("Metric:N", title=""),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Baseline", "Tuned"], range=["#94a3b8", "#3b82f6"])),
            xOffset="Model:N",
            tooltip=["Metric", "Model", alt.Tooltip("Score", format=".3f")]
        ).properties(height=350)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Data perbandingan tidak tersedia")

    st.write("")
    st.divider()

    # Feature importance
    st.subheader("Feature Importance")
    
    if not feat_imp.empty:
        top_n = st.slider("Jumlah Fitur Teratas", 5, 30, 15, 5)
        
        st.write("")
        
        feat_display = feat_imp.head(top_n)
        
        chart = alt.Chart(feat_display).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance Score"),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["feature", alt.Tooltip("importance", format=".4f")]
        ).properties(height=max(300, top_n * 20))
        
        st.altair_chart(chart, use_container_width=True)
        
        st.write("")
        
        # Cumulative importance
        feat_sorted = feat_imp.sort_values("importance", ascending=False).reset_index(drop=True)
        feat_sorted["cumulative_pct"] = feat_sorted["importance"].cumsum() / feat_sorted["importance"].sum()
        
        top5_pct = feat_sorted.head(5)["importance"].sum() / feat_sorted["importance"].sum()
        top10_pct = feat_sorted.head(10)["importance"].sum() / feat_sorted["importance"].sum()
        
        info_col1, info_col2 = st.columns(2, gap="large")
        with info_col1:
            st.metric("Top 5 Features", f"{top5_pct:.1%}", "dari total importance")
        with info_col2:
            st.metric("Top 10 Features", f"{top10_pct:.1%}", "dari total importance")
    else:
        st.info("Data feature importance tidak tersedia")

    st.write("")
    st.divider()

    # Model info
    st.subheader("Informasi Model")
    
    info_col1, info_col2, info_col3 = st.columns(3, gap="large")
    
    with info_col1:
        n_features = len(metadata.get("feature_columns", []))
        st.metric("Total Fitur", n_features)
    with info_col2:
        n_numeric = len(metadata.get("numeric_features", []))
        st.metric("Fitur Numerik", n_numeric)
    with info_col3:
        n_categorical = len(metadata.get("categorical_features", []))
        st.metric("Fitur Kategorikal", n_categorical)

    training_date = metadata.get("training_date")
    if training_date:
        st.caption(f"Model dilatih pada: {pd.to_datetime(training_date).strftime('%d %B %Y, %H:%M WIB')}")


@st.cache_data(show_spinner=False)
def predict_batch_csv(model, metadata, csv_bytes: bytes) -> pd.DataFrame:
    """Process batch prediction from CSV"""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df_eng = engineer_features(df)
    
    feature_cols = metadata["feature_columns"]
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    # Add missing columns
    for col in feature_cols:
        if col not in df_eng.columns:
            if col in numeric_features or col in DERIVED_COLUMNS:
                df_eng[col] = 0
            else:
                df_eng[col] = "Unknown"

    df_eng = df_eng[feature_cols].copy()
    
    for col in numeric_features:
        if col in df_eng.columns:
            df_eng[col] = pd.to_numeric(df_eng[col], errors="coerce").fillna(0)
    for col in categorical_features:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].astype(str)

    proba = model.predict_proba(df_eng)[:, 1]
    result = df.copy()
    result["churn_probability"] = proba
    result["risk_level"] = result["churn_probability"].apply(format_risk_level)
    
    return result


def render_batch_tab(model, metadata):
    """Batch prediction from CSV"""
    st.header("Batch Scoring")
    st.write("Upload file CSV untuk prediksi massal")
    
    st.write("")

    if uploaded_csv is None:
        st.info("Upload file CSV melalui sidebar")
        
        st.write("")
        
        with st.expander("Format CSV"):
            st.write("File CSV harus memiliki kolom yang sesuai dengan data training:")
            example_cols = metadata.get("feature_columns", [])[:15]
            st.code(", ".join(example_cols) + ", ...")
        return

    with st.status("Memproses CSV...", expanded=False) as status:
        try:
            result_df = predict_batch_csv(model, metadata, uploaded_csv.read())
            status.update(label="Selesai", state="complete")
        except Exception as e:
            status.update(label=f"Error: {e}", state="error")
            st.stop()

    st.success(f"Berhasil memprediksi {len(result_df):,} pelanggan")
    
    st.write("")
    st.write("")
    
    # Summary statistics
    st.subheader("Ringkasan Hasil")
    
    proba_col = "churn_probability"
    n_very_high = (result_df[proba_col] >= 0.7).sum()
    n_high = ((result_df[proba_col] >= 0.5) & (result_df[proba_col] < 0.7)).sum()
    n_medium = ((result_df[proba_col] >= 0.3) & (result_df[proba_col] < 0.5)).sum()
    n_low = (result_df[proba_col] < 0.3).sum()
    avg_prob = result_df[proba_col].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")
    
    col1.metric("Sangat Tinggi", n_very_high, f"{n_very_high/len(result_df)*100:.1f}%")
    col2.metric("Tinggi", n_high, f"{n_high/len(result_df)*100:.1f}%")
    col3.metric("Sedang", n_medium, f"{n_medium/len(result_df)*100:.1f}%")
    col4.metric("Rendah", n_low, f"{n_low/len(result_df)*100:.1f}%")
    col5.metric("Rata-rata Prob", f"{avg_prob:.1%}")

    st.write("")
    st.divider()
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2, gap="large")
    
    with viz_col1:
        st.write("**Distribusi Probabilitas**")
        
        hist_chart = alt.Chart(result_df).mark_bar().encode(
            x=alt.X(f"{proba_col}:Q", bin=alt.Bin(maxbins=30), title="Churn Probability"),
            y=alt.Y("count()", title="Jumlah Pelanggan"),
            tooltip=[alt.Tooltip(f"{proba_col}:Q", bin=True, format=".2f"), "count()"]
        ).properties(height=300)
        
        st.altair_chart(hist_chart, use_container_width=True)
    
    with viz_col2:
        st.write("**Kategori Risiko**")
        
        risk_counts = result_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Kategori", "Jumlah"]
        
        pie_chart = alt.Chart(risk_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta("Jumlah:Q"),
            color=alt.Color("Kategori:N"),
            tooltip=["Kategori", "Jumlah"]
        ).properties(height=300)
        
        st.altair_chart(pie_chart, use_container_width=True)

    st.write("")
    st.divider()
    
    # High-risk customers
    st.subheader("Pelanggan Berisiko Tinggi (Top 20)")
    
    high_risk_df = result_df.nlargest(20, proba_col)
    display_df = high_risk_df[[proba_col, "risk_level"]].copy()
    display_df[proba_col] = display_df[proba_col].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(display_df, use_container_width=True, height=400)

    st.write("")
    st.divider()
    
    # Preview
    st.subheader("Preview Data (100 baris pertama)")
    
    preview_df = result_df.head(100).copy()
    preview_df[proba_col] = preview_df[proba_col].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(preview_df, use_container_width=True, height=400)

    st.write("")
    st.divider()
    
    # Download
    st.subheader("Download Hasil")
    
    col_dl1, col_dl2 = st.columns(2, gap="large")
    
    with col_dl1:
        buf_full = io.BytesIO()
        result_df.to_csv(buf_full, index=False)
        st.download_button(
            "Download Semua Hasil (CSV)",
            data=buf_full.getvalue(),
            file_name="churn_predictions_all.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        high_risk_all = result_df[result_df[proba_col] >= 0.5]
        buf_high = io.BytesIO()
        high_risk_all.to_csv(buf_high, index=False)
        st.download_button(
            "Download Risiko Tinggi (CSV)",
            data=buf_high.getvalue(),
            file_name="churn_predictions_high_risk.csv",
            mime="text/csv",
            use_container_width=True
        )


# Main app
def main():
    model, metadata = load_artifacts()
    
    st.title("Telco Churn Prediction System")
    st.caption("Prediksi risiko churn dan rekomendasi retensi pelanggan")

    if model is None or metadata is None:
        st.error("Model belum tersedia. Jalankan notebook training terlebih dahulu.")
        return

    st.write("")
    st.divider()
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Prediksi", "Insight Model", "Batch Scoring"])
    
    with tab1:
        render_prediction_tab(model, metadata)
    with tab2:
        render_insights_tab(metadata)
    with tab3:
        render_batch_tab(model, metadata)

    st.write("")
    st.write("")
    st.divider()
    
    footer_col1, footer_col2 = st.columns([2, 1])
    with footer_col1:
        st.caption("Built with Streamlit + XGBoost")
    with footer_col2:
        st.caption("Author: Rhendy Saragih")


if __name__ == "__main__":
    main()
