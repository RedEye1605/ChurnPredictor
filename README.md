# Telco Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistem prediksi churn pelanggan telekomunikasi menggunakan machine learning dengan XGBoost, dilengkapi dengan web application berbasis Streamlit untuk analisis interaktif dan rekomendasi retensi.

## 📋 Daftar Isi

- [Overview](#overview)
- [Fitur Utama](#fitur-utama)
- [Struktur Project](#struktur-project)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Web Application](#web-application)
- [Results](#results)
- [Author](#author)

## 🎯 Overview

Project ini mengembangkan sistem prediksi customer churn untuk perusahaan telekomunikasi dengan tujuan:
- Mengidentifikasi pelanggan yang berisiko churn dengan akurasi tinggi
- Memberikan rekomendasi retensi yang actionable
- Menyediakan interface interaktif untuk analisis dan prediksi

Dataset mencakup informasi demografis, layanan, kontrak, dan perilaku pembayaran dari ribuan pelanggan.

## ✨ Fitur Utama

### 1. Machine Learning Pipeline
- **Data Preprocessing**: Cleaning, encoding, feature engineering
- **Feature Selection**: RFE (Recursive Feature Elimination) untuk mengurangi dimensi
- **Class Balancing**: SMOTE untuk handling imbalanced data
- **Model Training**: XGBoost dengan hyperparameter tuning (GridSearchCV)
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### 2. Web Application
- **Single Prediction**: Form input lengkap dengan visualisasi gauge
- **Model Insights**: Metrics, comparison baseline vs tuned, feature importance
- **Batch Scoring**: Upload CSV untuk prediksi massal dengan analytics dashboard
- **Sensitivity Analysis**: Interactive chart untuk melihat dampak perubahan fitur
- **Recommendations Engine**: Otomatis generate strategi retensi

### 3. Feature Engineering
- `tenure_years`: Konversi tenure ke tahun
- `tenure_bucket`: Kategorisasi tenure (0-1yr, 1-2yr, 2-4yr, 4+yr)
- `long_term_contract`: Binary flag untuk kontrak jangka panjang
- `revenue_per_month`: Total revenue dibagi tenure
- `total_streaming_services`: Count layanan streaming
- `total_addon_services`: Count layanan tambahan

## 📁 Struktur Project

```
ChurnPredictor/
├── data/
│   ├── Customer_Info.csv           # Data demografis pelanggan
│   ├── Location_Data.csv           # Data geografis
│   ├── Online_Services.csv         # Data layanan online
│   ├── Payment_Info.csv            # Data pembayaran
│   ├── Service_Options.csv         # Data opsi layanan
│   └── Status_Analysis.csv         # Status churn
├── artifacts/
│   ├── final_churn_model.joblib    # Model XGBoost terlatih
│   └── model_metadata.json         # Metadata (features, metrics, importance)
├── customer_churn_full_project.ipynb  # Notebook lengkap (training)
├── streamlit_app.py                # Web application
├── requirements.txt                # Dependencies
├── TELCO ER Diagram.png            # Entity relationship diagram
└── README.md                       # Dokumentasi

```

## 🛠️ Tech Stack

### Core Libraries
- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Preprocessing, feature selection, evaluation
- **XGBoost**: Gradient boosting model
- **Imbalanced-learn**: SMOTE untuk class balancing

### Visualization & Web
- **Streamlit**: Web application framework
- **Altair**: Interactive visualizations
- **Matplotlib/Seaborn**: Static plots (notebook)

### Others
- **Joblib**: Model serialization
- **JSON**: Metadata storage

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/RedEye1605/ChurnPredictor.git
cd ChurnPredictor
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import streamlit, xgboost, sklearn; print('All packages installed!')"
```

## 💻 Usage

### Training Model (Notebook)

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook customer_churn_full_project.ipynb
   ```

2. **Run All Cells**: Execute semua cell secara berurutan
   - Data Loading & Cleaning
   - EDA & Visualization
   - Feature Engineering
   - Model Training & Tuning
   - Evaluation
   - Save Artifacts

3. **Artifacts Generated**:
   - `artifacts/final_churn_model.joblib` (model)
   - `artifacts/model_metadata.json` (metadata)

### Running Web Application

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### Using the Web App

#### Tab 1: Prediksi
1. Isi form dengan data pelanggan (atau aktifkan "Mode Demo")
2. Klik "Prediksi Risiko Churn"
3. Lihat hasil: probabilitas, risk level, gauge chart
4. Baca rekomendasi retensi
5. Eksplorasi sensitivity analysis untuk fitur tertentu

#### Tab 2: Insight Model
1. Lihat metrics model (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Bandingkan performa baseline vs tuned
3. Analisis feature importance (adjustable top N)
4. Lihat cumulative importance contribution

#### Tab 3: Batch Scoring
1. Upload CSV file melalui sidebar
2. Lihat summary statistics (kategori risiko)
3. Visualisasi distribusi probabilitas & pie chart
4. Identifikasi top 20 pelanggan berisiko tinggi
5. Download hasil (semua atau high-risk only)

## 🤖 Model Details

### Architecture

```
Pipeline:
├── ColumnTransformer
│   ├── Numeric Features → StandardScaler
│   └── Categorical Features → OneHotEncoder
├── FixedFeatureSelector (26 features selected via RFE)
├── SMOTE (sampling_strategy=0.8)
└── XGBClassifier (tuned hyperparameters)
```

### Hyperparameters (Tuned)

```python
{
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 3,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'scale_pos_weight': 2
}
```

### Features Used (26 selected)

**Numeric**: `tenure`, `monthly_charges`, `total_charges`, `total_revenue`, `avg_monthly_gb_download`, dll

**Categorical**: `contract`, `payment_method`, `internet_service`, `online_security`, `device_protection`, dll

**Engineered**: `tenure_years`, `long_term_contract`, `revenue_per_month`, `total_streaming_services`, `total_addon_services`

### Performance Metrics

| Metric | Baseline | Tuned | Improvement |
|--------|----------|-------|-------------|
| Accuracy | 0.812 | 0.829 | +2.1% |
| Precision | 0.683 | 0.720 | +5.4% |
| Recall | 0.593 | 0.749 | +26.3% |
| F1-Score | 0.634 | 0.702 | +10.7% |
| ROC AUC | 0.862 | 0.898 | +4.2% |

**Key Achievement**: Recall improved significantly (59.3% → 74.9%), meaning model catches 75% of actual churners.

## 🌐 Web Application

### Design Principles
- **Clean & Minimal**: No custom CSS, native Streamlit components only
- **Responsive Layout**: Wide mode dengan columns & spacing konsisten
- **Interactive Charts**: Altair untuk visualisasi dinamis
- **User-Friendly**: Clear labels, tooltips, help text

### Key Components

#### 1. Sidebar
- Threshold slider (0.1 - 0.9)
- Quick demo toggle
- CSV uploader
- Model info

#### 2. Prediction Tab
- 2-column form (left: demografis, right: financial)
- Real-time prediction
- Gauge visualization
- Risk categorization (Rendah, Sedang, Tinggi, Sangat Tinggi)
- Actionable recommendations
- Sensitivity analysis with interactive chart

#### 3. Insights Tab
- 5-column metrics display
- Comparison table & bar chart
- Feature importance with adjustable slider
- Cumulative importance metrics
- Model metadata summary

#### 4. Batch Tab
- CSV upload & processing
- 5-metric summary (risk categories)
- Distribution histogram
- Pie chart visualization
- Top 20 high-risk customers table
- Dual download (all / high-risk only)

### Recommendations Engine

System generates personalized retention strategies based on:
- Contract type (Month-to-Month → offer long-term contract)
- Payment method (Electronic check → migrate to auto-pay)
- Add-on services (Low count → cross-sell opportunities)
- Monthly charges (High → review & optimize plan)
- Tenure (New customer → proactive engagement)

## 📊 Results

### Model Performance
- **Best Model**: XGBoost dengan SMOTE + Feature Selection (RFE)
- **F1-Score**: 0.702 (balanced precision & recall)
- **Recall**: 0.749 (detects 75% churners)
- **ROC-AUC**: 0.898 (excellent discrimination)

### Business Impact
- Identifikasi early warning untuk pelanggan berisiko tinggi
- Prioritas tindakan retensi berdasarkan probability score
- Reduce churn rate dengan strategi targeted
- ROI optimization melalui predictive intervention

### Key Insights
1. **Top Predictors**: Contract type, tenure, monthly charges, payment method
2. **High Risk Profile**: Month-to-month contracts, short tenure (<1 year), electronic check payment
3. **Retention Strategy**: Incentivize contract upgrade, optimize pricing, improve service quality

## 👤 Author

**Rhendy Saragih**
- GitHub: [@RedEye1605](https://github.com/RedEye1605)
- Project: [ChurnPredictor](https://github.com/RedEye1605/ChurnPredictor)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Telco Customer Churn (multi-table relational data)
- Framework: Streamlit for rapid web app development
- ML Library: XGBoost for high-performance gradient boosting
- Community: Stack Overflow, Kaggle, GitHub for resources & inspiration

---

**Built with** ❤️ **using Streamlit + XGBoost**

For questions or feedback, please open an issue on GitHub.
