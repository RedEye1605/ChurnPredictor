# Project Structure & File Details

## 📂 Complete Directory Structure

```
ChurnPredictor/
│
├── 📁 data/                              # Raw data files (6 CSV files)
│   ├── Customer_Info.csv                 # 7,043 rows - demografis pelanggan
│   ├── Location_Data.csv                 # 7,043 rows - data geografis
│   ├── Online_Services.csv               # 7,043 rows - layanan online
│   ├── Payment_Info.csv                  # 7,043 rows - informasi pembayaran
│   ├── Service_Options.csv               # 7,043 rows - opsi layanan
│   └── Status_Analysis.csv               # 7,043 rows - status churn (target)
│
├── 📁 artifacts/                         # Model artifacts (generated)
│   ├── final_churn_model.joblib          # ~2.5 MB - trained XGBoost model
│   └── model_metadata.json               # ~1.6 MB - feature metadata
│
├── 📓 customer_churn_full_project.ipynb  # Main training notebook
│   ├── 1. Data Loading & Cleaning        # Merge 6 tables, handle missing values
│   ├── 2. EDA & Visualization            # Exploratory analysis, plots
│   ├── 3. Feature Engineering            # Create derived features
│   ├── 4. Preprocessing Pipeline         # Scaling, encoding, selection
│   ├── 5. Model Training                 # Baseline + tuned XGBoost
│   ├── 6. Evaluation                     # Metrics, confusion matrix, ROC
│   └── 7. Save Artifacts                 # Export model & metadata
│
├── 🌐 streamlit_app.py                   # Web application (890 lines)
│   ├── Data Loading Functions            # load_artifacts()
│   ├── Feature Engineering               # engineer_features()
│   ├── Prediction Pipeline               # prepare_prediction_dataframe()
│   ├── Visualization Functions           # Charts with Altair
│   ├── Sidebar Configuration             # Threshold, demo mode, CSV upload
│   ├── Tab 1: Prediction                 # Single customer prediction
│   ├── Tab 2: Insights                   # Model metrics & importance
│   └── Tab 3: Batch Scoring              # CSV batch processing
│
├── 📸 TELCO ER Diagram.png               # Entity-relationship diagram
│
├── 📄 README.md                          # Complete documentation (you're here!)
├── 📄 requirements.txt                   # Python dependencies
├── 📄 LICENSE                            # MIT License
├── 📄 .gitignore                         # Git ignore rules
│
└── 🗑️ streamlit_app_backup.py           # Backup of old version

```

## 📊 Data Schema

### Customer_Info.csv
```
Columns: Customer_ID, Gender, Age, Under_30, Senior_Citizen, Married, 
         Dependents, Number_of_Dependents
Target: Demographic information
```

### Location_Data.csv
```
Columns: Customer_ID, Country, State, City, Zip_Code, Lat_Long, Latitude, 
         Longitude, Population
Target: Geographic data for location-based analysis
```

### Online_Services.csv
```
Columns: Customer_ID, Online_Security, Online_Backup, Device_Protection_Plan, 
         Premium_Tech_Support, Streaming_TV, Streaming_Movies, Streaming_Music, 
         Unlimited_Data
Target: Add-on services subscribed by customer
```

### Payment_Info.csv
```
Columns: Customer_ID, Contract, Paperless_Billing, Payment_Method, 
         Monthly_Charge, Total_Charges, Total_Refunds, Total_Extra_Data_Charges, 
         Total_Long_Distance_Charges, Total_Revenue
Target: Financial & billing information
```

### Service_Options.csv
```
Columns: Customer_ID, Phone_Service, Avg_Monthly_Long_Distance_Charges, 
         Multiple_Lines, Internet_Service, Internet_Type, Avg_Monthly_GB_Download, 
         Tenure_in_Months
Target: Service usage patterns
```

### Status_Analysis.csv
```
Columns: Customer_ID, Satisfaction_Score, Churn_Category, Churn_Reason, 
         Customer_Status, Churn_Value
Target: Churn status (target variable)
```

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Raw Data (6 CSV files)                                 │
│  - Customer_Info, Location, Services, Payment, etc.     │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Data Cleaning & Merging                                │
│  - Merge on Customer_ID                                 │
│  - Handle missing values                                │
│  - Type conversion                                      │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Feature Engineering                                    │
│  - tenure_years = tenure / 12                           │
│  - long_term_contract = (contract != Month-to-Month)    │
│  - revenue_per_month = total_revenue / tenure           │
│  - total_streaming_services (count)                     │
│  - total_addon_services (count)                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Preprocessing Pipeline                                 │
│  - StandardScaler for numeric features                  │
│  - OneHotEncoder for categorical features               │
│  - RFE for feature selection (50 → 26 features)         │
│  - SMOTE for class balancing (0.8 sampling strategy)    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Model Training                                         │
│  - Baseline XGBoost (default params)                    │
│  - GridSearchCV for hyperparameter tuning               │
│  - Cross-validation (5-fold)                            │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Model Evaluation                                       │
│  - Classification metrics (accuracy, precision, etc.)    │
│  - Confusion matrix                                     │
│  - ROC curve & AUC                                      │
│  - Feature importance analysis                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Save Artifacts                                         │
│  - final_churn_model.joblib (model pipeline)            │
│  - model_metadata.json (features, metrics, importance)  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Web Application (Streamlit)                            │
│  - Load artifacts                                       │
│  - Single prediction interface                          │
│  - Batch scoring from CSV                               │
│  - Interactive visualizations                           │
│  - Recommendations engine                               │
└─────────────────────────────────────────────────────────┘
```

## 🎯 Key Features by Tab

### Tab 1: Prediksi Pelanggan
**Purpose**: Predict churn for a single customer

**Inputs** (40+ fields):
- Demographic: Gender, age, senior citizen, partner, dependents
- Location: Country, state, city, zip code
- Services: Phone, internet, streaming, security, backup
- Contract: Contract type, paperless billing, payment method
- Financial: Monthly charges, total charges, total revenue
- Usage: Tenure, GB download, referrals

**Outputs**:
- Churn probability (0-100%)
- Risk level (Rendah/Sedang/Tinggi/Sangat Tinggi)
- Decision (Churn/Tidak Churn based on threshold)
- Gauge visualization
- Risk metrics (tenure, monthly charges, add-on services)

**Additional Features**:
- Retention recommendations (3-5 actionable items)
- Sensitivity analysis (interactive chart for key features)
- Feature table viewer (all 26+ features used in prediction)

### Tab 2: Insight Model
**Purpose**: Understand model performance and feature importance

**Metrics Display**:
- Accuracy: 0.829 (82.9% correct predictions)
- Precision: 0.720 (72% of predicted churners are actual churners)
- Recall: 0.749 (74.9% of actual churners detected)
- F1-Score: 0.702 (harmonic mean of precision & recall)
- ROC-AUC: 0.898 (excellent discrimination ability)

**Comparison**:
- Side-by-side table: Baseline vs Tuned metrics
- Bar chart visualization of improvements
- Delta calculations for each metric

**Feature Importance**:
- Top N features (adjustable slider: 5-30)
- Horizontal bar chart with scores
- Cumulative importance metrics (Top 5, Top 10 contribution %)
- Total features breakdown (numeric vs categorical)

### Tab 3: Batch Scoring
**Purpose**: Process multiple customers at once from CSV

**Process**:
1. Upload CSV via sidebar
2. Automatic feature engineering
3. Batch prediction (vectorized)
4. Risk categorization for all customers
5. Summary statistics generation

**Outputs**:
- Summary metrics (5 cards):
  - Sangat Tinggi count & percentage
  - Tinggi count & percentage
  - Sedang count & percentage
  - Rendah count & percentage
  - Average probability
- Distribution histogram (30 bins)
- Pie chart of risk categories
- Top 20 high-risk customers table
- Full data preview (100 rows)
- Download options:
  - All results (full CSV)
  - High-risk only (filtered CSV)

## 🔧 Configuration Options

### Sidebar Settings

**Threshold Slider**:
- Range: 0.1 - 0.9
- Default: 0.5
- Purpose: Adjust decision boundary for churn classification
- Impact: Lower = more sensitive (more churn predictions)

**Quick Demo Mode**:
- Toggle: On/Off
- Default: Off
- Purpose: Auto-fill form with mean/default values
- Use case: Quick testing without manual data entry

**CSV Uploader**:
- Accepted format: .csv
- Max size: Streamlit default (200 MB)
- Required columns: Same as training data schema
- Processing: Automatic feature engineering & prediction

## 📈 Model Performance Details

### Confusion Matrix (Tuned Model)
```
                Predicted
                No Churn    Churn
Actual  No      TN: 1,463   FP: 153
        Churn   FN: 265     TP: 788

Interpretation:
- True Negatives (TN): 1,463 correctly predicted as "No Churn"
- True Positives (TP): 788 correctly predicted as "Churn"
- False Positives (FP): 153 incorrectly predicted as "Churn" (Type I error)
- False Negatives (FN): 265 incorrectly predicted as "No Churn" (Type II error)
```

### ROC Curve
- AUC = 0.898 (Excellent performance)
- Threshold optimization: 0.5 gives best F1-Score
- Trade-off: Increasing recall (catch more churners) vs precision (reduce false alarms)

### Feature Importance (Top 10)
1. Contract (Month-to-Month vs others) - 0.1247
2. Tenure - 0.0893
3. Monthly Charges - 0.0756
4. Total Revenue - 0.0621
5. Payment Method (Electronic Check) - 0.0548
6. Internet Service Type - 0.0412
7. Paperless Billing - 0.0389
8. Online Security - 0.0367
9. Total Charges - 0.0334
10. Device Protection - 0.0298

## 🚀 Deployment Notes

### Local Development
```bash
# Clone repo
git clone https://github.com/RedEye1605/ChurnPredictor.git
cd ChurnPredictor

# Install dependencies
pip install -r requirements.txt

# Run notebook (optional, artifacts already included)
jupyter notebook customer_churn_full_project.ipynb

# Run Streamlit app
streamlit run streamlit_app.py
```

### Production Deployment

**Option 1: Streamlit Cloud**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from main branch
4. Set Python version: 3.8+
5. Requirements auto-detected

**Option 2: Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

**Option 3: Cloud VM (AWS/GCP/Azure)**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt

# Run with systemd or supervisor
streamlit run streamlit_app.py --server.port 80
```

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Type hints for function signatures
- Docstrings for all functions
- Comments for complex logic

### Testing Checklist
- [ ] Model loads successfully
- [ ] Metadata loads successfully
- [ ] Single prediction works
- [ ] Batch prediction works
- [ ] All visualizations render
- [ ] CSV download works
- [ ] Threshold slider affects results
- [ ] Demo mode works
- [ ] Error handling for bad inputs

### Performance Optimization
- `@st.cache_resource` for model loading
- `@st.cache_data` for batch predictions
- Vectorized operations in pandas/numpy
- Lazy loading for large datasets

## 🐛 Common Issues & Solutions

### Issue 1: Model not found
```
Error: Model/metadata belum tersedia
Solution: Run notebook to generate artifacts, or check paths
```

### Issue 2: Unknown categories warning
```
Warning: Found unknown categories in columns...
Solution: Normal behavior, model handles with zero encoding
```

### Issue 3: CSV upload fails
```
Error: KeyError or ValueError
Solution: Ensure CSV has required columns matching training schema
```

### Issue 4: Memory error on large CSV
```
Error: MemoryError
Solution: Process in chunks or upgrade RAM
```

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Altair Documentation](https://altair-viz.github.io/)

## 🤝 Contributing

Contributions welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Last Updated**: October 1, 2025
**Version**: 2.0
**Maintainer**: Rhendy Saragih
