# Quick Start Guide

Get up and running with Telco Churn Prediction System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB RAM minimum
- Modern web browser

## Installation (3 Steps)

### Step 1: Clone Repository

```bash
git clone https://github.com/RedEye1605/ChurnPredictor.git
cd ChurnPredictor
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (web framework)
- pandas, numpy (data processing)
- scikit-learn, xgboost (machine learning)
- altair (visualizations)
- And other required packages

**Note**: Installation may take 2-5 minutes depending on your internet speed.

### Step 3: Run Application

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## First Time Usage

### 1. Explore Single Prediction

**Quick Demo Mode**:
1. Click sidebar checkbox "Mode Demo"
2. Click "Prediksi Risiko Churn" button
3. View results instantly!

**Manual Entry**:
1. Uncheck "Mode Demo"
2. Fill in customer information (split into 2 columns):
   - Left: Demographics & services
   - Right: Location & financial
3. Click "Prediksi Risiko Churn"
4. Analyze results:
   - Probability gauge
   - Risk level
   - Retention recommendations
   - Sensitivity analysis

### 2. Check Model Performance

Click "Insight Model" tab:
- View 5 key metrics
- Compare baseline vs tuned model
- Explore top features (use slider to adjust)
- See cumulative importance

### 3. Try Batch Prediction

**Prepare CSV**:
Your CSV should have columns like:
```
customer_id,gender,age,tenure,monthly_charges,contract,payment_method,...
```

**Upload & Process**:
1. Click sidebar "Upload CSV" button
2. Select your CSV file
3. View summary statistics
4. Analyze distribution charts
5. Download results (all or high-risk only)

## Example Workflow

### Scenario: Identify High-Risk Customers

**Step 1**: Upload customer database CSV
```bash
# Your CSV with all customers
customers.csv (1000+ rows)
```

**Step 2**: View batch results
- 250 Sangat Tinggi (25%)
- 180 Tinggi (18%)
- 320 Sedang (32%)
- 250 Rendah (25%)

**Step 3**: Download high-risk customers
Click "Download Risiko Tinggi (CSV)" → Get 430 customers (Sangat Tinggi + Tinggi)

**Step 4**: Plan retention campaign
- Month-to-Month contracts → Offer annual discount
- Electronic check → Migrate to auto-pay
- Low tenure → Proactive engagement
- High charges → Review plan optimization

### Scenario: Analyze Individual Customer

**Profile**:
- Name: John Doe
- Tenure: 3 months (new customer)
- Contract: Month-to-Month
- Monthly Charges: $95
- Payment: Electronic check
- Services: Phone only (no add-ons)

**Prediction**:
- Probability: 78% (Sangat Tinggi)
- Recommendations:
  1. Offer 1-year contract with 15% discount
  2. Migrate to credit card auto-pay (incentive: $10 credit)
  3. Cross-sell streaming bundle (3 services for $20/month)
  4. Schedule satisfaction call in month 4
  5. Assign dedicated account manager

**Action**: High-priority retention case!

## Troubleshooting

### Issue: "Module not found"
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: "Model/metadata not found"
**Solution**:
Check that `artifacts/` folder exists with:
- `final_churn_model.joblib`
- `model_metadata.json`

If missing, run the notebook first:
```bash
jupyter notebook customer_churn_full_project.ipynb
# Run all cells to generate artifacts
```

### Issue: Port 8501 already in use
**Solution**:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Slow CSV processing
**Solution**:
- Reduce CSV size (process in batches)
- Close other applications
- Check RAM usage

## Tips & Best Practices

### For Single Predictions
✅ Use "Mode Demo" for quick testing
✅ Pay attention to risk level color coding
✅ Read all recommendations carefully
✅ Adjust threshold slider to see impact
✅ Use sensitivity analysis to understand key drivers

### For Batch Processing
✅ Validate CSV format before upload
✅ Process large files in chunks (<10K rows recommended)
✅ Download results immediately after processing
✅ Focus on "Sangat Tinggi" category first
✅ Use filters in Excel/Pandas for further analysis

### For Model Analysis
✅ Compare metrics with your business KPIs
✅ Adjust feature importance slider to see more/less
✅ Understand top 10 features for explainability
✅ Share insights with stakeholders

## Next Steps

### Learning Path
1. ✅ Run the app (you're here!)
2. 📖 Read [README.md](README.md) for overview
3. 📚 Read [DOCUMENTATION.md](DOCUMENTATION.md) for details
4. 🔬 Explore Jupyter notebook for model training
5. 🎨 Customize app for your use case

### Customization Ideas
- Add new features (NPS score, support tickets, etc.)
- Integrate with CRM system (API)
- Schedule automated batch predictions
- Create email alerts for high-risk customers
- Build executive dashboard with KPIs

### Integration Examples

**With CRM (Salesforce/HubSpot)**:
```python
# Pseudo-code
customers = crm.get_all_customers()
predictions = model.predict(customers)
crm.update_field('churn_risk', predictions)
```

**With Marketing Automation**:
```python
# Pseudo-code
high_risk = predictions[predictions.prob >= 0.7]
marketing.create_campaign(
    audience=high_risk,
    message="retention_offer",
    channel="email"
)
```

**With BI Tools (Tableau/Power BI)**:
```python
# Export predictions
predictions.to_csv('churn_predictions.csv')
# Import to Tableau/Power BI for dashboards
```

## Support & Community

**Need Help?**
- 📖 Check [DOCUMENTATION.md](DOCUMENTATION.md)
- 🐛 Report issues on [GitHub](https://github.com/RedEye1605/ChurnPredictor/issues)
- 💬 Join discussions on GitHub Discussions
- 📧 Contact: [Your Email]

**Contributing?**
- 🍴 Fork the repo
- 🌿 Create feature branch
- 📝 Submit pull request
- See [Contributing Guidelines](README.md#contributing)

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Tutorials](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- [Customer Churn Best Practices](https://www.example.com)
- [Retention Strategy Playbook](https://www.example.com)

---

**You're all set!** 🚀

Start predicting churn and saving customers today!

**Pro Tip**: Bookmark `http://localhost:8501` for quick access.
