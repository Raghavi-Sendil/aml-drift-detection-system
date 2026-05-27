# SynthAML Dataset - Quick Start Guide

## Your Dataset Files

You have the SynthAML dataset with **TWO** files:

1. **synthetic_transactions.csv** (~904 MB, 16+ million transactions)
   - Contains: AlertID, Timestamp, Entry (Credit/Debit), Type (Wire/Card/etc), Size
   - This is the detailed transaction data

2. **synthetic_alerts_fixed.csv** (~368 KB, 20,000 alerts)
   - Contains: AlertID, Date, Outcome (0=benign, 1=suspicious)
   - This is the labels file - tells us which alerts are actually suspicious

## ⚠️ Important: You MUST Upload BOTH Files

The system needs:
- **Transactions file** for the detailed transaction behavior
- **Alerts file** for the labels (which alerts are suspicious)

## Quick Start Options

### Option 1: Test Script (Recommended First)

Run the test script to verify everything works:

```bash
cd /Users/sycamore/aml_drift_detection_system
python test_synthaml.py
```

This will:
- Load both files
- Process and join them
- Engineer features
- Train a model
- Evaluate performance
- Classify sample alerts
- Show explanations

**Note**: This may take 5-15 minutes due to the large dataset size.

### Option 2: Web Interface

```bash
streamlit run web_interface.py
```

Then:
1. Go to "Upload Data" page
2. Upload **synthetic_transactions.csv** in the left box
3. Upload **synthetic_alerts_fixed.csv** in the right box
4. Click "Process Data"
5. Follow the wizard through the remaining steps

### Option 3: Command Line

```bash
# Train a model
python cli.py train \
    --transactions /Users/sycamore/Downloads/synthetic_transactions.csv \
    --alerts /Users/sycamore/Downloads/synthetic_alerts_fixed.csv \
    --model-type xgboost \
    --output synthaml_model.pkl

# Classify alerts (after training)
python cli.py classify \
    --model synthaml_model.pkl \
    --alert sample_alert.json \
    --explain

# Detect drift
python cli.py detect-drift \
    --model synthaml_model.pkl \
    --transactions new_transactions.csv \
    --alerts new_alerts.csv \
    --output drift_report.json
```

### Option 4: Python Script

```python
from main import AMLDriftDetectionSystem

# Initialize
system = AMLDriftDetectionSystem()

# Load BOTH files
system.upload_and_process_datasets(
    transactions_path='/Users/sycamore/Downloads/synthetic_transactions.csv',
    alerts_path='/Users/sycamore/Downloads/synthetic_alerts_fixed.csv'
)

# Engineer features
system.engineer_features()

# Split windows
system.split_temporal_windows()

# Train model
system.train_baseline_model(use_xgboost=True)

# Save
system.save_model('my_model.pkl')
```

## Dataset Details

### Transactions File Format

```csv
AlertID,Timestamp,Entry,Type,Size
1,2019-01-01 00:00:41,Credit,Wire,-0.4389405142366621
1,2019-01-01 00:00:41,Debit,Wire,-1.6565622884073743
2,2019-01-01 00:05:12,Credit,Card,-0.7492669313550533
```

**Columns:**
- **AlertID**: Links to alerts file (1 to 20,000)
- **Timestamp**: When transaction occurred
- **Entry**: Credit or Debit
- **Type**: Wire, Card, Cash, International
- **Size**: Transaction amount (appears to be log-scaled or normalized)

### Alerts File Format

```csv
AlertID,Date,Outcome
1,2020-01-01,0
14,2020-01-01,1
25,2020-01-01,1
```

**Columns:**
- **AlertID**: Alert identifier (1 to 20,000)
- **Date**: Date of alert
- **Outcome**: 0 = benign, 1 = suspicious

### Dataset Statistics

- **Total Transactions**: ~16,000,000
- **Total Alerts**: 20,000
- **Suspicious Alerts**: ~10% (typical for AML datasets)
- **Benign Alerts**: ~90%
- **Transactions per Alert**: Varies (1 to 100+)

## How the System Processes Your Data

1. **Loads transactions** - Reads all 16M transactions
2. **Loads alerts** - Reads the 20K alerts with labels
3. **Joins on AlertID** - Links transactions to their alert labels
4. **Aggregates** - Groups transactions by alert
5. **Engineers features** - Creates 18+ behavioral features per alert:
   - Transaction count
   - Mean/max transaction size
   - Wire transfer count
   - Burst score (clustering)
   - Net flow
   - Round amount ratio
   - And more...
6. **Splits temporally** - Divides by time (40% train, 30% val, 30% test)
7. **Trains model** - XGBoost or Logistic Regression
8. **Detects drift** - Monitors distribution changes
9. **Classifies new alerts** - Predicts suspicious vs benign
10. **Explains with SHAP** - Shows why each decision was made

## Performance Expectations

### Dataset Size
- **Transactions file**: ~904 MB (16M rows)
- **Alerts file**: ~368 KB (20K rows)
- **Processing time**: 2-5 minutes
- **Training time**: 5-15 minutes (XGBoost)
- **Memory usage**: 2-4 GB RAM

### Model Performance (Typical)
- **Recall**: 0.50-0.70 (catches 50-70% of suspicious alerts)
- **False Positive Rate**: 0.40-0.55 (40-55% of benign flagged incorrectly)
- **Accuracy**: 0.55-0.65

These are normal for AML detection - the goal is to reduce FPR while maintaining recall.

## Troubleshooting

### "Memory Error"
The dataset is large. Solutions:
1. Close other applications
2. Use a machine with 8+ GB RAM
3. Sample the data (use first N alerts)

### "File Not Found"
Check file paths:
```bash
ls -lh /Users/sycamore/Downloads/synthetic_*.csv
```

### "Slow Performance"
Normal for 16M transactions. Tips:
- Use XGBoost (faster than other algorithms)
- Be patient during initial load
- Consider using a sample for testing

### "No Drift Detected"
Expected if data is from same time period. To test drift:
1. Split data by date
2. Train on early dates
3. Test on late dates

## What's Different About SynthAML?

The SynthAML dataset requires **TWO files** because:

1. **Separation of concerns**: Transactions vs Labels
2. **Realistic structure**: Mimics real banking systems where transaction data and investigation outcomes are stored separately
3. **Size management**: Transactions are huge (904MB), alerts are small (368KB)

Our system now handles this automatically!

## Next Steps After Testing

Once you've verified the system works:

1. **Experiment with the web interface** - Most user-friendly
2. **Train on full dataset** - Use CLI for batch processing
3. **Set up drift monitoring** - Run weekly checks
4. **Deploy to production** - See USER_GUIDE.md
5. **Customize features** - Add domain-specific features
6. **Tune thresholds** - Adjust PSI thresholds for your needs

## Research Context

This is the **SynthAML dataset** from the research paper you provided. It's specifically designed for:
- Benchmarking AML detection algorithms
- Studying concept drift in financial transactions
- Reducing false positive alerts
- Explainable AI in financial crime detection

The system you have implements all the methodologies from that paper!

## Summary

✅ **What you have**: 2 CSV files (transactions + alerts)

✅ **What the system needs**: Both files uploaded together

✅ **How to use**: Web UI, CLI, or Python API

✅ **What it does**: Classifies alerts as suspicious/benign with explanations

✅ **Performance**: Processes 16M transactions, trains in 5-15 min

✅ **Output**: Predictions + probabilities + SHAP explanations

**Ready to start?**

```bash
python test_synthaml.py
```

Good luck! 🚀
