# Getting Started with Your Own Data

This guide will help you use the AML Drift Detection System with your actual transaction data.

## Step 1: Verify Installation

```bash
cd /Users/sycamore/aml_drift_detection_system
python check_installation.py
```

Expected output:
```
✅ All required packages are installed!
```

If you see any ❌, install missing packages:
```bash
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

### Required Format

Create a CSV file with these columns:

| Column | Type | Example | Required |
|--------|------|---------|----------|
| `alert_id` | int/string | 12345 | ✅ Yes |
| `timestamp` | datetime | 2024-01-15 10:30:00 | ✅ Yes |
| `amount` | float | 5000.00 | ✅ Yes |
| `transaction_type` | string | wire | ✅ Yes |
| `is_suspicious` | int | 1 or 0 | ✅ Yes |

### Example: transactions.csv

```csv
alert_id,timestamp,amount,transaction_type,is_suspicious
1001,2024-01-01 09:15:00,8500.00,wire,1
1001,2024-01-01 10:30:00,9000.00,wire,1
1002,2024-01-02 14:20:00,150.50,debit,0
1002,2024-01-02 15:10:00,200.00,credit,0
```

### Data Requirements

- **Minimum**: 1000+ transactions across 100+ alerts
- **Recommended**: 10,000+ transactions for better model performance
- **Labels**: Include both suspicious (1) and benign (0) alerts
- **Time span**: At least 3-6 months of historical data

## Step 3: Choose Your Interface

### Option A: Web Interface (Easiest)

```bash
streamlit run web_interface.py
```

Then:
1. Navigate to http://localhost:8501
2. Click "Upload Data" in sidebar
3. Upload your CSV file
4. Follow the on-screen instructions

### Option B: Python Script (Most Flexible)

Create `my_workflow.py`:

```python
from main import AMLDriftDetectionSystem

# Initialize
system = AMLDriftDetectionSystem()

# 1. Upload your data
print("Step 1: Loading data...")
system.upload_and_process_datasets(
    transactions_path='YOUR_DATA.csv'
)

# 2. Engineer features
print("Step 2: Creating features...")
system.engineer_features()

# 3. Split temporal windows
print("Step 3: Creating time windows...")
window1, window2, window3 = system.split_temporal_windows()

# 4. Train model
print("Step 4: Training model...")
system.train_baseline_model(use_xgboost=True)

# 5. Evaluate
print("Step 5: Evaluating...")
metrics = system.evaluate_model(window2, "Validation")
print(f"Recall: {metrics['recall']:.3f}")
print(f"FPR: {metrics['fpr']:.3f}")

# 6. Save model
print("Step 6: Saving model...")
system.save_model('my_aml_model.pkl')

print("\n✅ Done! Model saved to my_aml_model.pkl")
```

Run it:
```bash
python my_workflow.py
```

### Option C: Command Line (Quickest)

```bash
# Train in one command
python cli.py train \
    --data YOUR_DATA.csv \
    --model-type xgboost \
    --output my_model.pkl
```

## Step 4: Test Your Model

### Classify a Test Alert

Create `test_alert.json`:
```json
{
    "transaction_count": 15,
    "mean_transaction_size": 8500.00,
    "max_transaction_size": 15000.00,
    "min_transaction_size": 500.00,
    "std_transaction_size": 3200.00,
    "net_flow": 127500.00,
    "burst_score": 0.8,
    "avg_time_between_tx": 2.5,
    "credit_count": 5,
    "debit_count": 10,
    "wire_count": 8,
    "credit_debit_ratio": 0.5,
    "wire_ratio": 0.53,
    "transaction_velocity": 6.0,
    "amount_variance": 0.376,
    "weekend_transaction_ratio": 0.4,
    "night_transaction_ratio": 0.3,
    "round_amount_ratio": 0.6,
    "amount_concentration": 0.45
}
```

Classify:
```bash
python cli.py classify \
    --model my_model.pkl \
    --alert test_alert.json \
    --explain
```

## Step 5: Production Usage

### Real-Time Classification

```python
# Load once
system = AMLDriftDetectionSystem()
system.load_model('my_model.pkl')

# Classify as alerts arrive
def process_new_alert(alert_data):
    prediction, probability, explanation = system.classify_alert(alert_data)

    if prediction == 1:
        # Send to investigators
        send_to_investigation_queue(alert_data, explanation)
    else:
        # Auto-close
        close_alert(alert_data['alert_id'])

    return prediction, probability
```

### Batch Processing

```bash
python cli.py batch-classify \
    --model my_model.pkl \
    --data new_alerts.csv \
    --output classification_results.csv
```

### Drift Monitoring

```python
# Run weekly
system.load_model('my_model.pkl')

# Get last week's data
recent_data = get_recent_transactions(days=7)

# Detect drift
drift_detected, results = system.detect_drift(recent_data, "This Week")

if drift_detected:
    # Alert ops team
    send_alert("Drift detected - retraining recommended")

    # Retrain
    system.retrain_model(historical_data, recent_data)
    system.save_model('my_model_v2.pkl')
```

## Common Issues & Solutions

### Issue 1: "Missing required columns"

**Problem**: Your CSV doesn't have the expected column names

**Solution**: Rename your columns to match:
```python
import pandas as pd

df = pd.read_csv('your_data.csv')

# Rename columns
df.rename(columns={
    'your_alert_column': 'alert_id',
    'your_date_column': 'timestamp',
    'your_amount_column': 'amount',
    'your_type_column': 'transaction_type',
    'your_label_column': 'is_suspicious'
}, inplace=True)

# Save
df.to_csv('renamed_data.csv', index=False)
```

### Issue 2: "All labels are same class"

**Problem**: All alerts are labeled 0 or all labeled 1

**Solution**: Ensure you have both classes:
```python
print(df.groupby('alert_id')['is_suspicious'].first().value_counts())
```

Should show both 0 and 1. If not, check your labeling.

### Issue 3: "Model performance is poor"

**Possible causes and solutions**:

1. **Not enough data**
   - Need 1000+ transactions minimum
   - Solution: Collect more historical data

2. **Class imbalance**
   - Too few suspicious alerts
   - Solution: System handles this automatically, but ideally have >5% suspicious

3. **Poor feature values**
   - All transactions look the same
   - Solution: Ensure variety in transaction patterns

4. **Wrong time period**
   - Testing on very different time period than training
   - Solution: Use more recent training data

### Issue 4: "High false positive rate"

**Solutions**:
1. Adjust classification threshold:
```python
# Instead of using prediction directly
if probability > 0.7:  # Higher threshold = fewer false positives
    mark_suspicious()
```

2. Retrain with more recent data
3. Add more informative features
4. Use ensemble of models

## Next Steps

### 1. Optimize Performance
- Tune PSI threshold for your data
- Adjust retraining frequency
- Add custom features

### 2. Integrate with Your Systems
- Connect to your database
- Set up automated pipelines
- Create investigator dashboard

### 3. Monitor and Maintain
- Track FPR trends
- Review drift metrics
- Update model monthly

## Example Full Workflow

```python
from main import AMLDriftDetectionSystem
import pandas as pd

# Initialize
system = AMLDriftDetectionSystem()

# Load your data
print("Loading data...")
df = pd.read_csv('YOUR_TRANSACTIONS.csv')

# Process
print("Processing...")
system.upload_and_process_datasets('YOUR_TRANSACTIONS.csv')

# Engineer features
print("Engineering features...")
features_df = system.engineer_features()

# Split windows
print("Splitting windows...")
window1, window2, window3 = system.split_temporal_windows()

# Train
print("Training model...")
model = system.train_baseline_model(use_xgboost=True)

# Evaluate on validation set
print("\nValidation Performance:")
metrics_w2 = system.evaluate_model(window2, "Window 2")

# Evaluate on test set
print("\nTest Performance:")
metrics_w3 = system.evaluate_model(window3, "Window 3")

# Detect drift
print("\nDrift Detection:")
drift_detected, drift_results = system.detect_drift(window3, "Window 3")

# Decide on retraining
needs_retraining, reasons = system.decide_retraining(drift_detected, metrics_w3)

if needs_retraining:
    print("\n⚠️ Retraining recommended:")
    for reason in reasons:
        print(f"  - {reason}")

    # Retrain
    print("\nRetraining...")
    system.retrain_model(window1, window2)

    # Re-evaluate
    print("\nRetrained Model Performance:")
    metrics_retrained = system.evaluate_model(window3, "Window 3")

# Save final model
print("\nSaving model...")
system.save_model('production_model.pkl')

print("\n✅ Complete! Model ready for production.")
```

## Quick Reference

### Training
```bash
python cli.py train --data YOUR_DATA.csv --output model.pkl
```

### Classifying
```bash
python cli.py classify --model model.pkl --alert alert.json --explain
```

### Drift Detection
```bash
python cli.py detect-drift --model model.pkl --data new_data.csv
```

### Web Interface
```bash
streamlit run web_interface.py
```

## Need Help?

1. **Check documentation**
   - `USER_GUIDE.md` - Comprehensive guide
   - `README.md` - Overview
   - `PROJECT_SUMMARY.md` - Technical details

2. **Run examples**
   - `python example_workflow.py` - Full workflow
   - `python quick_start.py` - Interactive menu

3. **Verify installation**
   - `python check_installation.py`

## Success Checklist

Before deploying to production:

- [ ] Trained on at least 1000+ transactions
- [ ] Validated on separate time window
- [ ] Recall > 0.50
- [ ] FPR < 0.60
- [ ] Drift detection tested
- [ ] Explanations reviewed by investigators
- [ ] Model saved and backed up
- [ ] Integration tested with your systems
- [ ] Monitoring dashboard set up
- [ ] Retraining schedule planned

---

**You're ready to build your AML drift detection system! 🚀**

Start with the web interface for easiest experience, then move to Python API for production deployment.
