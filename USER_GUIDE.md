# AML Drift Detection System - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding the System](#understanding-the-system)
3. [Usage Methods](#usage-methods)
4. [Data Preparation](#data-preparation)
5. [Training Models](#training-models)
6. [Detecting Drift](#detecting-drift)
7. [Classifying Alerts](#classifying-alerts)
8. [Understanding Explanations](#understanding-explanations)
9. [Production Deployment](#production-deployment)
10. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **Check Python version** (requires Python 3.8+)
```bash
python --version
```

2. **Install dependencies**
```bash
cd /Users/sycamore/aml_drift_detection_system
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python check_installation.py
```

### Quick Start

**Option 1: Interactive Menu**
```bash
python quick_start.py
```

**Option 2: Full Workflow Example**
```bash
python example_workflow.py
```

**Option 3: Web Interface**
```bash
streamlit run web_interface.py
```

## Understanding the System

### What Does This System Do?

The AML Drift Detection System helps financial institutions:

1. **Classify AML Alerts** - Determines if an alert is suspicious or benign
2. **Detect Drift** - Monitors when transaction patterns change over time
3. **Decide on Retraining** - Automatically determines when models need updating
4. **Explain Decisions** - Provides human-readable explanations using SHAP

### Key Concepts

#### 1. Features
The system creates 18+ behavioral features from transaction data:
- Transaction frequency and amounts
- Wire transfer patterns
- Burst behavior (transaction clustering)
- Round amount ratios (suspicious indicator)
- Time-based patterns (weekend, night transactions)

#### 2. Drift Detection
Monitors three statistical measures:
- **PSI** (Population Stability Index): Overall distribution change
- **KS Test** (Kolmogorov-Smirnov): Maximum distribution difference
- **KL Divergence** (Kullback-Leibler): Information loss measurement

#### 3. Temporal Windows
Data is split chronologically:
- **Window 1 (40%)**: Training data
- **Window 2 (30%)**: Validation data
- **Window 3 (30%)**: Test data (future behavior)

#### 4. Retraining Logic
Model is retrained when:
- PSI > 0.2 (significant drift)
- False Positive Rate > 55%
- Accuracy < 50%

## Usage Methods

### Method 1: Python API

```python
from main import AMLDriftDetectionSystem

# Initialize
system = AMLDriftDetectionSystem()

# Upload data
system.upload_and_process_datasets('transactions.csv')

# Engineer features
system.engineer_features()

# Split windows
system.split_temporal_windows()

# Train
system.train_baseline_model(use_xgboost=True)

# Detect drift
drift_detected, results = system.detect_drift(new_data)

# Classify
prediction, probability, explanation = system.classify_alert(alert_features)

# Explain
system.explain_decision(explanation)
```

### Method 2: Command-Line Interface

```bash
# Train model
python cli.py train --data transactions.csv --output model.pkl

# Classify single alert
python cli.py classify --model model.pkl --alert alert.json --explain

# Detect drift
python cli.py detect-drift --model model.pkl --data new_data.csv

# Batch classify
python cli.py batch-classify --model model.pkl --data alerts.csv
```

### Method 3: Web Interface

```bash
streamlit run web_interface.py
```

Then navigate to http://localhost:8501 in your browser.

## Data Preparation

### Required CSV Format

Your `transactions.csv` must have these columns:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| alert_id | int/string | 12345 | Groups related transactions |
| timestamp | datetime | 2024-01-15 10:30:00 | When transaction occurred |
| amount | float | 5000.00 | Transaction amount |
| transaction_type | string | wire | credit/debit/wire/transfer |
| is_suspicious | int | 1 | 1=suspicious, 0=benign |

### Example CSV

```csv
alert_id,timestamp,amount,transaction_type,is_suspicious
1001,2024-01-01 09:15:00,8500.00,wire,1
1001,2024-01-01 10:30:00,9000.00,wire,1
1001,2024-01-01 11:45:00,8750.00,wire,1
1002,2024-01-02 14:20:00,150.50,debit,0
1002,2024-01-02 15:10:00,200.00,credit,0
```

### Alternative Column Names

The system automatically recognizes these alternatives:

- **alert_id**: alertid, alert, id, alert_key
- **timestamp**: date, datetime, transaction_date, tx_date
- **amount**: transaction_amount, tx_amount, value, usd_amount
- **transaction_type**: type, tx_type, payment_type

### Missing Columns?

If your data doesn't have `is_suspicious` labels:
- The system will create placeholder labels (all 0s)
- You can train on historical data where labels are known
- Then classify new unlabeled alerts

## Training Models

### Model Types

**1. XGBoost (Recommended)**
- High performance
- Handles non-linear patterns
- Better for complex laundering strategies
```python
system.train_baseline_model(use_xgboost=True)
```

**2. Logistic Regression**
- Interpretable
- Fast training
- Good baseline
```python
system.train_baseline_model(use_xgboost=False)
```

### Training Process

```python
# 1. Load data
system.upload_and_process_datasets('transactions.csv')

# 2. Engineer features
features_df = system.engineer_features()

# 3. Split windows
window1, window2, window3 = system.split_temporal_windows(
    train_ratio=0.4,  # 40% for training
    eval_ratio=0.3,   # 30% for validation
    test_ratio=0.3    # 30% for testing
)

# 4. Train
model = system.train_baseline_model(use_xgboost=True)

# 5. Evaluate
metrics = system.evaluate_model(window2, "Validation")
print(f"Recall: {metrics['recall']:.3f}")
print(f"FPR: {metrics['fpr']:.3f}")

# 6. Save
system.save_model('my_model.pkl')
```

### Understanding Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Recall** | % of suspicious alerts caught | > 0.60 |
| **Precision** | % of flagged alerts truly suspicious | > 0.20 |
| **FPR** | % of benign alerts wrongly flagged | < 0.50 |
| **Accuracy** | Overall correctness | > 0.55 |

**Important**: In AML, recall is more critical than precision. Missing a suspicious alert (false negative) is worse than investigating a benign one (false positive).

## Detecting Drift

### What is Drift?

Drift occurs when transaction patterns change over time due to:
- Evolving customer behavior
- New financial technologies
- Changing money laundering tactics
- Seasonal variations

### Running Drift Detection

```python
# Train on historical data
system.train_baseline_model(use_xgboost=True)

# Load new data
new_data = pd.read_csv('new_transactions.csv')
new_features = system.feature_engineer.create_features(new_data)

# Detect drift
drift_detected, drift_results = system.detect_drift(new_features, "New Data")

# Check results
for feature, stats in drift_results.items():
    if stats['drift_detected']:
        print(f"⚠️ {feature}: PSI = {stats['psi']:.3f}")
```

### Interpreting Drift Scores

**PSI (Population Stability Index)**
- < 0.1: No significant change
- 0.1 - 0.2: Moderate change, monitor closely
- > 0.2: **Significant drift, retraining recommended**

**KS Statistic**
- < 0.1: Stable
- > 0.1: Drift detected

### Automated Retraining

```python
# Evaluate on new data
metrics = system.evaluate_model(new_data, "Current")

# Decide if retraining needed
needs_retraining, reasons = system.decide_retraining(
    drift_detected,
    metrics
)

if needs_retraining:
    print("Reasons for retraining:")
    for reason in reasons:
        print(f"  - {reason}")

    # Retrain with combined data
    system.retrain_model(
        historical_data=window1,
        recent_data=window2
    )
```

## Classifying Alerts

### Single Alert Classification

```python
# Define alert features
alert = {
    'transaction_count': 15,
    'mean_transaction_size': 8500.00,
    'max_transaction_size': 15000.00,
    'wire_count': 8,
    'burst_score': 0.75,
    # ... other features
}

# Classify
prediction, probability, explanation = system.classify_alert(alert)

# Result
if prediction == 1:
    print(f"⚠️ SUSPICIOUS (Confidence: {probability:.1%})")
else:
    print(f"✅ BENIGN (Confidence: {(1-probability):.1%})")
```

### Batch Classification

```bash
# Using CLI
python cli.py batch-classify \
    --model trained_model.pkl \
    --data alerts_to_classify.csv \
    --output results.csv
```

Results CSV will contain:
```csv
alert_id,prediction,probability,classification
1001,1,0.87,suspicious
1002,0,0.23,benign
1003,1,0.65,suspicious
```

### Real-Time Classification

For production systems:

```python
# Load model once at startup
system = AMLDriftDetectionSystem()
system.load_model('production_model.pkl')

# Then classify alerts as they arrive
def classify_new_alert(transaction_data):
    # Process transactions
    features = extract_features(transaction_data)

    # Classify
    prediction, probability, explanation = system.classify_alert(features)

    # Log decision
    log_decision(alert_id, prediction, probability, explanation)

    return prediction, probability
```

## Understanding Explanations

### SHAP Values

SHAP (SHapley Additive exPlanations) shows:
- **Which features** contributed to the decision
- **How much** each feature contributed
- **Direction** (increasing or decreasing suspicion)

### Example Explanation

```
Top 5 Contributing Features:
┌─────────────────────┬──────────┬─────────┬───────────────┐
│ Feature             │ Value    │ Impact  │ Direction     │
├─────────────────────┼──────────┼─────────┼───────────────┤
│ wire_count          │ 8.00     │ 0.2847  │ → Suspicious  │
│ burst_score         │ 0.80     │ 0.2134  │ → Suspicious  │
│ mean_tx_size        │ 8500.00  │ 0.1923  │ → Suspicious  │
│ round_amount_ratio  │ 0.60     │ 0.1456  │ → Suspicious  │
│ transaction_velocity│ 6.00     │ 0.1102  │ → Suspicious  │
└─────────────────────┴──────────┴─────────┴───────────────┘
```

### Interpretation

**High Wire Count (8 transfers)**
- **Why suspicious**: Money launderers often use wire transfers for fast, cross-border movement
- **Normal range**: 0-2 wires per alert
- **This value**: 8 is significantly elevated

**High Burst Score (0.80)**
- **Why suspicious**: Transactions clustered in short time period
- **Interpretation**: All 8 wires occurred within 3 hours
- **Pattern**: Suggests rapid layering of funds

**Round Amounts (60%)**
- **Why suspicious**: 60% of transactions are round numbers ($10,000, $50,000)
- **Interpretation**: Money launderers often use round amounts
- **Normal behavior**: Legitimate transactions have varied amounts

### Investigator Recommendations

The system provides actionable guidance:

```
INVESTIGATOR RECOMMENDATION
─────────────────────────────────────────────
Priority Level: HIGH PRIORITY
Recommended Action: Immediate investigation

Investigation Focus:
  1. Examine wire transfer destinations
  2. Review transaction timing patterns
  3. Analyze relationship between accounts
  4. Check for structuring behavior
```

## Production Deployment

### Architecture

```
┌─────────────────────┐
│   AML System        │
│  (Your Bank's IT)   │
└─────────┬───────────┘
          │
          ↓
┌─────────────────────┐
│  Transaction        │
│  Monitoring         │
│  System             │
└─────────┬───────────┘
          │
          ↓
┌─────────────────────┐
│  Drift Detection    │
│  Model              │
│  (This System)      │
└─────────┬───────────┘
          │
          ↓
┌─────────────────────┐
│  Alert              │
│  Classification     │
│  + Explanation      │
└─────────┬───────────┘
          │
          ↓
┌─────────────────────┐
│  Investigator       │
│  Dashboard          │
└─────────────────────┘
```

### Integration Steps

1. **Extract Transaction Data**
```python
# From your database
transactions_df = pd.read_sql("""
    SELECT alert_id, timestamp, amount, type, is_suspicious
    FROM aml_alerts
    WHERE alert_date >= '2024-01-01'
""", connection)
```

2. **Train Initial Model**
```python
system = AMLDriftDetectionSystem()
system.upload_and_process_datasets(transactions_df)
system.engineer_features()
system.split_temporal_windows()
system.train_baseline_model(use_xgboost=True)
system.save_model('production_model.pkl')
```

3. **Schedule Drift Monitoring**
```python
# Run daily/weekly
def monitor_drift():
    system.load_model('production_model.pkl')
    recent_data = get_recent_transactions()

    drift_detected, results = system.detect_drift(recent_data)

    if drift_detected:
        alert_ops_team()
        retrain_model()
```

4. **Real-Time Classification**
```python
# When new alert triggers
def handle_new_alert(alert_id):
    system.load_model('production_model.pkl')

    transactions = get_alert_transactions(alert_id)
    features = extract_features(transactions)

    prediction, probability, explanation = system.classify_alert(features)

    if prediction == 1 and probability > 0.7:
        send_to_investigator(alert_id, explanation)
    else:
        auto_close_alert(alert_id)
```

### Performance Optimization

**1. Model Caching**
```python
# Load once, reuse many times
_cached_system = None

def get_system():
    global _cached_system
    if _cached_system is None:
        _cached_system = AMLDriftDetectionSystem()
        _cached_system.load_model('production_model.pkl')
    return _cached_system
```

**2. Batch Processing**
```python
# Process multiple alerts together
def classify_batch(alert_ids, batch_size=100):
    for i in range(0, len(alert_ids), batch_size):
        batch = alert_ids[i:i+batch_size]
        # Process batch
```

**3. Async Processing**
```python
import asyncio

async def classify_async(alert_features):
    # Non-blocking classification
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        system.classify_alert,
        alert_features
    )
```

## Troubleshooting

### Common Issues

**1. "Model not trained yet"**
```python
# Solution: Train model first
system.train_baseline_model()
```

**2. "Missing required columns"**
```python
# Solution: Check your CSV has required columns
print(df.columns)

# Or rename columns
df.rename(columns={
    'alert_id_column': 'alert_id',
    'date_column': 'timestamp'
}, inplace=True)
```

**3. "All labels are the same class"**
```python
# Solution: Ensure you have both suspicious (1) and benign (0) alerts
print(df['is_suspicious'].value_counts())
```

**4. "SHAP not available"**
```bash
# Solution: Install SHAP
pip install shap
```

**5. "Out of memory"**
```python
# Solution: Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Performance Issues

**Slow training**
- Use XGBoost with n_jobs=-1 (already default)
- Reduce n_estimators if very slow
- Consider using a sample of data for initial testing

**Slow drift detection**
- Drift detection is O(n), should be fast
- If slow, check data size
- Consider sampling for very large datasets

### Getting Help

1. Check this guide
2. Review example_workflow.py
3. Run check_installation.py
4. Check README.md

## Best Practices

### 1. Model Maintenance
- **Retrain monthly** or when significant drift detected
- Keep **6-12 months** of historical data
- Archive old models with dates

### 2. Threshold Tuning
- Adjust PSI threshold based on your data volatility
- Lower threshold (0.15) for stable environments
- Higher threshold (0.25) for volatile environments

### 3. Explanation Usage
- Always provide explanations to investigators
- Log SHAP values for audit trail
- Use top 5 features in investigator dashboard

### 4. Monitoring
- Track FPR trends over time
- Monitor drift metrics daily
- Set up alerts for FPR > 60%

### 5. Validation
- Always validate on hold-out test set
- Test on recent data before deployment
- Compare against rule-based baseline

## Appendix

### Feature Glossary

| Feature | Description | Suspicious When |
|---------|-------------|----------------|
| transaction_count | Number of transactions | Very high or very low |
| mean_transaction_size | Average amount | Just below reporting threshold |
| wire_count | Number of wire transfers | Elevated (>3) |
| burst_score | Transaction clustering | High (>0.7) |
| round_amount_ratio | % round numbers | High (>0.5) |
| net_flow | Total money movement | Very high |
| transaction_velocity | Frequency rate | Very high |
| night_transaction_ratio | % at night | High |
| weekend_transaction_ratio | % on weekend | High |

### Metric Targets

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Recall | > 0.70 | 0.50-0.70 | < 0.50 |
| FPR | < 0.40 | 0.40-0.55 | > 0.55 |
| Accuracy | > 0.60 | 0.50-0.60 | < 0.50 |

### Research Reference

This system implements methodology from:

**"AML Alert Classification Under Transaction Behavior Shifts for False Positive Reduction"**

Raghavi Sendil, Sai Santhosh Channaka, Sahil Anjum
Pace University, 2024

Key contributions:
- Temporal window-based evaluation
- Multi-metric drift detection (PSI, KS, KL)
- Drift-aware retraining strategy
- Feature engineering for AML detection
