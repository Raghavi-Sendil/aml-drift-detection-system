# AML Alert Classification System with Drift Detection

A comprehensive Anti-Money Laundering (AML) alert classification system based on the research paper: **"AML Alert Classification Under Transaction Behavior Shifts for False Positive Reduction"**

## Features

✅ **Automated Data Processing**
- Upload transaction and alert datasets
- Automatic preprocessing and table joining
- Handle missing values and data type conversions

✅ **Feature Engineering**
- Transaction count, mean/max size, net flow
- Burst score (transaction clustering)
- Average time between transactions
- Credit/Debit/Wire transfer counts
- 18+ behavioral features per research methodology

✅ **Drift Detection**
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) Test
- Kullback-Leibler (KL) Divergence
- Real-time monitoring of feature distribution shifts

✅ **Smart Retraining**
- Automatic drift detection
- Decision logic for retraining
- Combines historical and recent data
- Maintains model performance over time

✅ **Classification Models**
- Logistic Regression (interpretable baseline)
- XGBoost (high-performance ensemble)
- Handles class imbalance
- Optimized for false positive reduction

✅ **SHAP Explainability**
- Detailed explanation for each decision
- Top contributing features
- Human-readable interpretations
- Investigator recommendations

## Installation

### 1. Clone or download this repository

```bash
cd /Users/sycamore/aml_drift_detection_system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import pandas, numpy, sklearn, xgboost, shap; print('✓ All dependencies installed')"
```

## Quick Start

### Option 1: Run Full Workflow Example

```bash
python example_workflow.py
```

This will:
1. Generate synthetic AML data
2. Process and engineer features
3. Train baseline models
4. Detect drift across time windows
5. Retrain if needed
6. Classify and explain sample alerts

### Option 2: Use Your Own Data

Create a Python script:

```python
from main import AMLDriftDetectionSystem

# Initialize system
system = AMLDriftDetectionSystem()

# Upload your datasets
processed_data = system.upload_and_process_datasets(
    transactions_path='your_transactions.csv',
    alerts_path='your_alerts.csv'  # Optional
)

# Engineer features
features_df = system.engineer_features()

# Split temporal windows (40% train, 30% eval, 30% test)
window1, window2, window3 = system.split_temporal_windows()

# Train model
baseline_model = system.train_baseline_model(use_xgboost=True)

# Evaluate
metrics = system.evaluate_model(window2, "Window 2")

# Detect drift
drift_detected, drift_results = system.detect_drift(window3, "Window 3")

# Decide on retraining
needs_retraining, reasons = system.decide_retraining(drift_detected, metrics)

# Retrain if needed
if needs_retraining:
    system.retrain_model(window1, window2)

# Classify new alert
prediction, probability, explanation = system.classify_alert(new_alert_features)

# Explain decision
system.explain_decision(explanation)

# Save model
system.save_model('my_aml_model.pkl')
```

## Data Format

### Required Columns in transactions.csv

| Column | Type | Description |
|--------|------|-------------|
| `alert_id` | int/string | Alert identifier (groups related transactions) |
| `timestamp` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (USD or local currency) |
| `transaction_type` | string | Type: 'credit', 'debit', 'wire', etc. |
| `is_suspicious` | int | Label: 1=suspicious, 0=benign |

### Optional Columns

- `account_id`: Account identifier
- `customer_id`: Customer identifier
- Any additional transaction attributes

### Example CSV Format

```csv
alert_id,timestamp,amount,transaction_type,is_suspicious
1,2024-01-01 10:30:00,5000.00,wire,1
1,2024-01-01 11:15:00,4500.00,wire,1
2,2024-01-02 09:00:00,150.00,debit,0
2,2024-01-02 14:30:00,200.00,credit,0
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AML Drift Detection System                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Data Upload & Processing (data_processor.py)            │
│     • Load CSVs                                              │
│     • Join transactions with alerts                          │
│     • Handle missing values                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Feature Engineering (feature_engineer.py)                │
│     • Aggregate by alert                                     │
│     • Create 18+ behavioral features                         │
│     • Burst score, net flow, ratios, etc.                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Temporal Window Split                                    │
│     • Window 1 (40%): Training                               │
│     • Window 2 (30%): Evaluation                             │
│     • Window 3 (30%): Testing                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Model Training (model_trainer.py)                        │
│     • Logistic Regression / XGBoost                          │
│     • Class imbalance handling                               │
│     • Baseline statistics storage                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Drift Detection (drift_detector.py)                      │
│     • PSI: Population Stability Index                        │
│     • KS: Kolmogorov-Smirnov Test                           │
│     • KL: Kullback-Leibler Divergence                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Retraining Decision                                      │
│     • Check drift thresholds                                 │
│     • Check FPR increase                                     │
│     • Check accuracy drop                                    │
│     • Retrain if needed                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Alert Classification                                     │
│     • Predict: 0 (benign) or 1 (suspicious)                 │
│     • Probability score                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  8. SHAP Explanation (explainer.py)                          │
│     • Calculate SHAP values                                  │
│     • Top contributing features                              │
│     • Human-readable interpretation                          │
│     • Investigator recommendation                            │
└─────────────────────────────────────────────────────────────┘
```

## Methodology (From Research Paper)

### Feature Engineering

Following the paper's methodology, the system creates:

1. **Transaction Count**: Number of transactions per alert
2. **Mean Transaction Size**: Average transaction amount
3. **Max Transaction Size**: Largest transaction in alert
4. **Net Flow**: Total money movement (credits - debits)
5. **Burst Score**: Measures transaction clustering in time
6. **Avg Time Between Transactions**: Transaction frequency
7. **Credit/Debit/Wire Counts**: Payment type distribution

Plus 11 additional derived features for enhanced detection.

### Drift Detection

Three statistical methods as per the paper:

#### 1. Population Stability Index (PSI)
```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```
- PSI < 0.1: No significant change
- PSI 0.1-0.2: Moderate change
- PSI > 0.2: Significant drift (retraining recommended)

#### 2. Kolmogorov-Smirnov Test
- Measures maximum distance between cumulative distributions
- Higher KS statistic = more drift

#### 3. Kullback-Leibler Divergence
- Measures information loss between distributions
- Higher KL = more drift

### Retraining Strategy

The system automatically retrains when:
- PSI > 0.2 (significant drift detected)
- False Positive Rate > 0.55 (too many false alarms)
- Accuracy < 0.50 (poor performance)

Retraining combines:
- Historical data (Window 1): 40%
- Recent data (Window 2): 30%
- Maintains model stability while adapting to new patterns

## Performance Metrics

The system tracks key AML metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Recall** | % of suspicious alerts detected | > 0.60 |
| **False Positive Rate** | % of benign alerts flagged | < 0.50 |
| **Accuracy** | Overall correctness | > 0.55 |
| **Precision** | % of flagged alerts that are truly suspicious | > 0.20 |

## Example Output

### Classification Result

```
[STEP 9] Classifying Alert...
================================================================================

⚠ SUSPICIOUS (Report)
Probability of being suspicious: 78%

[STEP 10] Explanation (Why this decision?)...
================================================================================

Top 5 Contributing Features:
Feature                        Value           Impact          Direction
--------------------------------------------------------------------------------
wire_count                     8.00            0.2847          → Suspicious
burst_score                    0.80            0.2134          → Suspicious
mean_transaction_size          8500.00         0.1923          → Suspicious
round_amount_ratio             0.60            0.1456          → Suspicious
transaction_velocity           6.00            0.1102          → Suspicious

================================================================================
DETAILED EXPLANATION:
================================================================================

• wire_count = 8.00
  This feature increases suspicion by 0.2847
  Higher values indicate unusual transaction behavior

• burst_score = 0.80
  This feature increases suspicion by 0.2134
  High burst score indicates clustering of transactions in short time

• mean_transaction_size = 8500.00
  This feature increases suspicion by 0.1923
  Average transaction of $8,500.00 is unusually large
```

### Investigator Recommendation

```
================================================================================
INVESTIGATOR RECOMMENDATION
================================================================================

Priority Level: HIGH PRIORITY
Recommended Action: Immediate investigation recommended

Investigation Focus:
  1. Examine Wire Count pattern
  2. Examine Burst Score pattern
  3. Examine Mean Transaction Size pattern
================================================================================
```

## Module Documentation

### main.py
Main system orchestrator. Coordinates all components.

**Key Methods:**
- `upload_and_process_datasets()`: Load and preprocess data
- `engineer_features()`: Create behavioral features
- `train_baseline_model()`: Train classification model
- `detect_drift()`: Run drift detection
- `classify_alert()`: Classify new alert
- `explain_decision()`: Generate SHAP explanation

### data_processor.py
Handles data preprocessing and joining.

**Key Methods:**
- `process_and_join()`: Preprocess and join datasets
- `aggregate_by_alert()`: Aggregate transactions to alert-level

### feature_engineer.py
Creates features per research methodology.

**Key Methods:**
- `create_features()`: Generate all features
- `simulate_drift()`: Simulate behavioral shifts

### drift_detector.py
Statistical drift detection.

**Key Methods:**
- `compute_baseline_stats()`: Calculate baseline distributions
- `detect_drift()`: Run PSI, KS, KL tests
- `generate_drift_report()`: Create drift summary

### model_trainer.py
Model training and evaluation.

**Key Methods:**
- `train()`: Train Logistic Regression or XGBoost
- `evaluate()`: Calculate performance metrics
- `predict()`: Make predictions

### explainer.py
SHAP-based explainability.

**Key Methods:**
- `explain_prediction()`: Generate SHAP explanation
- `create_text_explanation()`: Human-readable explanation
- `generate_report_recommendation()`: Investigator guidance

## Research Paper Reference

This system implements the methodology from:

**"AML Alert Classification Under Transaction Behavior Shifts for False Positive Reduction"**

Authors: Raghavi Sendil, Sai Santhosh Channaka, Sahil Anjum
Institution: Pace University, Seidenberg School of CSIS

Key findings implemented:
- Drift increases model sensitivity but also false positives
- Retraining improves stability and reduces FPR
- PSI, KS, and KL divergence effectively detect drift
- Temporal window-based evaluation is critical

## License

This system is for educational and research purposes.

## Support

For questions or issues:
1. Check example_workflow.py for usage examples
2. Review module documentation above
3. Ensure data format matches requirements

## Version

Version: 1.0.0
Last Updated: 2024

