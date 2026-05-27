# Changes Made to Support Two Input Files

## Problem

The SynthAML dataset has **TWO separate CSV files**:
1. **synthetic_transactions.csv** - Transaction details (16M rows)
2. **synthetic_alerts_fixed.csv** - Alert labels (20K rows)

The original system only had one input field and didn't properly handle this separation.

## Solution

Updated the system to **require both files** and automatically join them on AlertID.

## Files Changed

### 1. data_processor.py ✅
**What changed:**
- Enhanced `process_and_join()` to detect SynthAML format
- Automatically renames columns (AlertID → alert_id, Outcome → is_suspicious, Size → amount)
- Properly joins transactions with alerts on alert_id
- Better error messages and validation

**Key code:**
```python
# Now properly handles:
# - Transactions: AlertID, Timestamp, Entry, Type, Size
# - Alerts: AlertID, Date, Outcome

if alerts_df is not None:
    # Merge on alert_id
    df = df.merge(alerts_df, on='alert_id', how='left')
```

### 2. main.py ✅
**What changed:**
- Updated `upload_and_process_datasets()` signature
- Better documentation explaining both files are needed
- Clearer warning messages if alerts file is missing

**Usage:**
```python
system.upload_and_process_datasets(
    transactions_path='synthetic_transactions.csv',
    alerts_path='synthetic_alerts_fixed.csv'  # Now properly supported
)
```

### 3. web_interface.py ✅
**What changed:**
- Added **TWO file upload fields** (side by side)
- Left: Transactions file
- Right: Alerts file
- Both files are saved and processed together
- Clear labels and help text

**UI:**
```
[Upload Transactions CSV]  [Upload Alerts CSV]
        ↓                          ↓
   Process Data Button
```

### 4. cli.py ✅
**What changed:**
- All commands now accept `--transactions` and `--alerts` flags
- Updated help text with examples
- Better error messages

**New syntax:**
```bash
# OLD (didn't work properly):
python cli.py train --data transactions.csv

# NEW (works correctly):
python cli.py train --transactions transactions.csv --alerts alerts.csv
```

**All commands updated:**
- `train` - Now requires --transactions, optional --alerts
- `detect-drift` - Now requires --transactions, optional --alerts
- `batch-classify` - Now requires --transactions, optional --alerts

### 5. test_synthaml.py ✅ NEW
**What it does:**
- Complete end-to-end test with your actual dataset
- Loads both files
- Processes, trains, evaluates, classifies
- Shows that everything works
- Takes 5-15 minutes to run

**Usage:**
```bash
python test_synthaml.py
```

### 6. SYNTHAML_QUICK_START.md ✅ NEW
**What it contains:**
- Step-by-step guide for your specific dataset
- Explains why you have two files
- All usage examples with correct paths
- Troubleshooting tips
- Performance expectations

## How It Works Now

### Before (Didn't work properly):
```
User uploads: transactions.csv
System: ❌ Can't find labels, creates placeholders
Result: ❌ All alerts labeled as 0 (benign)
```

### After (Works correctly):
```
User uploads: 
  - transactions.csv (16M transactions)
  - alerts.csv (20K labels)
  
System:
  1. Loads transactions → 16M rows
  2. Loads alerts → 20K rows with Outcome column
  3. Joins on AlertID → Links each transaction to its label
  4. Engineers features → Aggregates transactions per alert
  5. Ready for training ✅
```

## Quick Reference

### Python API
```python
system.upload_and_process_datasets(
    transactions_path='/path/to/synthetic_transactions.csv',
    alerts_path='/path/to/synthetic_alerts_fixed.csv'  # NEW!
)
```

### CLI
```bash
python cli.py train \
    --transactions /path/to/synthetic_transactions.csv \
    --alerts /path/to/synthetic_alerts_fixed.csv \
    --output model.pkl
```

### Web Interface
1. Run: `streamlit run web_interface.py`
2. Navigate to "Upload Data"
3. Upload transactions file (left box)
4. Upload alerts file (right box)
5. Click "Process Data"

## Testing

Run the test script to verify everything works:

```bash
cd /Users/sycamore/aml_drift_detection_system
python test_synthaml.py
```

Expected output:
```
✅ Successfully loaded and processed data!
   Total transactions: 16,000,000+
   Unique alerts: 20,000
   Benign alerts: ~18,000
   Suspicious alerts: ~2,000

✅ Successfully engineered features!
   Total features: 18+

✅ Model trained successfully!

✅ ALL TESTS PASSED!
```

## What Stayed the Same

- All feature engineering logic
- Model training approach
- Drift detection methods
- SHAP explanations
- Performance metrics
- Overall architecture

**Only the data input/loading was changed!**

## Backwards Compatibility

The system still works with single-file datasets:

```python
# If your data has labels in the transactions file already:
system.upload_and_process_datasets(
    transactions_path='combined_data.csv',
    alerts_path=None  # Optional
)
```

## Summary

✅ **Problem solved**: System now properly handles two separate CSV files

✅ **All interfaces updated**: Python API, CLI, Web UI

✅ **New documentation**: SYNTHAML_QUICK_START.md

✅ **Test script**: test_synthaml.py validates everything works

✅ **Your dataset ready**: Can now use synthetic_transactions.csv + synthetic_alerts_fixed.csv

## Next Steps

1. **Test the system:**
   ```bash
   python test_synthaml.py
   ```

2. **Try the web interface:**
   ```bash
   streamlit run web_interface.py
   ```

3. **Read the guide:**
   ```bash
   cat SYNTHAML_QUICK_START.md
   ```

4. **Train your model:**
   ```bash
   python cli.py train --transactions ... --alerts ...
   ```

Everything is now set up to work with your two-file dataset! 🎉
