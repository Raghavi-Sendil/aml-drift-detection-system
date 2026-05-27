# Working with Large Files (>200MB)

## Your Dataset

Your synthetic_transactions.csv is **904 MB** - too large for browser upload!

## 🚀 Best Options for Large Files

### Option 1: Direct Python Script ⭐ RECOMMENDED

**Fastest and easiest for large files!**

1. Edit the file paths in `run_with_paths.py`:
```python
TRANSACTIONS_PATH = "/Users/sycamore/Downloads/synthetic_transactions.csv"
ALERTS_PATH = "/Users/sycamore/Downloads/synthetic_alerts_fixed.csv"
```

2. Run:
```bash
python run_with_paths.py
```

**That's it!** The script will:
- ✅ Load both files directly (no upload)
- ✅ Process and join them
- ✅ Engineer features
- ✅ Train model
- ✅ Evaluate performance
- ✅ Detect drift
- ✅ Save trained model
- ⏱️ Takes 10-20 minutes total

### Option 2: Command Line Interface

```bash
python cli.py train \
    --transactions /Users/sycamore/Downloads/synthetic_transactions.csv \
    --alerts /Users/sycamore/Downloads/synthetic_alerts_fixed.csv \
    --model-type xgboost \
    --output my_model.pkl
```

**Advantages:**
- ✅ No file upload needed
- ✅ Direct file access
- ✅ Can run in background
- ✅ Easy to automate

### Option 3: Web Interface with File Paths

The web interface now has **TWO input modes**:

1. Launch web interface:
```bash
streamlit run web_interface.py
```

2. Go to "Upload Data" page

3. Select: **"📁 File Path (Recommended for large files > 200MB)"**

4. Enter your file paths:
   - Transactions: `/Users/sycamore/Downloads/synthetic_transactions.csv`
   - Alerts: `/Users/sycamore/Downloads/synthetic_alerts_fixed.csv`

5. Click "Process Data"

**Advantages:**
- ✅ Visual interface
- ✅ No upload (reads directly from path)
- ✅ See progress in real-time
- ✅ Interactive dashboards

## ⚡ Performance Comparison

| Method | File Access | Speed | Best For |
|--------|-------------|-------|----------|
| **run_with_paths.py** | Direct | ⚡⚡⚡ Fastest | Large files, automation |
| **CLI** | Direct | ⚡⚡⚡ Fastest | Batch processing, scripts |
| **Web (File Path)** | Direct | ⚡⚡⚡ Fast | Visual interface needed |
| **Web (Upload)** | Browser upload | 🐌 Very slow | Small files only (<50MB) |

## 📊 Expected Performance

### With Your Dataset (904MB + 368KB)

**Using Direct File Access:**
```
Loading data:          2-5 minutes
Engineering features:  1-2 minutes
Training XGBoost:      5-15 minutes
Evaluation:           1-2 minutes
Total:                10-25 minutes
```

**Using Browser Upload (NOT RECOMMENDED):**
```
Uploading file:       15-30 minutes ❌ SLOW!
Processing:           2-5 minutes
Training:             5-15 minutes
Total:                25-50 minutes
```

## 🎯 Recommended Workflow

### For First Time

Use `run_with_paths.py`:

```bash
# 1. Edit the file - update paths at the top
nano run_with_paths.py

# 2. Run it
python run_with_paths.py

# 3. Wait 10-20 minutes
# 4. Model saved as: my_aml_model.pkl
```

### For Regular Use

Use CLI:

```bash
# Train
python cli.py train \
    --transactions /path/to/transactions.csv \
    --alerts /path/to/alerts.csv \
    --output model.pkl

# Classify
python cli.py classify \
    --model model.pkl \
    --alert alert.json \
    --explain

# Detect drift
python cli.py detect-drift \
    --model model.pkl \
    --transactions new_transactions.csv \
    --alerts new_alerts.csv
```

### For Exploration

Use web interface with file paths:

```bash
streamlit run web_interface.py
# Then select "File Path" option
```

## 💡 Tips for Large Files

### 1. Use SSD Storage
If your files are on external HDD, copy to SSD first:
```bash
cp /Volumes/ExternalDrive/data.csv ~/Downloads/
```

### 2. Check Available Memory
```bash
# Your file is 904MB, need at least 4GB RAM free
# Check with:
top -l 1 | grep PhysMem
```

### 3. Sample for Testing
To test quickly, use first 1M rows:
```bash
head -1000000 synthetic_transactions.csv > sample_transactions.csv
```

### 4. Run in Background
For long training runs:
```bash
nohup python run_with_paths.py > output.log 2>&1 &
```

Check progress:
```bash
tail -f output.log
```

### 5. Use Fast Model for Testing
Edit `run_with_paths.py`:
```python
MODEL_TYPE = 'logistic'  # Faster than xgboost for testing
```

Then switch to xgboost for final model:
```python
MODEL_TYPE = 'xgboost'  # Better performance
```

## 🔧 Troubleshooting

### "Memory Error"
Your dataset is large. Solutions:
1. Close other applications
2. Use a machine with 8+ GB RAM
3. Sample the data (first 50% of alerts)

### "File Too Large" in Browser
Don't use browser upload! Use:
- ✅ `run_with_paths.py` (recommended)
- ✅ CLI commands
- ✅ Web interface with file paths

### "Processing Very Slow"
Normal for 16M transactions. Speed it up:
1. Use SSD instead of HDD
2. Use XGBoost (parallelized)
3. Close other applications
4. Be patient (10-20 min is normal)

## 📝 Quick Reference

### Your File Paths
```
Transactions: /Users/sycamore/Downloads/synthetic_transactions.csv (904 MB)
Alerts:       /Users/sycamore/Downloads/synthetic_alerts_fixed.csv (368 KB)
```

### Fastest Method
```bash
# Edit paths in run_with_paths.py, then:
python run_with_paths.py
```

### CLI Method
```bash
python cli.py train \
    --transactions /Users/sycamore/Downloads/synthetic_transactions.csv \
    --alerts /Users/sycamore/Downloads/synthetic_alerts_fixed.csv \
    --output model.pkl
```

### Web Interface Method
```bash
streamlit run web_interface.py
# Select "File Path" option
# Enter paths above
# Click "Process Data"
```

## ✅ Summary

For your **904MB file**:

1. ❌ **Don't** use browser upload (too slow)
2. ✅ **Do** use `run_with_paths.py` (fastest)
3. ✅ **Do** use CLI commands (flexible)
4. ✅ **Do** use web interface with file paths (visual)

All three methods read files directly - **no upload needed!**

**Start here:**
```bash
python run_with_paths.py
```

Good luck! 🚀
