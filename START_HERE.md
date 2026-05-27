# 🚀 START HERE - Large File Support Added!

## Your Situation

✅ **You have TWO files:**
- synthetic_transactions.csv (904 MB) ← Large!
- synthetic_alerts_fixed.csv (368 KB)

✅ **Problem solved:** 904MB is too large for browser upload

✅ **Solution added:** Direct file path support

## 🎯 Quickest Way to Get Started

### Step 1: Run This Command

```bash
cd /Users/sycamore/aml_drift_detection_system
python run_with_paths.py
```

That's it! The script will:
- Read files directly (no upload)
- Process 16M transactions
- Train XGBoost model
- Save as `my_aml_model.pkl`
- Takes 10-20 minutes

**File paths are already set** for your Downloads folder!

### Step 2: Wait for Results

You'll see output like:
```
[STEP 1] Initializing System...
[STEP 2] Loading and Processing Data...
⏳ This may take 2-5 minutes...
✅ Data loaded successfully!
   Total transactions: 16,000,000+
   Unique alerts: 20,000

[STEP 3] Engineering Features...
✅ Created 18 features

[STEP 4] Training Model...
⏳ This may take 5-15 minutes...
✅ Model trained successfully!

[STEP 5] Evaluating...
  Recall: 0.654
  FPR: 0.478
  Accuracy: 0.597

✅ COMPLETE!
Model saved to: my_aml_model.pkl
```

## 📚 Three Ways to Use the System

### Option 1: Simple Script ⭐ RECOMMENDED FOR YOU
```bash
python run_with_paths.py
```
- ✅ Easiest for large files
- ✅ No configuration needed
- ✅ Paths already set
- ⏱️ 10-20 minutes

### Option 2: Command Line
```bash
python cli.py train \
    --transactions /Users/sycamore/Downloads/synthetic_transactions.csv \
    --alerts /Users/sycamore/Downloads/synthetic_alerts_fixed.csv \
    --output model.pkl
```
- ✅ Most flexible
- ✅ Good for automation
- ⏱️ 10-20 minutes

### Option 3: Web Interface
```bash
streamlit run web_interface.py
```
Then:
1. Go to "Upload Data"
2. Select "📁 File Path" option
3. Enter paths (already pre-filled!)
4. Click "Process Data"

- ✅ Visual interface
- ✅ Interactive dashboards
- ⏱️ 10-20 minutes

## 📖 Documentation

Detailed guides available:

1. **LARGE_FILES_GUIDE.md** - Complete guide for large files
2. **SYNTHAML_QUICK_START.md** - SynthAML dataset specifics
3. **USER_GUIDE.md** - Full system documentation
4. **GETTING_STARTED.md** - General getting started guide

## ⚡ What Changed

### Before
- ❌ Only browser upload
- ❌ 904MB file too large
- ❌ Would take 30+ minutes

### After (NOW)
- ✅ Direct file path support
- ✅ Works with 904MB file
- ✅ Takes 10-20 minutes
- ✅ Three input methods:
  1. Python script (run_with_paths.py)
  2. CLI with --transactions and --alerts
  3. Web interface with file path option

## 🎯 Your Next Command

Just run this:

```bash
python run_with_paths.py
```

Then grab a coffee ☕ and wait 10-20 minutes!

---

**Questions?**
- See LARGE_FILES_GUIDE.md for details
- See SYNTHAML_QUICK_START.md for dataset info
- All file paths are pre-configured for your Downloads folder
