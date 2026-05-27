"""
Simple script to run the AML system with file paths
Perfect for large datasets that can't be uploaded through web interface

Usage:
  python run_with_paths.py
"""

from main import AMLDriftDetectionSystem
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Your file paths
TRANSACTIONS_PATH = "/Users/sycamore/Downloads/synthetic_transactions.csv"
ALERTS_PATH = "/Users/sycamore/Downloads/synthetic_alerts_fixed.csv"

# Output model name
MODEL_OUTPUT = "my_aml_model.pkl"

# Model type: 'xgboost' or 'logistic'
MODEL_TYPE = 'xgboost'  # XGBoost is recommended for better performance

# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("AML DRIFT DETECTION SYSTEM - DIRECT FILE PATH MODE")
    print("=" * 80)

    # Validate file paths
    print("\n[STEP 0] Validating File Paths...")
    print("-" * 80)

    transactions_file = Path(TRANSACTIONS_PATH)
    alerts_file = Path(ALERTS_PATH)

    if not transactions_file.exists():
        print(f"❌ ERROR: Transactions file not found at:")
        print(f"   {TRANSACTIONS_PATH}")
        print("\nPlease update TRANSACTIONS_PATH in this script.")
        return

    if not alerts_file.exists():
        print(f"❌ ERROR: Alerts file not found at:")
        print(f"   {ALERTS_PATH}")
        print("\nPlease update ALERTS_PATH in this script.")
        return

    # Show file info
    tx_size_mb = transactions_file.stat().st_size / (1024 * 1024)
    alert_size_kb = alerts_file.stat().st_size / 1024

    print(f"✅ Transactions file: {TRANSACTIONS_PATH}")
    print(f"   Size: {tx_size_mb:.1f} MB")
    print(f"✅ Alerts file: {ALERTS_PATH}")
    print(f"   Size: {alert_size_kb:.1f} KB")

    # Initialize system
    print("\n[STEP 1] Initializing System...")
    print("-" * 80)
    system = AMLDriftDetectionSystem()

    # Load and process data
    print("\n[STEP 2] Loading and Processing Data...")
    print("-" * 80)
    print("⏳ This may take 2-5 minutes for large files...")

    try:
        processed_data = system.upload_and_process_datasets(
            transactions_path=TRANSACTIONS_PATH,
            alerts_path=ALERTS_PATH
        )

        print(f"\n✅ Data loaded successfully!")
        print(f"   Total transactions: {len(processed_data):,}")
        print(f"   Unique alerts: {processed_data['alert_id'].nunique():,}")

    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        print("\nPlease check:")
        print("  1. File paths are correct")
        print("  2. Files are not corrupted")
        print("  3. Files have the expected format")
        return

    # Engineer features
    print("\n[STEP 3] Engineering Features...")
    print("-" * 80)

    try:
        features_df = system.engineer_features()
        print(f"✅ Created {len([c for c in features_df.columns if c not in ['alert_id', 'is_suspicious', 'timestamp']])} features")

    except Exception as e:
        print(f"❌ ERROR engineering features: {e}")
        return

    # Split temporal windows
    print("\n[STEP 4] Creating Temporal Windows...")
    print("-" * 80)

    try:
        window1, window2, window3 = system.split_temporal_windows()
        print(f"✅ Window 1 (Training): {len(window1):,} alerts")
        print(f"✅ Window 2 (Validation): {len(window2):,} alerts")
        print(f"✅ Window 3 (Testing): {len(window3):,} alerts")

    except Exception as e:
        print(f"❌ ERROR splitting windows: {e}")
        return

    # Train model
    print("\n[STEP 5] Training Model...")
    print("-" * 80)
    print(f"⏳ Training {MODEL_TYPE.upper()} model...")
    print("   This may take 5-15 minutes for large datasets...")

    try:
        use_xgboost = (MODEL_TYPE == 'xgboost')
        model = system.train_baseline_model(use_xgboost=use_xgboost)
        print(f"✅ Model trained successfully!")

    except Exception as e:
        print(f"❌ ERROR training model: {e}")
        return

    # Evaluate on validation set
    print("\n[STEP 6] Evaluating Model...")
    print("-" * 80)

    try:
        print("\nPerformance on Window 2 (Validation):")
        metrics_w2 = system.evaluate_model(window2, "Window 2")

        print("\nPerformance on Window 3 (Test):")
        metrics_w3 = system.evaluate_model(window3, "Window 3")

        # Show key metrics
        print("\n" + "=" * 80)
        print("KEY METRICS SUMMARY")
        print("=" * 80)
        print(f"\nValidation Set (Window 2):")
        print(f"  Recall: {metrics_w2['recall']:.3f} (% of suspicious alerts caught)")
        print(f"  FPR: {metrics_w2['fpr']:.3f} (% of benign alerts flagged)")
        print(f"  Accuracy: {metrics_w2['accuracy']:.3f}")

        print(f"\nTest Set (Window 3):")
        print(f"  Recall: {metrics_w3['recall']:.3f}")
        print(f"  FPR: {metrics_w3['fpr']:.3f}")
        print(f"  Accuracy: {metrics_w3['accuracy']:.3f}")

    except Exception as e:
        print(f"❌ ERROR evaluating model: {e}")
        return

    # Detect drift
    print("\n[STEP 7] Detecting Drift...")
    print("-" * 80)

    try:
        drift_detected, drift_results = system.detect_drift(window3, "Window 3")

        if drift_detected:
            print("\n⚠️ DRIFT DETECTED")
            print("   Consider retraining the model with more recent data")

            # Show top drifted features
            sorted_features = sorted(drift_results.items(),
                                    key=lambda x: x[1]['psi'],
                                    reverse=True)
            print("\n   Top 5 drifted features:")
            for i, (feature, stats) in enumerate(sorted_features[:5], 1):
                print(f"     {i}. {feature}: PSI={stats['psi']:.4f}")
        else:
            print("\n✅ No significant drift detected")
            print("   Model is stable")

    except Exception as e:
        print(f"❌ ERROR detecting drift: {e}")

    # Classify a sample alert
    print("\n[STEP 8] Testing Classification...")
    print("-" * 80)

    try:
        # Get a suspicious alert for demo
        suspicious_alerts = window3[window3['is_suspicious'] == 1]
        if len(suspicious_alerts) > 0:
            sample = suspicious_alerts.iloc[0:1]
        else:
            sample = window3.iloc[0:1]

        print(f"\nClassifying Alert ID: {sample['alert_id'].values[0]}")
        print(f"True label: {'Suspicious' if sample['is_suspicious'].values[0] == 1 else 'Benign'}")

        prediction, probability, explanation = system.classify_alert(sample)

        print(f"\nPrediction: {'⚠️ SUSPICIOUS' if prediction == 1 else '✅ BENIGN'}")
        print(f"Confidence: {probability:.1%}")

        # Show top 3 features
        print("\nTop 3 Contributing Features:")
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        feature_values = explanation['feature_values']

        abs_shap = abs(shap_values)
        top_indices = sorted(range(len(abs_shap)), key=lambda i: abs_shap[i], reverse=True)[:3]

        for i, idx in enumerate(top_indices, 1):
            feature = feature_names[idx]
            value = feature_values[idx]
            impact = shap_values[idx]
            direction = "→ Suspicious" if impact > 0 else "→ Benign"
            print(f"  {i}. {feature} = {value:.2f} (impact: {abs(impact):.4f}) {direction}")

    except Exception as e:
        print(f"⚠️ Warning: Could not test classification: {e}")

    # Save model
    print("\n[STEP 9] Saving Model...")
    print("-" * 80)

    try:
        system.save_model(MODEL_OUTPUT)
        print(f"✅ Model saved to: {MODEL_OUTPUT}")

    except Exception as e:
        print(f"❌ ERROR saving model: {e}")
        return

    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)

    print(f"\nYour trained model is ready: {MODEL_OUTPUT}")

    print("\n📊 Model Performance Summary:")
    print(f"  • Recall: {metrics_w3['recall']:.1%} of suspicious alerts detected")
    print(f"  • FPR: {metrics_w3['fpr']:.1%} of benign alerts flagged")
    print(f"  • Accuracy: {metrics_w3['accuracy']:.1%}")

    print("\n🎯 Next Steps:")
    print("  1. Use for classification:")
    print(f"     python cli.py classify --model {MODEL_OUTPUT} --alert alert.json --explain")
    print("\n  2. Monitor for drift:")
    print(f"     python cli.py detect-drift --model {MODEL_OUTPUT} --transactions ... --alerts ...")
    print("\n  3. Deploy to production:")
    print("     See USER_GUIDE.md for deployment instructions")

    print("\n✨ Done!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
