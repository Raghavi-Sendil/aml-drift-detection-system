"""
Test script for SynthAML dataset
Tests the system with synthetic_transactions.csv and synthetic_alerts_fixed.csv
"""

from main import AMLDriftDetectionSystem
import sys

def main():
    print("\n" + "=" * 80)
    print("TESTING AML SYSTEM WITH SYNTHAML DATASET")
    print("=" * 80)

    # File paths
    transactions_path = "/Users/sycamore/Downloads/synthetic_transactions.csv"
    alerts_path = "/Users/sycamore/Downloads/synthetic_alerts_fixed.csv"

    print(f"\nTransactions file: {transactions_path}")
    print(f"Alerts file: {alerts_path}")

    try:
        # Initialize system
        system = AMLDriftDetectionSystem()

        # Step 1: Upload and process both datasets
        print("\n" + "-" * 80)
        print("STEP 1: Loading and Processing Data")
        print("-" * 80)
        processed_data = system.upload_and_process_datasets(
            transactions_path=transactions_path,
            alerts_path=alerts_path
        )

        print(f"\n✅ Successfully loaded and processed data!")
        print(f"   Total transactions: {len(processed_data):,}")
        print(f"   Unique alerts: {processed_data['alert_id'].nunique():,}")

        # Show class distribution
        alert_labels = processed_data.groupby('alert_id')['is_suspicious'].first()
        print(f"\nClass Distribution:")
        print(f"   Benign alerts (0): {(alert_labels == 0).sum():,}")
        print(f"   Suspicious alerts (1): {(alert_labels == 1).sum():,}")

        # Show sample data
        print(f"\nSample of processed data:")
        print(processed_data[['alert_id', 'timestamp', 'amount', 'transaction_type', 'is_suspicious']].head(10))

        # Step 2: Engineer features
        print("\n" + "-" * 80)
        print("STEP 2: Engineering Features")
        print("-" * 80)
        features_df = system.engineer_features()

        print(f"\n✅ Successfully engineered features!")
        print(f"   Total features: {len([c for c in features_df.columns if c not in ['alert_id', 'is_suspicious', 'timestamp']])}")
        print(f"\nFeatures created:")
        for col in features_df.columns:
            if col not in ['alert_id', 'is_suspicious', 'timestamp']:
                print(f"   • {col}")

        # Step 3: Split temporal windows
        print("\n" + "-" * 80)
        print("STEP 3: Splitting Temporal Windows")
        print("-" * 80)
        window1, window2, window3 = system.split_temporal_windows()

        print(f"\n✅ Successfully split windows!")
        print(f"   Window 1 (Training): {len(window1):,} alerts")
        print(f"   Window 2 (Validation): {len(window2):,} alerts")
        print(f"   Window 3 (Testing): {len(window3):,} alerts")

        # Step 4: Train model
        print("\n" + "-" * 80)
        print("STEP 4: Training Model")
        print("-" * 80)
        print("This may take a few minutes with a large dataset...")

        model = system.train_baseline_model(use_xgboost=True)

        print(f"\n✅ Model trained successfully!")

        # Step 5: Evaluate
        print("\n" + "-" * 80)
        print("STEP 5: Evaluating Model")
        print("-" * 80)

        print("\nPerformance on Window 2 (Validation):")
        metrics_w2 = system.evaluate_model(window2, "Window 2")

        print("\nPerformance on Window 3 (Test):")
        metrics_w3 = system.evaluate_model(window3, "Window 3")

        # Step 6: Detect drift
        print("\n" + "-" * 80)
        print("STEP 6: Detecting Drift")
        print("-" * 80)

        drift_detected, drift_results = system.detect_drift(window3, "Window 3")

        if drift_detected:
            print("\n⚠️ DRIFT DETECTED")
        else:
            print("\n✅ No significant drift detected")

        # Step 7: Classify a sample alert
        print("\n" + "-" * 80)
        print("STEP 7: Classifying Sample Alert")
        print("-" * 80)

        # Get a suspicious alert for demo
        suspicious_alert = window3[window3['is_suspicious'] == 1].iloc[0:1] if (window3['is_suspicious'] == 1).any() else window3.iloc[0:1]

        print(f"\nClassifying Alert ID: {suspicious_alert['alert_id'].values[0]}")
        print(f"True label: {'Suspicious' if suspicious_alert['is_suspicious'].values[0] == 1 else 'Benign'}")

        prediction, probability, explanation = system.classify_alert(suspicious_alert)

        print(f"\nPrediction: {'⚠️ SUSPICIOUS' if prediction == 1 else '✅ BENIGN'}")
        print(f"Confidence: {probability:.1%}")

        # Show top features
        system.explain_decision(explanation, top_n=5)

        # Save model
        print("\n" + "-" * 80)
        print("STEP 8: Saving Model")
        print("-" * 80)
        system.save_model("synthaml_model.pkl")

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe system is working correctly with your SynthAML dataset!")
        print("\nNext steps:")
        print("  1. Use the web interface: streamlit run web_interface.py")
        print("  2. Train on full dataset: python cli.py train --transactions ... --alerts ...")
        print("  3. Deploy to production (see USER_GUIDE.md)")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("  1. File paths are correct")
        print("  2. CSV files are not corrupted")
        print("  3. Files contain the expected columns")
        sys.exit(1)


if __name__ == "__main__":
    main()
