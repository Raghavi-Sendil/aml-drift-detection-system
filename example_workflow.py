"""
Complete Example Workflow
Demonstrates end-to-end AML alert classification with drift detection
"""

import pandas as pd
import numpy as np
from main import AMLDriftDetectionSystem


def generate_synthetic_data(n_alerts=1000):
    """
    Generate synthetic AML data for demonstration

    In production, replace this with your actual dataset upload
    """
    print("\nGenerating synthetic AML data for demonstration...")

    np.random.seed(42)

    # Generate alerts
    alert_ids = range(1, n_alerts + 1)

    # Generate transactions for each alert
    transactions = []

    for alert_id in alert_ids:
        # Random number of transactions per alert
        n_transactions = np.random.randint(1, 20)

        # Randomly determine if alert is suspicious (10% suspicious)
        is_suspicious = 1 if np.random.random() < 0.1 else 0

        for tx_num in range(n_transactions):
            # Base timestamp
            base_time = pd.Timestamp('2024-01-01') + pd.Timedelta(days=alert_id % 365)

            # Add transaction-specific offset
            tx_time = base_time + pd.Timedelta(hours=np.random.randint(0, 48))

            # Generate transaction amount
            if is_suspicious:
                # Suspicious: larger amounts, more wires, round amounts
                amount = np.random.choice([
                    np.random.uniform(5000, 50000),  # Large amounts
                    np.random.uniform(9000, 9999),   # Just below reporting threshold
                    10000, 50000, 100000             # Round amounts
                ])
                tx_type = np.random.choice(['wire', 'debit', 'credit'],
                                           p=[0.5, 0.3, 0.2])  # More wires
            else:
                # Benign: normal amounts, varied types
                amount = np.random.uniform(10, 5000)
                tx_type = np.random.choice(['debit', 'credit', 'wire'],
                                           p=[0.5, 0.4, 0.1])

            transactions.append({
                'alert_id': alert_id,
                'timestamp': tx_time,
                'amount': amount,
                'transaction_type': tx_type,
                'is_suspicious': is_suspicious
            })

    df = pd.DataFrame(transactions)

    print(f"✓ Generated {len(df):,} transactions across {n_alerts:,} alerts")
    print(f"✓ Suspicious alerts: {df.groupby('alert_id')['is_suspicious'].first().sum()}")
    print(f"✓ Benign alerts: {n_alerts - df.groupby('alert_id')['is_suspicious'].first().sum()}")

    return df


def run_full_workflow():
    """
    Run complete AML drift detection workflow
    """
    print("\n" + "=" * 80)
    print("AML DRIFT DETECTION SYSTEM - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 80)

    # Initialize system
    system = AMLDriftDetectionSystem()

    # Generate or load data
    # In production: transactions_df = pd.read_csv('your_transactions.csv')
    transactions_df = generate_synthetic_data(n_alerts=1000)

    # Save to CSV for reference
    transactions_df.to_csv('/Users/sycamore/aml_drift_detection_system/sample_data.csv',
                           index=False)
    print(f"\n✓ Sample data saved to sample_data.csv")

    # Step 1: Upload and process datasets
    processed_data = system.upload_and_process_datasets(
        transactions_path='/Users/sycamore/aml_drift_detection_system/sample_data.csv'
    )

    # Step 2: Engineer features
    features_df = system.engineer_features()

    # Step 3: Split into temporal windows
    window1, window2, window3 = system.split_temporal_windows()

    # Step 4: Train baseline model
    baseline_model = system.train_baseline_model(use_xgboost=True)

    # Step 5: Evaluate on Window 2
    print("\n" + "=" * 80)
    print("EVALUATING ON WINDOW 2 (INTERMEDIATE DATA)")
    print("=" * 80)
    metrics_w2 = system.evaluate_model(window2, "Window 2")

    # Step 6: Detect drift in Window 3
    drift_detected, drift_results = system.detect_drift(window3, "Window 3")

    # Step 7: Evaluate on Window 3 (with potential drift)
    print("\n" + "=" * 80)
    print("EVALUATING ON WINDOW 3 (RECENT DATA)")
    print("=" * 80)
    metrics_w3 = system.evaluate_model(window3, "Window 3")

    # Step 8: Decide if retraining needed
    needs_retraining, reasons = system.decide_retraining(drift_detected, metrics_w3)

    # Step 9: Retrain if needed
    if needs_retraining:
        print("\n" + "=" * 80)
        print("RETRAINING MODEL WITH UPDATED DATA")
        print("=" * 80)
        system.retrain_model(window1, window2)

        # Evaluate retrained model
        print("\n" + "=" * 80)
        print("EVALUATING RETRAINED MODEL ON WINDOW 3")
        print("=" * 80)
        metrics_retrained = system.evaluate_model(window3, "Window 3 (Retrained)")

    # Step 10: Simulate drift and test robustness
    print("\n" + "=" * 80)
    print("SIMULATING BEHAVIORAL DRIFT")
    print("=" * 80)

    # Apply drift to Window 3
    window3_drift = system.feature_engineer.simulate_drift(
        window3,
        tx_size_increase=0.3,
        tx_freq_increase=0.2,
        wire_increase=0.5
    )

    # Detect drift in simulated data
    drift_detected_sim, drift_results_sim = system.detect_drift(window3_drift,
                                                                 "Window 3 (Simulated Drift)")

    # Evaluate on drifted data
    print("\n" + "=" * 80)
    print("EVALUATING ON DRIFTED DATA")
    print("=" * 80)
    metrics_drift = system.evaluate_model(window3_drift, "Window 3 (Drift)")

    # Step 11: Classify a single alert and explain
    print("\n" + "=" * 80)
    print("EXAMPLE: CLASSIFYING AND EXPLAINING A SINGLE ALERT")
    print("=" * 80)

    # Take a sample alert from Window 3
    sample_alert = window3.iloc[10:11].copy()

    # Classify
    prediction, probability, explanation = system.classify_alert(sample_alert)

    # Explain
    system.explain_decision(explanation, top_n=5)

    # Generate full report
    text_explanation = system.explainer.create_text_explanation(
        explanation, prediction, probability
    )
    print("\n" + text_explanation)

    # Generate investigator recommendation
    recommendation = system.explainer.generate_report_recommendation(
        explanation, prediction, probability
    )
    print(recommendation)

    # Step 12: Save model
    system.save_model("aml_trained_model.pkl")

    # Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nPerformance Across Windows:")
    print(f"  Window 2: Recall={metrics_w2['recall']:.3f}, FPR={metrics_w2['fpr']:.3f}")
    print(f"  Window 3: Recall={metrics_w3['recall']:.3f}, FPR={metrics_w3['fpr']:.3f}")
    print(f"  W3 Drift: Recall={metrics_drift['recall']:.3f}, FPR={metrics_drift['fpr']:.3f}")

    print(f"\nDrift Detection:")
    print(f"  Drift detected: {drift_detected or drift_detected_sim}")
    print(f"  Retraining needed: {needs_retraining}")

    print("\n✓ System ready for production deployment")
    print("=" * 80)


def quick_classification_example():
    """
    Quick example: Load model and classify a new alert
    """
    print("\n" + "=" * 80)
    print("QUICK EXAMPLE: CLASSIFY NEW ALERT")
    print("=" * 80)

    # Initialize system
    system = AMLDriftDetectionSystem()

    # Load pre-trained model
    try:
        system.load_model("aml_trained_model.pkl")
    except:
        print("\n⚠ No trained model found. Run full workflow first.")
        return

    # Create a sample alert (in production, this comes from your monitoring system)
    new_alert = {
        'transaction_count': 15,
        'mean_transaction_size': 8500.00,
        'max_transaction_size': 15000.00,
        'min_transaction_size': 500.00,
        'std_transaction_size': 3200.00,
        'net_flow': 127500.00,
        'burst_score': 0.8,
        'avg_time_between_tx': 2.5,
        'credit_count': 5,
        'debit_count': 10,
        'wire_count': 8,
        'credit_debit_ratio': 0.5,
        'wire_ratio': 0.53,
        'transaction_velocity': 6.0,
        'amount_variance': 0.376,
        'weekend_transaction_ratio': 0.4,
        'night_transaction_ratio': 0.3,
        'round_amount_ratio': 0.6,
        'amount_concentration': 0.45
    }

    # Classify
    prediction, probability, explanation = system.classify_alert(new_alert)

    # Explain
    system.explain_decision(explanation, top_n=5)

    # Full explanation
    text_explanation = system.explainer.create_text_explanation(
        explanation, prediction, probability
    )
    print("\n" + text_explanation)

    # Recommendation
    recommendation = system.explainer.generate_report_recommendation(
        explanation, prediction, probability
    )
    print(recommendation)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_classification_example()
    else:
        run_full_workflow()

    print("\n" + "=" * 80)
    print("To run quick classification: python example_workflow.py quick")
    print("=" * 80)
