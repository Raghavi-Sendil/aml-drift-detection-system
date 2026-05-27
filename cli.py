#!/usr/bin/env python3
"""
Command-Line Interface for AML Drift Detection System

Usage examples:
  python cli.py train --data transactions.csv --output model.pkl
  python cli.py classify --model model.pkl --alert alert.json
  python cli.py detect-drift --model model.pkl --data new_transactions.csv
"""

import argparse
import json
import sys
import pandas as pd
from pathlib import Path

from main import AMLDriftDetectionSystem


def train_command(args):
    """Train a new model"""
    print(f"\n🎯 Training model")
    print(f"  Transactions: {args.transactions}")
    if args.alerts:
        print(f"  Alerts: {args.alerts}")

    system = AMLDriftDetectionSystem()

    # Load and process data
    print("[1/5] Loading and processing data...")
    processed_data = system.upload_and_process_datasets(
        transactions_path=args.transactions,
        alerts_path=args.alerts
    )

    # Engineer features
    print("[2/5] Engineering features...")
    features_df = system.engineer_features()

    # Split windows
    print("[3/5] Creating temporal windows...")
    window1, window2, window3 = system.split_temporal_windows()

    # Train model
    print("[4/5] Training model...")
    use_xgboost = args.model_type == 'xgboost'
    model = system.train_baseline_model(use_xgboost=use_xgboost)

    # Evaluate
    print("[5/5] Evaluating model...")
    metrics = system.evaluate_model(window2, "Validation")

    print("\n✅ Training complete!")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   FPR: {metrics['fpr']:.3f}")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")

    # Save model
    output_path = args.output or "aml_model.pkl"
    system.save_model(output_path)
    print(f"\n💾 Model saved to: {output_path}")


def classify_command(args):
    """Classify a single alert"""
    print(f"\n🔮 Classifying alert from {args.alert}")

    # Load model
    system = AMLDriftDetectionSystem()
    system.load_model(args.model)

    # Load alert data
    if args.alert.endswith('.json'):
        with open(args.alert, 'r') as f:
            alert_data = json.load(f)
    elif args.alert.endswith('.csv'):
        alert_data = pd.read_csv(args.alert).iloc[0].to_dict()
    else:
        print("⚠️ Alert file must be .json or .csv")
        return

    # Classify
    prediction, probability, explanation = system.classify_alert(alert_data)

    # Print result
    print("\n" + "=" * 80)
    if prediction == 1:
        print("⚠️ SUSPICIOUS ALERT")
        print("Recommendation: Report to investigators")
    else:
        print("✅ BENIGN ALERT")
        print("Recommendation: No action needed")

    print(f"\nConfidence: {probability:.1%}")
    print("=" * 80)

    # Show explanation
    if args.explain:
        system.explain_decision(explanation, top_n=5)

        # Generate full report
        text_explanation = system.explainer.create_text_explanation(
            explanation, prediction, probability
        )
        print("\n" + text_explanation)

        recommendation = system.explainer.generate_report_recommendation(
            explanation, prediction, probability
        )
        print(recommendation)

    # Save result if output specified
    if args.output:
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'classification': 'suspicious' if prediction == 1 else 'benign',
            'shap_values': explanation['shap_values'].tolist(),
            'feature_names': explanation['feature_names'],
            'feature_values': explanation['feature_values'].tolist()
        }

        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n💾 Results saved to: {args.output}")


def detect_drift_command(args):
    """Detect drift in new data"""
    print(f"\n📊 Detecting drift")
    print(f"  Transactions: {args.transactions}")
    if args.alerts:
        print(f"  Alerts: {args.alerts}")

    # Load model with baseline stats
    system = AMLDriftDetectionSystem()
    system.load_model(args.model)

    # Load new data
    print("[1/3] Loading new data...")
    processed_data = system.upload_and_process_datasets(
        transactions_path=args.transactions,
        alerts_path=args.alerts
    )

    # Engineer features
    print("[2/3] Engineering features...")
    features_df = system.engineer_features()

    # Detect drift
    print("[3/3] Detecting drift...")
    X_current = features_df.drop(['is_suspicious', 'alert_id', 'timestamp'],
                                  axis=1, errors='ignore')

    drift_detected, drift_results = system.detect_drift(features_df, "Current Data")

    # Print results
    print("\n" + "=" * 80)
    if drift_detected:
        print("⚠️ DRIFT DETECTED")
        print("Model retraining recommended")
    else:
        print("✅ NO SIGNIFICANT DRIFT")
        print("Model is still valid")
    print("=" * 80)

    # Show top drifted features
    print("\nTop Drifted Features:")
    sorted_features = sorted(drift_results.items(),
                            key=lambda x: x[1]['psi'],
                            reverse=True)

    for i, (feature, stats) in enumerate(sorted_features[:10], 1):
        status = "⚠️" if stats['drift_detected'] else "✓"
        print(f"  {i}. {status} {feature}: PSI={stats['psi']:.4f}")

    # Save report if output specified
    if args.output:
        report = {
            'drift_detected': drift_detected,
            'features': {
                feature: {
                    'psi': float(stats['psi']),
                    'ks_statistic': float(stats['ks_statistic']),
                    'kl_divergence': float(stats['kl_divergence']),
                    'drift_detected': bool(stats['drift_detected'])
                }
                for feature, stats in drift_results.items()
            }
        }

        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Report saved to: {args.output}")


def batch_classify_command(args):
    """Classify multiple alerts from CSV"""
    print(f"\n📦 Batch classifying alerts")
    print(f"  Transactions: {args.transactions}")
    if args.alerts:
        print(f"  Alerts: {args.alerts}")

    # Load model
    system = AMLDriftDetectionSystem()
    system.load_model(args.model)

    # Load data
    print("[1/3] Loading data...")
    system.upload_and_process_datasets(
        transactions_path=args.transactions,
        alerts_path=args.alerts
    )

    # Process and engineer features
    print("[2/3] Processing features...")
    features_df = system.engineer_features()

    # Classify each alert
    print("[3/3] Classifying alerts...")
    results = []

    for idx, row in features_df.iterrows():
        alert_features = row.drop(['is_suspicious', 'alert_id', 'timestamp'],
                                  errors='ignore')

        prediction, probability, explanation = system.classify_alert(alert_features)

        results.append({
            'alert_id': row['alert_id'],
            'prediction': prediction,
            'probability': probability,
            'classification': 'suspicious' if prediction == 1 else 'benign'
        })

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(features_df)} alerts...")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print("\n✅ Classification complete!")
    print(f"   Total alerts: {len(results_df)}")
    print(f"   Suspicious: {(results_df['prediction'] == 1).sum()}")
    print(f"   Benign: {(results_df['prediction'] == 0).sum()}")

    # Save results
    output_path = args.output or "classification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AML Drift Detection System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python cli.py train --transactions transactions.csv --alerts alerts.csv --model-type xgboost --output model.pkl

  # Classify a single alert
  python cli.py classify --model model.pkl --alert alert.json --explain

  # Detect drift
  python cli.py detect-drift --model model.pkl --transactions new_transactions.csv --alerts new_alerts.csv --output drift_report.json

  # Batch classify
  python cli.py batch-classify --model model.pkl --transactions transactions.csv --alerts alerts.csv --output results.csv

  # For SynthAML dataset
  python cli.py train --transactions synthetic_transactions.csv --alerts synthetic_alerts_fixed.csv --output model.pkl
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--transactions', required=True, help='Transactions CSV file')
    train_parser.add_argument('--alerts', help='Alerts CSV file (recommended, contains labels)')
    train_parser.add_argument('--model-type', choices=['xgboost', 'logistic'],
                             default='xgboost', help='Model type')
    train_parser.add_argument('--output', help='Output model file (default: aml_model.pkl)')

    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify a single alert')
    classify_parser.add_argument('--model', required=True, help='Trained model file')
    classify_parser.add_argument('--alert', required=True, help='Alert data (JSON or CSV)')
    classify_parser.add_argument('--explain', action='store_true',
                                help='Show detailed explanation')
    classify_parser.add_argument('--output', help='Save results to JSON file')

    # Detect drift command
    drift_parser = subparsers.add_parser('detect-drift', help='Detect drift in new data')
    drift_parser.add_argument('--model', required=True, help='Trained model file')
    drift_parser.add_argument('--transactions', required=True, help='New transactions CSV')
    drift_parser.add_argument('--alerts', help='New alerts CSV (if available)')
    drift_parser.add_argument('--output', help='Save drift report to JSON file')

    # Batch classify command
    batch_parser = subparsers.add_parser('batch-classify', help='Classify multiple alerts')
    batch_parser.add_argument('--model', required=True, help='Trained model file')
    batch_parser.add_argument('--transactions', required=True, help='Transactions CSV file')
    batch_parser.add_argument('--alerts', help='Alerts CSV file (if available)')
    batch_parser.add_argument('--output', help='Output CSV file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'classify':
            classify_command(args)
        elif args.command == 'detect-drift':
            detect_drift_command(args)
        elif args.command == 'batch-classify':
            batch_classify_command(args)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
