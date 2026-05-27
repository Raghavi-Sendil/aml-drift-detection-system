"""
AML Alert Classification System with Drift Detection and SHAP Explainability
Based on: "AML Alert Classification Under Transaction Behavior Shifts for False Positive Reduction"

This system:
1. Accepts dataset uploads (transactions and alerts)
2. Automatically preprocesses and joins data
3. Engineers features per the research methodology
4. Detects concept drift using PSI, KS test, and KL divergence
5. Decides if retraining is needed
6. Classifies alerts as suspicious or not
7. Explains predictions using SHAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from drift_detector import DriftDetector
from model_trainer import ModelTrainer
from explainer import SHAPExplainer

class AMLDriftDetectionSystem:
    def __init__(self, base_path="./aml_system_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.data_processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.drift_detector = DriftDetector()
        self.model_trainer = ModelTrainer()
        self.explainer = SHAPExplainer()

        self.baseline_model = None
        self.baseline_stats = None
        self.is_trained = False

        print("=" * 80)
        print("AML Alert Classification System with Drift Detection")
        print("=" * 80)

    def upload_and_process_datasets(self, transactions_path, alerts_path=None):
        """
        Step 1: Upload and process datasets

        Args:
            transactions_path: Path to transactions CSV
            alerts_path: Path to alerts CSV (HIGHLY RECOMMENDED - contains labels)

        Note: For the SynthAML dataset, you MUST provide both:
            - transactions_path: synthetic_transactions.csv
            - alerts_path: synthetic_alerts_fixed.csv
        """
        print("\n[STEP 1] Loading and Processing Datasets...")
        print("-" * 80)

        # Load transactions data
        print(f"Loading transactions from: {transactions_path}")
        transactions_df = pd.read_csv(transactions_path)
        print(f"✓ Loaded {len(transactions_df):,} transactions")

        # Load alerts data
        if alerts_path:
            print(f"Loading alerts from: {alerts_path}")
            alerts_df = pd.read_csv(alerts_path)
            print(f"✓ Loaded {len(alerts_df):,} alerts")
        else:
            alerts_df = None
            print("⚠ Warning: No alerts file provided")
            print("  If using SynthAML dataset, you MUST provide alerts file!")
            print("  Usage: system.upload_and_process_datasets(")
            print("    transactions_path='synthetic_transactions.csv',")
            print("    alerts_path='synthetic_alerts_fixed.csv'")
            print("  )")

        # Process and join
        self.processed_data = self.data_processor.process_and_join(
            transactions_df,
            alerts_df
        )

        print(f"\n✓ Final dataset: {len(self.processed_data):,} transaction records")
        print(f"✓ Unique alerts: {self.processed_data['alert_id'].nunique():,}")

        return self.processed_data

    def engineer_features(self):
        """
        Step 2: Feature Engineering
        Following the methodology from the paper:
        - Transaction count per alert
        - Mean/Max transaction size
        - Net flow of money
        - Burst score
        - Average time between transactions
        - Credit/Debit/Wire transfer counts
        """
        print("\n[STEP 2] Engineering Features...")
        print("-" * 80)

        self.features_df = self.feature_engineer.create_features(self.processed_data)

        print(f"✓ Created {len(self.features_df.columns)} features:")
        for feature in self.features_df.columns:
            print(f"  • {feature}")

        return self.features_df

    def split_temporal_windows(self, train_ratio=0.4, eval_ratio=0.3, test_ratio=0.3):
        """
        Step 3: Split data into temporal windows
        Window 1 (40%): Training
        Window 2 (30%): Evaluation
        Window 3 (30%): Testing
        """
        print("\n[STEP 3] Creating Temporal Windows...")
        print("-" * 80)

        # Sort by timestamp
        if 'timestamp' in self.features_df.columns:
            self.features_df = self.features_df.sort_values('timestamp')

        n = len(self.features_df)
        train_end = int(n * train_ratio)
        eval_end = train_end + int(n * eval_ratio)

        self.window1 = self.features_df.iloc[:train_end].copy()
        self.window2 = self.features_df.iloc[train_end:eval_end].copy()
        self.window3 = self.features_df.iloc[eval_end:].copy()

        print(f"✓ Window 1 (Training): {len(self.window1):,} samples")
        print(f"✓ Window 2 (Evaluation): {len(self.window2):,} samples")
        print(f"✓ Window 3 (Testing): {len(self.window3):,} samples")

        return self.window1, self.window2, self.window3

    def train_baseline_model(self, use_xgboost=True):
        """
        Step 4: Train baseline model on Window 1
        """
        print("\n[STEP 4] Training Baseline Model...")
        print("-" * 80)

        # Prepare training data (drop non-feature columns)
        X_train = self.window1.drop(['is_suspicious', 'alert_id', 'timestamp'], axis=1, errors='ignore')
        y_train = self.window1['is_suspicious']

        # Train model
        self.baseline_model = self.model_trainer.train(
            X_train,
            y_train,
            use_xgboost=use_xgboost
        )

        # Store baseline statistics for drift detection
        self.baseline_stats = self.drift_detector.compute_baseline_stats(X_train)

        self.is_trained = True

        print(f"✓ Model trained successfully")
        print(f"✓ Model type: {'XGBoost' if use_xgboost else 'Logistic Regression'}")

        return self.baseline_model

    def evaluate_model(self, window_data, window_name="Window 2"):
        """
        Step 5: Evaluate model on a time window
        """
        print(f"\n[STEP 5] Evaluating Model on {window_name}...")
        print("-" * 80)

        X_test = window_data.drop(['is_suspicious', 'alert_id', 'timestamp'], axis=1, errors='ignore')
        y_test = window_data['is_suspicious']

        metrics = self.model_trainer.evaluate(X_test, y_test)

        print(f"✓ Recall (Suspicious): {metrics['recall']:.3f}")
        print(f"✓ False Positive Rate: {metrics['fpr']:.3f}")
        print(f"✓ Accuracy: {metrics['accuracy']:.3f}")
        print(f"✓ Precision: {metrics['precision']:.3f}")
        print(f"✓ F1 Score: {metrics['f1']:.3f}")

        return metrics

    def detect_drift(self, current_window, window_name="Current"):
        """
        Step 6: Detect drift using PSI, KS test, and KL divergence
        """
        print(f"\n[STEP 6] Detecting Drift in {window_name}...")
        print("-" * 80)

        X_current = current_window.drop(['is_suspicious', 'alert_id'], axis=1, errors='ignore')

        drift_results = self.drift_detector.detect_drift(
            self.baseline_stats,
            X_current
        )

        print("\nDrift Detection Results:")
        print(f"{'Feature':<30} {'PSI':<10} {'KS Stat':<10} {'KL Div':<10} {'Status':<15}")
        print("-" * 80)

        drift_detected = False
        for feature, stats in drift_results.items():
            status = "⚠ DRIFT" if stats['drift_detected'] else "✓ Stable"
            if stats['drift_detected']:
                drift_detected = True

            print(f"{feature:<30} {stats['psi']:<10.4f} {stats['ks_statistic']:<10.4f} "
                  f"{stats['kl_divergence']:<10.4f} {status:<15}")

        return drift_detected, drift_results

    def decide_retraining(self, drift_detected, metrics, drift_threshold=0.2):
        """
        Step 7: Decide if retraining is needed

        Retraining criteria:
        1. Significant drift detected (PSI > threshold)
        2. FPR increased significantly
        3. Accuracy dropped
        """
        print(f"\n[STEP 7] Retraining Decision...")
        print("-" * 80)

        needs_retraining = False
        reasons = []

        if drift_detected:
            needs_retraining = True
            reasons.append("Significant drift detected in features")

        if metrics['fpr'] > 0.55:
            needs_retraining = True
            reasons.append(f"High false positive rate: {metrics['fpr']:.3f}")

        if metrics['accuracy'] < 0.50:
            needs_retraining = True
            reasons.append(f"Low accuracy: {metrics['accuracy']:.3f}")

        if needs_retraining:
            print("⚠ RETRAINING RECOMMENDED")
            print("\nReasons:")
            for reason in reasons:
                print(f"  • {reason}")
        else:
            print("✓ Model performing well - No retraining needed")

        return needs_retraining, reasons

    def retrain_model(self, historical_data, recent_data):
        """
        Step 8: Retrain model with combined historical and recent data
        """
        print(f"\n[STEP 8] Retraining Model...")
        print("-" * 80)

        # Combine data
        combined_data = pd.concat([historical_data, recent_data], ignore_index=True)

        X_train = combined_data.drop(['is_suspicious', 'alert_id', 'timestamp'], axis=1, errors='ignore')
        y_train = combined_data['is_suspicious']

        print(f"✓ Training on {len(combined_data):,} samples")
        print(f"  - Historical: {len(historical_data):,}")
        print(f"  - Recent: {len(recent_data):,}")

        # Retrain
        self.baseline_model = self.model_trainer.train(X_train, y_train)
        self.baseline_stats = self.drift_detector.compute_baseline_stats(X_train)

        print("✓ Model retrained successfully")

        return self.baseline_model

    def classify_alert(self, alert_features):
        """
        Step 9: Classify a new alert as suspicious or not

        Args:
            alert_features: Dictionary or DataFrame with alert features

        Returns:
            prediction: 0 (benign) or 1 (suspicious)
            probability: Probability of being suspicious
            explanation: SHAP explanation
        """
        print(f"\n[STEP 9] Classifying Alert...")
        print("-" * 80)

        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")

        # Convert to DataFrame if dict
        if isinstance(alert_features, dict):
            alert_features = pd.DataFrame([alert_features])

        # Remove non-feature columns
        X = alert_features.drop(['is_suspicious', 'alert_id', 'timestamp'],
                                axis=1, errors='ignore')

        # Predict
        prediction = self.model_trainer.predict(X)[0]
        probability = self.model_trainer.predict_proba(X)[0][1]

        # Generate SHAP explanation
        explanation = self.explainer.explain_prediction(
            self.baseline_model,
            X,
            feature_names=X.columns.tolist()
        )

        # Print result
        status = "⚠ SUSPICIOUS (Report)" if prediction == 1 else "✓ BENIGN (No Report)"
        print(f"\n{status}")
        print(f"Probability of being suspicious: {probability:.2%}")

        return prediction, probability, explanation

    def explain_decision(self, explanation, top_n=5):
        """
        Step 10: Explain why the model made its decision using SHAP
        """
        print(f"\n[STEP 10] Explanation (Why this decision?)...")
        print("-" * 80)

        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        feature_values = explanation['feature_values']

        # Get top contributing features
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]

        print(f"\nTop {top_n} Contributing Features:")
        print(f"{'Feature':<30} {'Value':<15} {'Impact':<15} {'Direction':<15}")
        print("-" * 80)

        for idx in top_indices:
            feature = feature_names[idx]
            value = feature_values[idx]
            impact = abs_shap[idx]
            direction = "→ Suspicious" if shap_values[idx] > 0 else "→ Benign"

            print(f"{feature:<30} {value:<15.2f} {impact:<15.4f} {direction:<15}")

        # Detailed explanation
        print("\n" + "=" * 80)
        print("DETAILED EXPLANATION:")
        print("=" * 80)

        for idx in top_indices:
            feature = feature_names[idx]
            value = feature_values[idx]
            shap_val = shap_values[idx]

            if shap_val > 0:
                print(f"\n• {feature} = {value:.2f}")
                print(f"  This feature increases suspicion by {abs(shap_val):.4f}")
                print(f"  Higher values indicate unusual transaction behavior")
            else:
                print(f"\n• {feature} = {value:.2f}")
                print(f"  This feature decreases suspicion by {abs(shap_val):.4f}")
                print(f"  This value is within normal transaction patterns")

        return explanation

    def save_model(self, path="aml_model.pkl"):
        """Save trained model and statistics"""
        save_path = self.base_path / path

        model_data = {
            'model': self.baseline_model,
            'baseline_stats': self.baseline_stats,
            'feature_names': self.features_df.columns.tolist(),
            'trained_date': datetime.now().isoformat()
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Model saved to {save_path}")

    def load_model(self, path="aml_model.pkl"):
        """Load trained model and statistics"""
        load_path = self.base_path / path

        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        self.baseline_model = model_data['model']
        self.baseline_stats = model_data['baseline_stats']
        self.is_trained = True

        print(f"\n✓ Model loaded from {load_path}")
        print(f"✓ Trained on: {model_data['trained_date']}")


def main():
    """
    Example usage of the AML Drift Detection System
    """
    print("\n" + "=" * 80)
    print("AML ALERT CLASSIFICATION SYSTEM - DEMO")
    print("=" * 80)

    # Initialize system
    system = AMLDriftDetectionSystem()

    print("\n")
    print("INSTRUCTIONS:")
    print("-" * 80)
    print("1. Prepare your datasets:")
    print("   - transactions.csv: Financial transaction records")
    print("   - alerts.csv: AML alerts (optional)")
    print("\n2. Required columns in transactions.csv:")
    print("   - alert_id: Alert identifier")
    print("   - timestamp: Transaction timestamp")
    print("   - amount: Transaction amount")
    print("   - transaction_type: Type (credit/debit/wire)")
    print("   - is_suspicious: Label (1=suspicious, 0=benign)")
    print("\n3. The system will automatically:")
    print("   ✓ Preprocess and join tables")
    print("   ✓ Engineer features")
    print("   ✓ Train models")
    print("   ✓ Detect drift")
    print("   ✓ Decide on retraining")
    print("   ✓ Classify new alerts")
    print("   ✓ Explain decisions with SHAP")
    print("=" * 80)

    print("\n\nWaiting for dataset upload...")
    print("Please provide the path to your transactions CSV file.")


if __name__ == "__main__":
    main()
