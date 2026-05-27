"""
Model Trainer Module
Trains and evaluates AML classification models:
- Logistic Regression (interpretable baseline)
- XGBoost (high-performance ensemble)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not installed. Using Logistic Regression only.")


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = None

    def train(self, X_train, y_train, use_xgboost=True):
        """
        Train AML classification model

        Args:
            X_train: Training features
            y_train: Training labels
            use_xgboost: Whether to use XGBoost (if False, uses Logistic Regression)

        Returns:
            Trained model
        """
        print(f"\n  Training {'XGBoost' if use_xgboost and XGBOOST_AVAILABLE else 'Logistic Regression'} model...")

        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        total = len(y_train)

        # Weight inversely proportional to class frequency
        weight_for_0 = total / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0
        weight_for_1 = total / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0

        sample_weights = np.where(y_train == 1, weight_for_1, weight_for_0)

        print(f"  Class distribution: Benign={class_counts[0]}, Suspicious={class_counts[1]}")
        print(f"  Sample weights: Benign={weight_for_0:.2f}, Suspicious={weight_for_1:.2f}")

        # Convert to standard numpy dtypes (avoid pandas extension dtypes)
        import pandas as pd
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.astype(float)
        elif hasattr(X_train, 'values'):
            X_train = X_train.values.astype(float)
        else:
            X_train = np.array(X_train, dtype=float)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        if use_xgboost and XGBOOST_AVAILABLE:
            self.model_type = 'xgboost'
            self._train_xgboost(X_train_scaled, y_train, sample_weights)
        else:
            self.model_type = 'logistic'
            self._train_logistic(X_train_scaled, y_train, sample_weights)

        print(f"  ✓ Model trained successfully")

        return self.model

    def _train_logistic(self, X_train, y_train, sample_weights):
        """Train Logistic Regression model"""
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',  # Handle imbalance
            solver='lbfgs',
            n_jobs=-1
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def _train_xgboost(self, X_train, y_train, sample_weights):
        """Train XGBoost model"""
        # Calculate scale_pos_weight for XGBoost
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def predict(self, X_test):
        """
        Make predictions

        Args:
            X_test: Test features

        Returns:
            Predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Convert to standard numpy dtypes
        import pandas as pd
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.astype(float)
        elif hasattr(X_test, 'values'):
            X_test = X_test.values.astype(float)
        else:
            X_test = np.array(X_test, dtype=float)

        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def predict_proba(self, X_test):
        """
        Get prediction probabilities

        Args:
            X_test: Test features

        Returns:
            Probability array [prob_class_0, prob_class_1]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Convert to standard numpy dtypes
        import pandas as pd
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.astype(float)
        elif hasattr(X_test, 'values'):
            X_test = X_test.values.astype(float)
        else:
            X_test = np.array(X_test, dtype=float)

        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with performance metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # False Positive Rate (critical metric for AML)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # False Negative Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'fpr': fpr,
            'fnr': fnr,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        return metrics

    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if self.model_type == 'xgboost':
            importances = self.model.feature_importances_
        elif self.model_type == 'logistic':
            # Use absolute coefficient values
            importances = np.abs(self.model.coef_[0])
        else:
            return {}

        # Sort by importance
        importance_dict = dict(zip(feature_names, importances))
        importance_dict = dict(sorted(importance_dict.items(),
                                      key=lambda x: x[1], reverse=True))

        return importance_dict

    def print_evaluation(self, metrics):
        """
        Print evaluation metrics in a formatted way

        Args:
            metrics: Dictionary from evaluate()
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  AUC-ROC:   {metrics['auc']:.3f}")

        print(f"\nCritical AML Metrics:")
        print(f"  False Positive Rate: {metrics['fpr']:.3f}")
        print(f"  False Negative Rate: {metrics['fnr']:.3f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['true_negatives']:,}")
        print(f"  False Positives: {metrics['false_positives']:,}")
        print(f"  False Negatives: {metrics['false_negatives']:,}")
        print(f"  True Positives:  {metrics['true_positives']:,}")

        # Interpretation
        print(f"\nInterpretation:")
        if metrics['fpr'] > 0.5:
            print(f"  ⚠ High false positive rate - many benign alerts flagged as suspicious")
        else:
            print(f"  ✓ False positive rate acceptable")

        if metrics['fnr'] > 0.3:
            print(f"  ⚠ High false negative rate - missing suspicious alerts")
        else:
            print(f"  ✓ False negative rate acceptable")

        if metrics['recall'] > 0.7:
            print(f"  ✓ Good recall - detecting most suspicious activities")
        else:
            print(f"  ⚠ Low recall - missing many suspicious activities")

        print("=" * 80)


class EnsembleTrainer:
    """
    Train ensemble of models for improved robustness
    """

    def __init__(self):
        self.models = []
        self.weights = []

    def train_ensemble(self, X_train, y_train):
        """
        Train multiple models and combine predictions

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            List of trained models
        """
        print("\n  Training ensemble of models...")

        # Train Logistic Regression
        lr_trainer = ModelTrainer()
        lr_model = lr_trainer.train(X_train, y_train, use_xgboost=False)
        self.models.append(lr_trainer)

        # Train XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_trainer = ModelTrainer()
            xgb_model = xgb_trainer.train(X_train, y_train, use_xgboost=True)
            self.models.append(xgb_trainer)

        print(f"  ✓ Trained {len(self.models)} models in ensemble")

        return self.models

    def predict_ensemble(self, X_test):
        """
        Make predictions using ensemble (majority vote)

        Args:
            X_test: Test features

        Returns:
            Ensemble predictions
        """
        predictions = []

        for model_trainer in self.models:
            pred = model_trainer.predict(X_test)
            predictions.append(pred)

        # Majority vote
        predictions = np.array(predictions)
        ensemble_pred = np.round(predictions.mean(axis=0)).astype(int)

        return ensemble_pred

    def predict_proba_ensemble(self, X_test):
        """
        Get ensemble prediction probabilities (average)

        Args:
            X_test: Test features

        Returns:
            Average probability array
        """
        probabilities = []

        for model_trainer in self.models:
            proba = model_trainer.predict_proba(X_test)
            probabilities.append(proba)

        # Average probabilities
        avg_proba = np.mean(probabilities, axis=0)

        return avg_proba
