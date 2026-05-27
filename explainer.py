"""
SHAP Explainer Module
Provides interpretable explanations for AML alert classifications using SHAP
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    def __init__(self):
        self.explainer = None
        self.background_data = None

    def explain_prediction(self, model, X, feature_names=None, background_samples=100):
        """
        Generate SHAP explanation for a prediction

        Args:
            model: Trained model (ModelTrainer instance or sklearn/xgboost model)
            X: Features to explain (DataFrame or array)
            feature_names: List of feature names
            background_samples: Number of samples for background data

        Returns:
            Dictionary with SHAP values and explanation
        """
        # Extract actual model if ModelTrainer instance
        if hasattr(model, 'model'):
            actual_model = model.model
            scaler = model.scaler
            X_scaled = scaler.transform(X)
        else:
            actual_model = model
            X_scaled = X

        # Convert to DataFrame if needed
        if not isinstance(X_scaled, pd.DataFrame):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_scaled.shape[1])]
            X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        else:
            feature_names = X_scaled.columns.tolist()

        if not SHAP_AVAILABLE:
            # Fallback: Use simple feature importance
            return self._fallback_explanation(actual_model, X_scaled, feature_names)

        # Create SHAP explainer
        try:
            # Try TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(actual_model)
        except:
            try:
                # Try LinearExplainer for linear models
                self.explainer = shap.LinearExplainer(actual_model, X_scaled[:background_samples])
            except:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(
                    actual_model.predict_proba if hasattr(actual_model, 'predict_proba') else actual_model.predict,
                    shap.sample(X_scaled, background_samples)
                )

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_scaled)

        # Handle multi-output (for classifiers, take positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        # For single prediction
        if len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
            shap_values = shap_values[0]

        explanation = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'feature_values': X_scaled.iloc[0].values if isinstance(X_scaled, pd.DataFrame) else X_scaled[0],
            'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        }

        return explanation

    def _fallback_explanation(self, model, X, feature_names):
        """
        Fallback explanation when SHAP is not available
        Uses feature importance or coefficients
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear model
            importances = np.abs(model.coef_[0])
        else:
            # No importance available
            importances = np.ones(len(feature_names))

        # Normalize importances
        importances = importances / np.sum(importances)

        # Multiply by feature values to get pseudo-SHAP values
        feature_values = X.iloc[0].values if isinstance(X, pd.DataFrame) else X[0]
        pseudo_shap = importances * feature_values

        explanation = {
            'shap_values': pseudo_shap,
            'feature_names': feature_names,
            'feature_values': feature_values,
            'base_value': 0,
            'note': 'Fallback explanation (SHAP not available)'
        }

        return explanation

    def create_text_explanation(self, explanation, prediction, probability):
        """
        Create human-readable text explanation

        Args:
            explanation: Dictionary from explain_prediction()
            prediction: Model prediction (0 or 1)
            probability: Prediction probability

        Returns:
            String with detailed explanation
        """
        shap_values = explanation['shap_values']
        feature_names = explanation['feature_names']
        feature_values = explanation['feature_values']

        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        sorted_indices = np.argsort(abs_shap)[::-1]

        # Build explanation text
        lines = []
        lines.append("=" * 80)
        lines.append("DECISION EXPLANATION")
        lines.append("=" * 80)
        lines.append("")

        # Overall decision
        decision = "SUSPICIOUS (Report to investigators)" if prediction == 1 else "BENIGN (No action needed)"
        lines.append(f"Decision: {decision}")
        lines.append(f"Confidence: {probability:.1%}")
        lines.append("")

        # Top factors
        lines.append("Top Factors Influencing Decision:")
        lines.append("-" * 80)

        for i, idx in enumerate(sorted_indices[:5], 1):
            feature = feature_names[idx]
            value = feature_values[idx]
            impact = shap_values[idx]
            abs_impact = abs_shap[idx]

            direction = "increases" if impact > 0 else "decreases"
            suspicion = "suspicion" if impact > 0 else "confidence in benign classification"

            lines.append(f"\n{i}. {self._format_feature_name(feature)}")
            lines.append(f"   Value: {value:.2f}")
            lines.append(f"   Impact: {abs_impact:.4f} (weight)")
            lines.append(f"   Effect: This {direction} {suspicion}")
            lines.append(f"   Reason: {self._get_feature_interpretation(feature, value, impact)}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_feature_name(self, feature):
        """
        Format feature name for human readability
        """
        # Convert snake_case to Title Case
        words = feature.split('_')
        return ' '.join(word.capitalize() for word in words)

    def _get_feature_interpretation(self, feature, value, impact):
        """
        Get human-readable interpretation of feature impact

        Args:
            feature: Feature name
            value: Feature value
            impact: SHAP value (positive = suspicious, negative = benign)

        Returns:
            Interpretation string
        """
        interpretations = {
            'transaction_count': {
                'high_suspicious': f"{int(value)} transactions is unusually high, indicating possible structuring",
                'low_suspicious': f"{int(value)} transactions is very low, which can indicate test transactions",
                'high_benign': f"{int(value)} transactions is within normal business activity range",
                'low_benign': f"{int(value)} transactions is typical for regular account activity"
            },
            'mean_transaction_size': {
                'high_suspicious': f"Average transaction of ${value:,.2f} is unusually large",
                'low_suspicious': f"Average transaction of ${value:,.2f} may indicate smurfing (breaking into small amounts)",
                'high_benign': f"Average transaction of ${value:,.2f} is consistent with legitimate activity",
                'low_benign': f"Average transaction of ${value:,.2f} is typical for routine payments"
            },
            'max_transaction_size': {
                'high_suspicious': f"Maximum transaction of ${value:,.2f} exceeds typical patterns",
                'low_suspicious': f"Maximum transaction of ${value:,.2f} is consistent with alert threshold",
                'high_benign': f"Maximum transaction of ${value:,.2f} is within expected range",
                'low_benign': f"Maximum transaction of ${value:,.2f} is normal"
            },
            'net_flow': {
                'high_suspicious': f"Net flow of ${value:,.2f} suggests rapid fund movement (red flag)",
                'low_suspicious': f"Negative net flow of ${value:,.2f} may indicate layering activity",
                'high_benign': f"Net flow of ${value:,.2f} is typical for business operations",
                'low_benign': f"Balanced net flow suggests normal account activity"
            },
            'burst_score': {
                'high_suspicious': f"High burst score ({value:.3f}) indicates clustering of transactions in short time",
                'low_suspicious': f"Burst score ({value:.3f}) shows some transaction clustering",
                'high_benign': f"Burst score ({value:.3f}) is normal for this account type",
                'low_benign': f"Low burst score indicates well-distributed transaction timing"
            },
            'wire_count': {
                'high_suspicious': f"{int(value)} wire transfers is unusually high (common in laundering)",
                'low_suspicious': f"{int(value)} wire transfers warrants review",
                'high_benign': f"{int(value)} wire transfers is normal for international business",
                'low_benign': f"{int(value)} wire transfers is typical"
            },
            'wire_ratio': {
                'high_suspicious': f"{value:.1%} of transactions being wires is unusually high",
                'low_suspicious': f"Wire ratio of {value:.1%} is slightly elevated",
                'high_benign': f"Wire ratio of {value:.1%} is normal for this account",
                'low_benign': f"Wire ratio of {value:.1%} is typical"
            },
            'round_amount_ratio': {
                'high_suspicious': f"{value:.1%} round amounts suggests possible structuring to avoid reporting",
                'low_suspicious': f"{value:.1%} round amounts is slightly suspicious",
                'high_benign': f"{value:.1%} round amounts is normal for business payments",
                'low_benign': f"Variable amount pattern is typical for legitimate activity"
            }
        }

        # Determine if value is high or low (simplified)
        is_high = value > 0.5  # Adjust based on feature scale

        # Determine interpretation key
        if impact > 0:  # Increases suspicion
            key = 'high_suspicious' if is_high else 'low_suspicious'
        else:  # Decreases suspicion (more benign)
            key = 'high_benign' if is_high else 'low_benign'

        # Get interpretation
        if feature in interpretations:
            return interpretations[feature].get(key, f"Feature value: {value:.2f}")
        else:
            # Generic interpretation
            if impact > 0:
                return f"This value ({value:.2f}) is associated with suspicious patterns"
            else:
                return f"This value ({value:.2f}) is associated with normal patterns"

    def generate_report_recommendation(self, explanation, prediction, probability):
        """
        Generate detailed investigator recommendation

        Args:
            explanation: SHAP explanation
            prediction: Model prediction
            probability: Prediction probability

        Returns:
            String with investigation recommendation
        """
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("INVESTIGATOR RECOMMENDATION")
        lines.append("=" * 80)
        lines.append("")

        if prediction == 1:
            # Suspicious alert
            if probability > 0.8:
                priority = "HIGH PRIORITY"
                action = "Immediate investigation recommended"
            elif probability > 0.6:
                priority = "MEDIUM PRIORITY"
                action = "Investigation recommended within 24 hours"
            else:
                priority = "LOW PRIORITY"
                action = "Review when resources available"

            lines.append(f"Priority Level: {priority}")
            lines.append(f"Recommended Action: {action}")
            lines.append("")
            lines.append("Investigation Focus:")

            # Identify key suspicious factors
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']

            suspicious_features = []
            for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
                if shap_val > 0:  # Contributing to suspicion
                    suspicious_features.append((feature, shap_val))

            suspicious_features.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, _) in enumerate(suspicious_features[:3], 1):
                lines.append(f"  {i}. Examine {self._format_feature_name(feature)} pattern")

        else:
            # Benign alert
            lines.append("Priority Level: FALSE POSITIVE (No Investigation)")
            lines.append("Recommended Action: Close alert")
            lines.append("")
            lines.append(f"Confidence Level: {(1-probability):.1%}")
            lines.append("")
            lines.append("Key factors supporting benign classification:")

            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']

            benign_features = []
            for feature, shap_val in zip(feature_names, shap_values):
                if shap_val < 0:  # Contributing to benign classification
                    benign_features.append((feature, abs(shap_val)))

            benign_features.sort(key=lambda x: x[1], reverse=True)

            for i, (feature, _) in enumerate(benign_features[:3], 1):
                lines.append(f"  {i}. {self._format_feature_name(feature)} is within normal range")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
