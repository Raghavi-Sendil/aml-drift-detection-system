"""
Drift Detector Module
Implements statistical drift detection techniques:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test
- Kullback-Leibler Divergence
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import distance


class DriftDetector:
    def __init__(self, psi_threshold=0.2, ks_threshold=0.1):
        """
        Args:
            psi_threshold: PSI threshold for drift detection (0.1-0.2 = significant drift)
            ks_threshold: KS test threshold for drift detection
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def compute_baseline_stats(self, baseline_data):
        """
        Compute baseline statistics for drift detection

        Args:
            baseline_data: DataFrame with baseline (training) data

        Returns:
            Dictionary with baseline statistics for each feature
        """
        print("\n  Computing baseline statistics...")

        baseline_stats = {}

        for column in baseline_data.columns:
            if baseline_data[column].dtype in ['int64', 'float64']:
                stats_dict = {
                    'mean': baseline_data[column].mean(),
                    'std': baseline_data[column].std(),
                    'min': baseline_data[column].min(),
                    'max': baseline_data[column].max(),
                    'distribution': baseline_data[column].values.copy()
                }

                # Create bins for PSI calculation
                stats_dict['bins'] = self._create_bins(baseline_data[column])

                baseline_stats[column] = stats_dict

        print(f"  ✓ Baseline statistics computed for {len(baseline_stats)} features")

        return baseline_stats

    def detect_drift(self, baseline_stats, current_data):
        """
        Detect drift using PSI, KS test, and KL divergence

        Args:
            baseline_stats: Dictionary with baseline statistics
            current_data: DataFrame with current data

        Returns:
            Dictionary with drift detection results for each feature
        """
        drift_results = {}

        for column in current_data.columns:
            if column not in baseline_stats:
                continue

            # Calculate drift metrics
            psi = self._calculate_psi(
                baseline_stats[column],
                current_data[column]
            )

            ks_statistic, ks_pvalue = self._calculate_ks_test(
                baseline_stats[column]['distribution'],
                current_data[column].values
            )

            kl_divergence = self._calculate_kl_divergence(
                baseline_stats[column]['distribution'],
                current_data[column].values
            )

            # Determine if drift detected
            drift_detected = (
                psi > self.psi_threshold or
                ks_statistic > self.ks_threshold
            )

            drift_results[column] = {
                'psi': psi,
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'kl_divergence': kl_divergence,
                'drift_detected': drift_detected
            }

        return drift_results

    def _calculate_psi(self, baseline_stats, current_data):
        """
        Calculate Population Stability Index (PSI)

        PSI measures distribution shift between baseline and current data
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change (drift)

        Formula: PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        bins = baseline_stats['bins']
        baseline_dist = baseline_stats['distribution']

        # Bin both distributions
        baseline_counts, _ = np.histogram(baseline_dist, bins=bins)
        current_counts, _ = np.histogram(current_data, bins=bins)

        # Convert to percentages
        baseline_pct = (baseline_counts + 1) / (len(baseline_dist) + len(bins))
        current_pct = (current_counts + 1) / (len(current_data) + len(bins))

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi

    def _calculate_ks_test(self, baseline_dist, current_dist):
        """
        Calculate Kolmogorov-Smirnov test statistic

        KS test measures the maximum distance between cumulative distributions
        Higher KS statistic = more drift
        """
        ks_statistic, ks_pvalue = stats.ks_2samp(baseline_dist, current_dist)
        return ks_statistic, ks_pvalue

    def _calculate_kl_divergence(self, baseline_dist, current_dist):
        """
        Calculate Kullback-Leibler divergence

        KL divergence measures information loss when approximating one distribution
        with another
        Higher KL = more drift
        """
        # Create probability distributions
        bins = self._create_bins(baseline_dist)

        baseline_counts, _ = np.histogram(baseline_dist, bins=bins)
        current_counts, _ = np.histogram(current_dist, bins=bins)

        # Normalize to probabilities (add smoothing)
        baseline_prob = (baseline_counts + 1) / (baseline_counts.sum() + len(bins))
        current_prob = (current_counts + 1) / (current_counts.sum() + len(bins))

        # Calculate KL divergence
        kl_div = np.sum(current_prob * np.log(current_prob / baseline_prob))

        return kl_div

    def _create_bins(self, data, n_bins=10):
        """
        Create bins for histogram-based drift detection

        Args:
            data: Array-like data
            n_bins: Number of bins

        Returns:
            Array of bin edges
        """
        # Handle constant features
        if data.std() == 0:
            return np.array([data.min() - 1, data.min(), data.min() + 1])

        # Create quantile-based bins
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = np.quantile(data, quantiles)
            bins = np.unique(bins)  # Remove duplicates

            # Ensure at least 2 bins
            if len(bins) < 2:
                bins = np.array([data.min(), data.max()])

            return bins
        except:
            # Fallback to linear bins
            return np.linspace(data.min(), data.max(), n_bins + 1)

    def generate_drift_report(self, drift_results):
        """
        Generate a comprehensive drift report

        Args:
            drift_results: Dictionary from detect_drift()

        Returns:
            Dictionary with summary statistics
        """
        total_features = len(drift_results)
        drifted_features = sum(1 for r in drift_results.values() if r['drift_detected'])

        high_drift_features = []
        for feature, result in drift_results.items():
            if result['psi'] > 0.25:  # Very high drift
                high_drift_features.append({
                    'feature': feature,
                    'psi': result['psi'],
                    'ks_statistic': result['ks_statistic']
                })

        report = {
            'total_features': total_features,
            'drifted_features': drifted_features,
            'drift_percentage': (drifted_features / total_features * 100) if total_features > 0 else 0,
            'high_drift_features': high_drift_features,
            'needs_retraining': drifted_features / total_features > 0.3  # >30% drifted
        }

        return report

    def monitor_drift_over_time(self, baseline_stats, data_windows):
        """
        Monitor drift across multiple time windows

        Args:
            baseline_stats: Baseline statistics
            data_windows: List of DataFrames for each time window

        Returns:
            DataFrame with drift metrics over time
        """
        drift_over_time = []

        for i, window in enumerate(data_windows):
            drift_results = self.detect_drift(baseline_stats, window)

            # Aggregate drift metrics
            avg_psi = np.mean([r['psi'] for r in drift_results.values()])
            avg_ks = np.mean([r['ks_statistic'] for r in drift_results.values()])
            pct_drifted = sum(1 for r in drift_results.values() if r['drift_detected']) / len(drift_results)

            drift_over_time.append({
                'window': i + 1,
                'avg_psi': avg_psi,
                'avg_ks': avg_ks,
                'pct_features_drifted': pct_drifted
            })

        return pd.DataFrame(drift_over_time)


class AdaptiveThresholdDriftDetector(DriftDetector):
    """
    Enhanced drift detector with adaptive thresholds

    Adjusts thresholds based on model performance and business requirements
    """

    def __init__(self, initial_psi_threshold=0.2, initial_ks_threshold=0.1):
        super().__init__(initial_psi_threshold, initial_ks_threshold)
        self.threshold_history = []

    def adjust_thresholds(self, false_positive_rate, target_fpr=0.5):
        """
        Adjust drift detection thresholds based on false positive rate

        If FPR is too high, be more sensitive to drift (lower thresholds)
        If FPR is acceptable, be less sensitive (higher thresholds)

        Args:
            false_positive_rate: Current FPR
            target_fpr: Target FPR
        """
        if false_positive_rate > target_fpr * 1.2:  # FPR too high
            # Be more sensitive to drift
            self.psi_threshold *= 0.9
            self.ks_threshold *= 0.9
            adjustment = "Increased sensitivity (lowered thresholds)"

        elif false_positive_rate < target_fpr * 0.8:  # FPR too low
            # Be less sensitive to drift
            self.psi_threshold *= 1.1
            self.ks_threshold *= 1.1
            adjustment = "Decreased sensitivity (raised thresholds)"

        else:
            adjustment = "No adjustment needed"

        # Keep thresholds in reasonable range
        self.psi_threshold = np.clip(self.psi_threshold, 0.05, 0.3)
        self.ks_threshold = np.clip(self.ks_threshold, 0.05, 0.2)

        self.threshold_history.append({
            'psi_threshold': self.psi_threshold,
            'ks_threshold': self.ks_threshold,
            'fpr': false_positive_rate,
            'adjustment': adjustment
        })

        print(f"  Threshold adjustment: {adjustment}")
        print(f"  New PSI threshold: {self.psi_threshold:.3f}")
        print(f"  New KS threshold: {self.ks_threshold:.3f}")
