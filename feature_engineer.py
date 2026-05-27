"""
Feature Engineer Module
Creates features following the methodology from the research paper:
- Transaction count per alert
- Mean/Max transaction size
- Net flow of money
- Burst score
- Average time between transactions
- Credit/Debit/Wire transfer counts
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def create_features(self, df):
        """
        Create all features from transaction data

        Args:
            df: DataFrame with processed transaction data

        Returns:
            DataFrame with engineered features
        """
        print("\n  Engineering features per research methodology...")

        # Group by alert_id
        alert_groups = df.groupby('alert_id')

        features = pd.DataFrame()
        features['alert_id'] = alert_groups['alert_id'].first()

        # Feature 1: Transaction count per alert
        features['transaction_count'] = alert_groups.size()
        print("  ✓ Transaction count")

        # Feature 2: Mean transaction size
        features['mean_transaction_size'] = alert_groups['amount_abs'].mean()
        print("  ✓ Mean transaction size")

        # Feature 3: Maximum transaction size
        features['max_transaction_size'] = alert_groups['amount_abs'].max()
        print("  ✓ Maximum transaction size")

        # Feature 4: Minimum transaction size
        features['min_transaction_size'] = alert_groups['amount_abs'].min()

        # Feature 5: Standard deviation of transaction size
        features['std_transaction_size'] = alert_groups['amount_abs'].std().fillna(0)

        # Feature 6: Net flow of money
        features['net_flow'] = alert_groups['amount'].sum()
        print("  ✓ Net flow of money")

        # Feature 7: Burst score (transaction frequency concentration)
        features['burst_score'] = self._calculate_burst_score(df)
        print("  ✓ Burst score")

        # Feature 8: Average time between transactions
        features['avg_time_between_tx'] = self._calculate_avg_time_between(df)
        print("  ✓ Average time between transactions")

        # Feature 9: Credit transaction count
        features['credit_count'] = alert_groups['is_credit'].sum()
        print("  ✓ Credit transaction count")

        # Feature 10: Debit transaction count
        features['debit_count'] = alert_groups['is_debit'].sum()
        print("  ✓ Debit transaction count")

        # Feature 11: Wire transfer count
        features['wire_count'] = alert_groups['is_wire'].sum()
        print("  ✓ Wire transfer count")

        # Additional derived features
        features['credit_debit_ratio'] = (
            features['credit_count'] / (features['debit_count'] + 1)
        )

        features['wire_ratio'] = (
            features['wire_count'] / features['transaction_count']
        )

        features['transaction_velocity'] = (
            features['transaction_count'] / (features['avg_time_between_tx'] + 1)
        )

        features['amount_variance'] = features['std_transaction_size'] / (
            features['mean_transaction_size'] + 1
        )

        # Time-based features
        features['weekend_transaction_ratio'] = (
            alert_groups['is_weekend'].mean()
        )

        features['night_transaction_ratio'] = (
            alert_groups['hour'].apply(lambda x: ((x >= 22) | (x <= 6)).sum()).values /
            features['transaction_count']
        )

        # Round-amount analysis (suspicious behavior indicator)
        features['round_amount_ratio'] = self._calculate_round_amount_ratio(df)

        # Amount concentration (Gini coefficient)
        features['amount_concentration'] = self._calculate_amount_concentration(df)

        # Add label
        features['is_suspicious'] = alert_groups['is_suspicious'].first().values

        # Store timestamp separately (to avoid dtype conflicts)
        timestamp_col = alert_groups['timestamp'].min()

        # Handle infinite and NaN values (only on numeric columns)
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        # Add timestamp back after numeric operations
        features['timestamp'] = timestamp_col.values

        self.feature_names = [col for col in features.columns
                              if col not in ['alert_id', 'is_suspicious', 'timestamp']]

        print(f"\n  ✓ Total features created: {len(self.feature_names)}")

        return features

    def _calculate_burst_score(self, df):
        """
        Calculate burst score: measures transaction frequency concentration

        Higher burst score = transactions clustered in time (suspicious)
        """
        burst_scores = []

        for alert_id, group in df.groupby('alert_id'):
            if len(group) <= 1:
                burst_scores.append(0)
                continue

            # Sort by timestamp
            group = group.sort_values('timestamp')
            timestamps = group['timestamp'].values

            # Calculate time differences in hours
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 'h')
                time_diffs.append(diff)

            if not time_diffs:
                burst_scores.append(0)
                continue

            # Burst score: inverse of average time difference
            # (shorter intervals = higher burst)
            avg_diff = np.mean(time_diffs)
            burst_score = 1.0 / (avg_diff + 1.0)

            burst_scores.append(burst_score)

        return burst_scores

    def _calculate_avg_time_between(self, df):
        """
        Calculate average time between transactions in hours
        """
        avg_times = []

        for alert_id, group in df.groupby('alert_id'):
            if len(group) <= 1:
                avg_times.append(0)
                continue

            group = group.sort_values('timestamp')
            timestamps = group['timestamp'].values

            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 'h')
                time_diffs.append(diff)

            avg_time = np.mean(time_diffs) if time_diffs else 0
            avg_times.append(avg_time)

        return avg_times

    def _calculate_round_amount_ratio(self, df):
        """
        Calculate ratio of round amounts (ending in 00, 000, etc.)

        Money launderers often use round amounts to avoid detection
        """
        round_ratios = []

        for alert_id, group in df.groupby('alert_id'):
            total_tx = len(group)
            round_tx = 0

            for amount in group['amount_abs']:
                # Check if amount is round (divisible by 100, 1000, etc.)
                if amount % 1000 == 0 or amount % 500 == 0:
                    round_tx += 1

            ratio = round_tx / total_tx if total_tx > 0 else 0
            round_ratios.append(ratio)

        return round_ratios

    def _calculate_amount_concentration(self, df):
        """
        Calculate Gini coefficient for amount concentration

        Higher concentration = suspicious (e.g., one large transaction among many small)
        """
        concentrations = []

        for alert_id, group in df.groupby('alert_id'):
            amounts = group['amount_abs'].values

            if len(amounts) <= 1:
                concentrations.append(0)
                continue

            # Sort amounts
            sorted_amounts = np.sort(amounts)

            # Calculate Gini coefficient
            n = len(sorted_amounts)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_amounts)) / (n * np.sum(sorted_amounts)) - (n + 1) / n

            concentrations.append(gini)

        return concentrations

    def simulate_drift(self, features_df, tx_size_increase=0.3, tx_freq_increase=0.2,
                       wire_increase=0.5):
        """
        Simulate behavioral drift by modifying features

        Following the paper's drift simulation:
        - Increase transaction size by 30%
        - Increase transaction frequency by 20%
        - Increase wire transfer activity by 50%

        Args:
            features_df: DataFrame with features
            tx_size_increase: Percentage increase in transaction size
            tx_freq_increase: Percentage increase in transaction frequency
            wire_increase: Percentage increase in wire transfers

        Returns:
            DataFrame with simulated drift
        """
        print(f"\n  Simulating drift:")
        print(f"  • Transaction size +{tx_size_increase*100:.0f}%")
        print(f"  • Transaction frequency +{tx_freq_increase*100:.0f}%")
        print(f"  • Wire transfer activity +{wire_increase*100:.0f}%")

        df_drift = features_df.copy()

        # Increase transaction sizes
        df_drift['mean_transaction_size'] *= (1 + tx_size_increase)
        df_drift['max_transaction_size'] *= (1 + tx_size_increase)
        df_drift['min_transaction_size'] *= (1 + tx_size_increase)

        # Increase transaction frequency (more transactions in same time period)
        df_drift['transaction_count'] = (
            df_drift['transaction_count'] * (1 + tx_freq_increase)
        ).astype(int)

        # Adjust time-based features
        df_drift['avg_time_between_tx'] /= (1 + tx_freq_increase)
        df_drift['burst_score'] *= (1 + tx_freq_increase)
        df_drift['transaction_velocity'] *= (1 + tx_freq_increase)

        # Increase wire transfers
        df_drift['wire_count'] = (
            df_drift['wire_count'] * (1 + wire_increase)
        ).astype(int)
        df_drift['wire_ratio'] = (
            df_drift['wire_count'] / df_drift['transaction_count']
        )

        print("  ✓ Drift simulation complete")

        return df_drift
