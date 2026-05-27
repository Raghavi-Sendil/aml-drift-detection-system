"""
Data Processor Module
Handles preprocessing and joining of transaction and alert data
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataProcessor:
    def __init__(self):
        self.required_columns = ['alert_id', 'timestamp', 'amount']

    def process_and_join(self, transactions_df, alerts_df=None):
        """
        Process and join transactions with alerts

        Args:
            transactions_df: DataFrame with transaction data
            alerts_df: Optional DataFrame with alert labels (REQUIRED for this dataset)

        Returns:
            Processed and joined DataFrame
        """
        print("\n  Processing transactions and alerts...")

        # Make a copy
        df = transactions_df.copy()

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle the SynthAML dataset format specifically
        # Transactions have: AlertID, Timestamp, Entry, Type, Size
        if 'size' in df.columns and 'entry' in df.columns:
            print("  ✓ Detected SynthAML dataset format")

            # Rename columns to standard format
            column_mapping = {
                'alertid': 'alert_id',
                'size': 'amount',
                'entry': 'entry_type',
                'type': 'transaction_type'
            }
            df.rename(columns=column_mapping, inplace=True)

            # Convert size (appears to be log-scaled) - use absolute value as amount
            df['amount'] = df['amount'].abs()

        # Check for required columns (more flexible now)
        if 'alert_id' not in df.columns:
            if 'alertid' in df.columns:
                df.rename(columns={'alertid': 'alert_id'}, inplace=True)

        # Check for timestamp column and rename if needed
        if 'timestamp' not in df.columns:
            for alt in ['date', 'datetime', 'transaction_date', 'tx_date', 'time']:
                if alt in df.columns:
                    df.rename(columns={alt: 'timestamp'}, inplace=True)
                    print(f"  ✓ Renamed '{alt}' to 'timestamp'")
                    break

        # Process timestamp
        df = self._process_timestamp(df)

        # Process amounts
        df = self._process_amounts(df)

        # Process transaction types
        df = self._process_transaction_types(df)

        # Join with alerts if provided (REQUIRED)
        if alerts_df is not None:
            print("  ✓ Joining with alerts file...")
            alerts_df = alerts_df.copy()
            alerts_df.columns = alerts_df.columns.str.lower().str.strip()

            # Rename alert columns
            if 'alertid' in alerts_df.columns:
                alerts_df.rename(columns={'alertid': 'alert_id'}, inplace=True)

            if 'outcome' in alerts_df.columns:
                alerts_df.rename(columns={'outcome': 'is_suspicious'}, inplace=True)

            # Keep only alert_id and is_suspicious from alerts
            alert_cols = ['alert_id', 'is_suspicious']
            if 'date' in alerts_df.columns:
                alert_cols.append('date')

            alerts_df = alerts_df[alert_cols]

            # Merge
            df = df.merge(alerts_df, on='alert_id', how='left', suffixes=('', '_alert'))
            print(f"  ✓ Joined {len(df):,} transactions with {alerts_df['alert_id'].nunique():,} alerts")
        else:
            print("  ⚠ Warning: No alerts file provided")

        # Ensure is_suspicious column exists
        if 'is_suspicious' not in df.columns:
            if 'label' in df.columns:
                df['is_suspicious'] = df['label']
            elif 'is_sar' in df.columns:
                df['is_suspicious'] = df['is_sar']
            elif 'outcome' in df.columns:
                df['is_suspicious'] = df['outcome']
            else:
                print("  ⚠ Warning: No label column found. Creating placeholder labels.")
                df['is_suspicious'] = 0  # Placeholder

        # Handle missing values
        df = self._handle_missing_values(df)

        print(f"  ✓ Processed {len(df):,} transaction records")
        print(f"  ✓ Alerts: {df['alert_id'].nunique():,}")
        print(f"  ✓ Suspicious: {df.groupby('alert_id')['is_suspicious'].first().sum()}")
        print(f"  ✓ Benign: {df.groupby('alert_id')['is_suspicious'].first().value_counts().get(0, 0)}")

        return df

    def _validate_columns(self, df):
        """Validate that required columns exist"""
        missing = []
        for col in self.required_columns:
            if col not in df.columns:
                # Try alternative names
                alternatives = self._get_alternative_names(col)
                found = False
                for alt in alternatives:
                    if alt in df.columns:
                        df.rename(columns={alt: col}, inplace=True)
                        found = True
                        break
                if not found:
                    missing.append(col)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _get_alternative_names(self, column):
        """Get alternative column names"""
        alternatives = {
            'alert_id': ['alertid', 'alert', 'id', 'alert_key'],
            'timestamp': ['date', 'datetime', 'transaction_date', 'tx_date', 'time'],
            'amount': ['transaction_amount', 'tx_amount', 'value', 'usd_amount']
        }
        return alternatives.get(column, [])

    def _process_timestamp(self, df):
        """Process timestamp column"""
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            print("  ⚠ Warning: No timestamp column found")
            df['timestamp'] = pd.to_datetime('2020-01-01')

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print("  ✓ Converted timestamp to datetime")
            except Exception as e:
                print(f"  ⚠ Warning: Could not parse timestamps: {e}")
                print("    Using default timestamp: 2020-01-01")
                df['timestamp'] = pd.to_datetime('2020-01-01')

        # Verify it's datetime before using .dt accessor
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Add time-based features
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour'] = df['timestamp'].dt.hour
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        else:
            print("  ⚠ Warning: Timestamp not in datetime format, using defaults")
            df['day_of_week'] = 0
            df['hour'] = 0
            df['is_weekend'] = 0

        return df

    def _process_amounts(self, df):
        """Process transaction amounts"""
        # Convert to numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        # Handle negative amounts (some systems use negative for debits)
        df['amount_abs'] = df['amount'].abs()

        # Handle zero or missing amounts
        df['amount'] = df['amount'].fillna(0)
        df['amount_abs'] = df['amount_abs'].fillna(0)

        return df

    def _process_transaction_types(self, df):
        """Process transaction types"""
        # Look for transaction type column
        type_cols = ['transaction_type', 'type', 'tx_type', 'payment_type']
        type_col = None

        for col in type_cols:
            if col in df.columns:
                type_col = col
                break

        if type_col:
            df['transaction_type'] = df[type_col].str.lower().str.strip()
        else:
            print("  ⚠ Warning: No transaction type column found")
            df['transaction_type'] = 'unknown'

        # Standardize types
        df['is_credit'] = df['transaction_type'].str.contains('credit|deposit|incoming',
                                                               case=False, na=False).astype(int)
        df['is_debit'] = df['transaction_type'].str.contains('debit|withdrawal|outgoing',
                                                              case=False, na=False).astype(int)
        df['is_wire'] = df['transaction_type'].str.contains('wire|transfer|ach',
                                                             case=False, na=False).astype(int)

        return df

    def _handle_missing_values(self, df):
        """Handle missing values"""
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Fill categorical columns with 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

        return df

    def aggregate_by_alert(self, df):
        """
        Aggregate transaction-level data to alert-level

        This is useful when you have multiple transactions per alert
        """
        print("\n  Aggregating transactions by alert...")

        # Sort by alert and timestamp
        df = df.sort_values(['alert_id', 'timestamp'])

        # Group by alert
        alert_groups = df.groupby('alert_id')

        # Aggregate features
        alert_data = pd.DataFrame()
        alert_data['alert_id'] = alert_groups['alert_id'].first()
        alert_data['transaction_count'] = alert_groups.size()
        alert_data['total_amount'] = alert_groups['amount_abs'].sum()
        alert_data['mean_amount'] = alert_groups['amount_abs'].mean()
        alert_data['max_amount'] = alert_groups['amount_abs'].max()
        alert_data['min_amount'] = alert_groups['amount_abs'].min()
        alert_data['std_amount'] = alert_groups['amount_abs'].std().fillna(0)

        # Net flow
        alert_data['net_flow'] = alert_groups['amount'].sum()

        # Transaction type counts
        alert_data['credit_count'] = alert_groups['is_credit'].sum()
        alert_data['debit_count'] = alert_groups['is_debit'].sum()
        alert_data['wire_count'] = alert_groups['is_wire'].sum()

        # Time-based features
        alert_data['time_span_hours'] = (
            alert_groups['timestamp'].max() - alert_groups['timestamp'].min()
        ).dt.total_seconds() / 3600

        # Label
        alert_data['is_suspicious'] = alert_groups['is_suspicious'].first()

        print(f"  ✓ Aggregated to {len(alert_data):,} alerts")

        return alert_data.reset_index(drop=True)
