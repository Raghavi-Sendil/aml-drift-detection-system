"""
Web Interface for AML Drift Detection System
Interactive dashboard for uploading data, training models, and classifying alerts

Run with: streamlit run web_interface.py
"""

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    exit(1)

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from main import AMLDriftDetectionSystem


# Page configuration
st.set_page_config(
    page_title="AML Drift Detection System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = AMLDriftDetectionSystem()
    st.session_state.trained = False
    st.session_state.features_df = None
    st.session_state.windows = None


def main():
    st.title("🔍 AML Alert Classification & Drift Detection")
    st.markdown("""
    **Anti-Money Laundering alert classification system with automatic drift detection and SHAP explainability**

    Based on research: *"AML Alert Classification Under Transaction Behavior Shifts for False Positive Reduction"*
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Home",
        "📤 Upload Data",
        "🔧 Feature Engineering",
        "🎯 Train Model",
        "📊 Drift Detection",
        "🔮 Classify Alert",
        "🔬 SHAP Explained",
        "📈 Analytics"
    ])

    if page == "🏠 Home":
        show_home()
    elif page == "📤 Upload Data":
        show_upload_data()
    elif page == "🔧 Feature Engineering":
        show_feature_engineering()
    elif page == "🎯 Train Model":
        show_train_model()
    elif page == "📊 Drift Detection":
        show_drift_detection()
    elif page == "🔮 Classify Alert":
        show_classify_alert()
    elif page == "🔬 SHAP Explained":
        show_shap_explained()
    elif page == "📈 Analytics":
        show_analytics()


def show_home():
    st.header("Welcome to AML Drift Detection System")

    # Highlight SHAP prominently
    st.markdown("""
    ### 🎯 **Powered by SHAP (SHapley Additive exPlanations)**

    **The main feature of this system:** Every prediction comes with **transparent, explainable reasoning**
    that compliance teams can trust. Not just a classification - a complete explanation of *why*.

    🏆 **Gold standard for explainable AI in financial services**
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🔍 Explainability", "SHAP", "AI You Can Trust")
        st.success("**Why it matters:** Meet regulatory requirements with transparent AI decisions")

    with col2:
        st.metric("🎯 Features", "18+", "Behavioral Patterns")
        st.info("Automated feature engineering following research methodology")

    with col3:
        st.metric("📊 Drift Detection", "3 Methods", "PSI, KS, KL")
        st.info("Real-time monitoring of distribution shifts")

    # SHAP explanation section
    st.markdown("---")
    st.subheader("🔬 What is SHAP?")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** is the gold standard for explainable AI in financial services.

        **Why SHAP for AML?**
        - ✅ **Regulatory Compliance**: Explain every decision to auditors
        - ✅ **Investigator Trust**: Show *why* an alert is suspicious
        - ✅ **Reduce False Positives**: Understand model reasoning
        - ✅ **Audit Trail**: Document decision-making process
        - ✅ **Model Validation**: Verify model uses correct features

        **What You Get:**
        - Feature importance for each prediction
        - Direction of impact (increases/decreases suspicion)
        - Magnitude of contribution
        - Human-readable explanations
        """)

    with col2:
        st.info("""
        **Example Output:**

        Alert #1234
        ⚠️ **SUSPICIOUS**

        Top Factors:
        1. Wire Count: 8
           → Increases suspicion
        2. Burst Score: 0.8
           → Rapid transactions
        3. Round Amounts: 60%
           → Structuring pattern
        """)

    st.markdown("---")

    st.markdown("---")

    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Upload Data**: Provide transaction CSV file
    2. **Feature Engineering**: Automatically create behavioral features
    3. **Train Model**: Train XGBoost or Logistic Regression
    4. **Drift Detection**: Monitor for behavioral shifts
    5. **Classify Alerts**: Get predictions with explanations
    """)

    st.markdown("---")

    st.subheader("📋 Required Data Format")
    st.code("""
    CSV columns required:
    - alert_id: Alert identifier
    - timestamp: Transaction timestamp
    - amount: Transaction amount
    - transaction_type: Type (credit/debit/wire)
    - is_suspicious: Label (1=suspicious, 0=benign)
    """)


def show_upload_data():
    st.header("📤 Upload Transaction Data")

    st.info("📋 **For SynthAML dataset**: Upload BOTH files - transactions and alerts")

    # Choose input method
    input_method = st.radio(
        "Choose input method:",
        ["📁 File Path (Recommended for large files > 200MB)", "⬆️ Upload from Computer"],
        help="For large files, using file path is much faster than uploading"
    )

    if input_method == "📁 File Path (Recommended for large files > 200MB)":
        # File path input
        st.subheader("Enter File Paths")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**1. Transactions File**")
            transactions_path = st.text_input(
                "Transactions CSV path",
                value="/Users/sycamore/Downloads/synthetic_transactions.csv",
                help="Full path to transactions CSV file"
            )

            # Check if file exists
            if transactions_path:
                from pathlib import Path
                if Path(transactions_path).exists():
                    size_mb = Path(transactions_path).stat().st_size / (1024 * 1024)
                    st.success(f"✅ File exists ({size_mb:.1f} MB)")
                else:
                    st.error("❌ File not found")

        with col2:
            st.markdown("**2. Alerts File**")
            alerts_path = st.text_input(
                "Alerts CSV path",
                value="/Users/sycamore/Downloads/synthetic_alerts_fixed.csv",
                help="Full path to alerts CSV file"
            )

            # Check if file exists
            if alerts_path:
                from pathlib import Path
                if Path(alerts_path).exists():
                    size_kb = Path(alerts_path).stat().st_size / 1024
                    st.success(f"✅ File exists ({size_kb:.1f} KB)")
                else:
                    st.error("❌ File not found")

        # Process button
        if st.button("Process Data", type="primary", disabled=(not transactions_path)):
            if transactions_path:
                try:
                    with st.spinner("Processing data... This may take a few minutes for large files..."):
                        # Process data directly from paths
                        processed_data = st.session_state.system.upload_and_process_datasets(
                            transactions_path=transactions_path,
                            alerts_path=alerts_path if alerts_path else None
                        )

                    st.success(f"✅ Processed {len(processed_data):,} transaction records")

                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(processed_data.head(100))

                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", f"{len(processed_data):,}")
                    with col2:
                        n_alerts = processed_data['alert_id'].nunique()
                        st.metric("Unique Alerts", f"{n_alerts:,}")
                    with col3:
                        n_suspicious = processed_data.groupby('alert_id')['is_suspicious'].first().sum()
                        st.metric("Suspicious Alerts", f"{n_suspicious:,}")
                    with col4:
                        n_benign = n_alerts - n_suspicious
                        st.metric("Benign Alerts", f"{n_benign:,}")

                    # Store in session
                    st.session_state.processed_data = processed_data

                except Exception as e:
                    st.error(f"Error processing data: {e}")
                    st.exception(e)
                    st.info("Please ensure your file paths are correct and files are in the expected format")
            else:
                st.warning("Please enter the transactions file path")

    else:
        # File upload input (original method)
        st.warning("⚠️ Note: For files > 200MB, this may be slow. Consider using file path option above.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Transactions File")
            transactions_file = st.file_uploader(
                "Upload transactions CSV",
                type=['csv'],
                help="CSV file with transaction records (e.g., synthetic_transactions.csv)",
                key="transactions"
            )

        with col2:
            st.subheader("2. Alerts File")
            alerts_file = st.file_uploader(
                "Upload alerts CSV",
                type=['csv'],
                help="CSV file with alert labels (e.g., synthetic_alerts_fixed.csv)",
                key="alerts"
            )

        # Process button
        if st.button("Process Data", type="primary", disabled=(transactions_file is None)):
            if transactions_file is not None:
                try:
                    with st.spinner("Processing data... This may take a few minutes..."):
                        # Save uploaded files temporarily
                        temp_transactions_path = "/tmp/uploaded_transactions.csv"
                        st.info("Reading transactions file...")
                        transactions_df = pd.read_csv(transactions_file)
                        transactions_df.to_csv(temp_transactions_path, index=False)

                        temp_alerts_path = None
                        if alerts_file is not None:
                            st.info("Reading alerts file...")
                            temp_alerts_path = "/tmp/uploaded_alerts.csv"
                            alerts_df = pd.read_csv(alerts_file)
                            alerts_df.to_csv(temp_alerts_path, index=False)

                        # Process data
                        st.info("Processing and joining data...")
                        processed_data = st.session_state.system.upload_and_process_datasets(
                            transactions_path=temp_transactions_path,
                            alerts_path=temp_alerts_path
                        )

                    st.success(f"✅ Processed {len(processed_data):,} transaction records")

                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(processed_data.head(100))

                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Transactions", f"{len(processed_data):,}")
                    with col2:
                        n_alerts = processed_data['alert_id'].nunique()
                        st.metric("Unique Alerts", f"{n_alerts:,}")
                    with col3:
                        n_suspicious = processed_data.groupby('alert_id')['is_suspicious'].first().sum()
                        st.metric("Suspicious Alerts", f"{n_suspicious:,}")
                    with col4:
                        n_benign = n_alerts - n_suspicious
                        st.metric("Benign Alerts", f"{n_benign:,}")

                    # Store in session
                    st.session_state.processed_data = processed_data

                except Exception as e:
                    st.error(f"Error processing data: {e}")
                    st.exception(e)
                    st.info("Please ensure your CSVs are in the correct format")
            else:
                st.warning("Please upload the transactions file first")


def show_feature_engineering():
    st.header("🔧 Feature Engineering")

    if not hasattr(st.session_state, 'processed_data'):
        st.warning("⚠️ Please upload data first")
        return

    if st.button("Generate Features", type="primary"):
        with st.spinner("Engineering features..."):
            try:
                features_df = st.session_state.system.engineer_features()
                st.session_state.features_df = features_df

                st.success(f"✅ Created {len(features_df.columns)} features")

                # Show features
                st.subheader("Generated Features")
                feature_cols = [col for col in features_df.columns
                               if col not in ['alert_id', 'is_suspicious', 'timestamp']]

                for i, feature in enumerate(feature_cols):
                    if i % 3 == 0:
                        cols = st.columns(3)
                    with cols[i % 3]:
                        st.metric(feature, f"{features_df[feature].mean():.2f}")

                # Show distribution
                st.subheader("Feature Distributions")
                selected_feature = st.selectbox("Select feature to visualize", feature_cols)

                fig = px.histogram(
                    features_df,
                    x=selected_feature,
                    color='is_suspicious',
                    title=f"Distribution of {selected_feature}",
                    labels={'is_suspicious': 'Suspicious'}
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")


def show_train_model():
    st.header("🎯 Train Classification Model")

    if st.session_state.features_df is None:
        st.warning("⚠️ Please engineer features first")
        return

    # Model configuration
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Model Type", ["XGBoost", "Logistic Regression"])

    with col2:
        train_split = st.slider("Training Window (%)", 20, 60, 40)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Split windows
                window1, window2, window3 = st.session_state.system.split_temporal_windows(
                    train_ratio=train_split/100
                )
                st.session_state.windows = (window1, window2, window3)

                # Train
                use_xgboost = (model_type == "XGBoost")
                model = st.session_state.system.train_baseline_model(use_xgboost=use_xgboost)

                st.session_state.trained = True

                st.success(f"✅ Model trained successfully")

                # Evaluate on Window 2
                metrics = st.session_state.system.evaluate_model(window2, "Window 2")

                # Display metrics
                st.subheader("Model Performance")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col4:
                    st.metric("FPR", f"{metrics['fpr']:.3f}")

                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm_data = [
                    ['True Negative', metrics['true_negatives']],
                    ['False Positive', metrics['false_positives']],
                    ['False Negative', metrics['false_negatives']],
                    ['True Positive', metrics['true_positives']]
                ]
                st.table(pd.DataFrame(cm_data, columns=['Metric', 'Count']))

            except Exception as e:
                st.error(f"Error: {e}")


def show_drift_detection():
    st.header("📊 Drift Detection")

    if not st.session_state.trained:
        st.warning("⚠️ Please train model first")
        return

    if st.button("Detect Drift", type="primary"):
        with st.spinner("Analyzing drift..."):
            try:
                window1, window2, window3 = st.session_state.windows

                # Detect drift in Window 3
                drift_detected, drift_results = st.session_state.system.detect_drift(
                    window3, "Window 3"
                )

                if drift_detected:
                    st.error("⚠️ DRIFT DETECTED")
                else:
                    st.success("✅ No significant drift")

                # Show drift metrics
                st.subheader("Drift Metrics by Feature")

                drift_df = pd.DataFrame([
                    {
                        'Feature': feature,
                        'PSI': results['psi'],
                        'KS Statistic': results['ks_statistic'],
                        'KL Divergence': results['kl_divergence'],
                        'Drift': '⚠️ Yes' if results['drift_detected'] else '✓ No'
                    }
                    for feature, results in drift_results.items()
                ])

                st.dataframe(drift_df, use_container_width=True)

                # Visualize drift
                st.subheader("PSI Scores")
                fig = px.bar(
                    drift_df.sort_values('PSI', ascending=False),
                    x='Feature',
                    y='PSI',
                    color='Drift',
                    title="Population Stability Index by Feature"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Retraining recommendation
                metrics = st.session_state.system.evaluate_model(window3, "Window 3")
                needs_retraining, reasons = st.session_state.system.decide_retraining(
                    drift_detected, metrics
                )

                st.subheader("Retraining Recommendation")
                if needs_retraining:
                    st.warning("⚠️ Retraining recommended")
                    for reason in reasons:
                        st.write(f"- {reason}")

                    if st.button("Retrain Model"):
                        with st.spinner("Retraining..."):
                            st.session_state.system.retrain_model(window1, window2)
                            st.success("✅ Model retrained successfully")
                else:
                    st.success("✓ Model performing well - No retraining needed")

            except Exception as e:
                st.error(f"Error: {e}")


def show_classify_alert():
    st.header("🔮 Classify New Alert")

    if not st.session_state.trained:
        st.warning("⚠️ Please train model first")
        return

    st.subheader("Enter Alert Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        tx_count = st.number_input("Transaction Count", 1, 100, 10)
        mean_size = st.number_input("Mean Transaction Size", 0.0, 100000.0, 5000.0)
        max_size = st.number_input("Max Transaction Size", 0.0, 100000.0, 10000.0)
        net_flow = st.number_input("Net Flow", -100000.0, 100000.0, 50000.0)
        burst_score = st.number_input("Burst Score", 0.0, 1.0, 0.5)
        avg_time = st.number_input("Avg Time Between Tx (hours)", 0.0, 100.0, 5.0)

    with col2:
        credit_count = st.number_input("Credit Count", 0, 100, 5)
        debit_count = st.number_input("Debit Count", 0, 100, 5)
        wire_count = st.number_input("Wire Count", 0, 100, 3)
        wire_ratio = st.number_input("Wire Ratio", 0.0, 1.0, 0.3)
        round_ratio = st.number_input("Round Amount Ratio", 0.0, 1.0, 0.4)

    with col3:
        tx_velocity = st.number_input("Transaction Velocity", 0.0, 20.0, 2.0)
        amount_variance = st.number_input("Amount Variance", 0.0, 5.0, 0.5)
        weekend_ratio = st.number_input("Weekend Ratio", 0.0, 1.0, 0.2)
        night_ratio = st.number_input("Night Ratio", 0.0, 1.0, 0.1)
        amount_concentration = st.number_input("Amount Concentration", 0.0, 1.0, 0.3)

    if st.button("Classify Alert", type="primary"):
        with st.spinner("Classifying..."):
            try:
                # Create feature dict
                alert_features = {
                    'transaction_count': tx_count,
                    'mean_transaction_size': mean_size,
                    'max_transaction_size': max_size,
                    'min_transaction_size': mean_size * 0.5,
                    'std_transaction_size': mean_size * 0.3,
                    'net_flow': net_flow,
                    'burst_score': burst_score,
                    'avg_time_between_tx': avg_time,
                    'credit_count': credit_count,
                    'debit_count': debit_count,
                    'wire_count': wire_count,
                    'credit_debit_ratio': credit_count / (debit_count + 1),
                    'wire_ratio': wire_ratio,
                    'transaction_velocity': tx_velocity,
                    'amount_variance': amount_variance,
                    'weekend_transaction_ratio': weekend_ratio,
                    'night_transaction_ratio': night_ratio,
                    'round_amount_ratio': round_ratio,
                    'amount_concentration': amount_concentration
                }

                # Classify
                prediction, probability, explanation = st.session_state.system.classify_alert(
                    alert_features
                )

                # Show result with prominent display
                st.markdown("---")
                st.markdown("## 🎯 Classification Result")

                col1, col2 = st.columns([1, 2])

                with col1:
                    if prediction == 1:
                        st.error("### ⚠️ SUSPICIOUS")
                        st.markdown("**Action Required**")
                        st.markdown("🔴 **Report to Investigators**")
                    else:
                        st.success("### ✅ BENIGN")
                        st.markdown("**No Action Needed**")
                        st.markdown("🟢 **Close Alert**")

                    st.metric("Confidence", f"{probability:.1%}",
                             delta=f"{abs(probability - 0.5):.1%} from threshold")

                with col2:
                    if prediction == 1:
                        st.info("""
                        **Next Steps:**
                        1. Review SHAP explanation below
                        2. Investigate top contributing factors
                        3. Check transaction history
                        4. Document findings
                        5. File SAR if confirmed
                        """)
                    else:
                        st.info("""
                        **Reasoning:**
                        - Transaction patterns are normal
                        - Key risk indicators below thresholds
                        - See SHAP explanation for details
                        - Safe to close alert
                        """)

                # SHAP Explanation - Make it prominent
                st.markdown("---")
                st.markdown("## 🔬 SHAP Explanation: Why This Decision?")

                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** breaks down exactly how each feature
                contributed to this classification. This explanation is:
                - ✅ **Auditable** - Can be shown to regulators
                - ✅ **Trustworthy** - Based on game theory (Nobel Prize-winning concept)
                - ✅ **Actionable** - Shows what to investigate
                """)

                shap_values = explanation['shap_values']
                feature_names = explanation['feature_names']
                feature_values = explanation['feature_values']

                # Top features
                abs_shap = np.abs(shap_values)
                top_indices = np.argsort(abs_shap)[-5:][::-1]

                # Visual SHAP chart
                st.subheader("📊 Feature Impact Visualization")

                fig = go.Figure(go.Bar(
                    x=[abs_shap[i] for i in top_indices],
                    y=[feature_names[i] for i in top_indices],
                    orientation='h',
                    marker_color=['#ff4444' if shap_values[i] > 0 else '#44ff44' for i in top_indices],
                    text=[f"{abs_shap[i]:.4f}" for i in top_indices],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="<b>SHAP Values - Feature Contribution to Decision</b>",
                    xaxis_title="<b>Impact Magnitude</b>",
                    yaxis_title="<b>Feature</b>",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Detailed explanation table
                st.subheader("📋 Detailed Feature Analysis")

                explanation_data = []
                for idx in top_indices:
                    feature = feature_names[idx]
                    value = feature_values[idx]
                    impact = shap_values[idx]
                    direction = "🔴 Increases Suspicion" if impact > 0 else "🟢 Decreases Suspicion"

                    # Add interpretation
                    if impact > 0:
                        interpretation = f"This value is {abs(impact):.4f} points higher than baseline"
                    else:
                        interpretation = f"This value is {abs(impact):.4f} points lower than baseline"

                    explanation_data.append({
                        'Feature': feature.replace('_', ' ').title(),
                        'Value': f"{value:.2f}",
                        'SHAP Impact': f"{impact:+.4f}",
                        'Effect': direction,
                        'Interpretation': interpretation
                    })

                st.dataframe(pd.DataFrame(explanation_data), use_container_width=True)

                # Human-readable summary
                st.subheader("💬 Plain English Explanation")

                with st.expander("🔍 Click to see detailed reasoning", expanded=True):
                    for i, idx in enumerate(top_indices[:3], 1):
                        feature = feature_names[idx]
                        value = feature_values[idx]
                        impact = shap_values[idx]

                        if impact > 0:
                            st.markdown(f"""
                            **{i}. {feature.replace('_', ' ').title()}** (Value: {value:.2f})
                            - **Increases** the likelihood of this being suspicious
                            - **Impact strength:** {abs(impact):.4f}
                            - **Why it matters:** This feature value is unusual compared to typical benign transactions
                            """)
                        else:
                            st.markdown(f"""
                            **{i}. {feature.replace('_', ' ').title()}** (Value: {value:.2f})
                            - **Decreases** the likelihood of this being suspicious
                            - **Impact strength:** {abs(impact):.4f}
                            - **Why it matters:** This feature value is consistent with normal transaction patterns
                            """)

                # Investigator guidance
                st.markdown("---")
                st.subheader("👨‍💼 Investigator Guidance")

                if prediction == 1:
                    st.warning("""
                    **Focus Your Investigation On:**
                    """)
                    for i, idx in enumerate(top_indices[:3], 1):
                        if shap_values[idx] > 0:
                            feature = feature_names[idx]
                            st.markdown(f"{i}. **{feature.replace('_', ' ').title()}** - Key risk indicator detected")
                else:
                    st.success("""
                    **Why This Alert is Likely Benign:**

                    The model found strong evidence that this transaction pattern is normal:
                    - Key risk indicators are within acceptable ranges
                    - Behavior matches legitimate business activity patterns
                    - No unusual clustering or structuring detected
                    """)

            except Exception as e:
                st.error(f"Error: {e}")


def show_shap_explained():
    st.header("🔬 SHAP Explainability - Deep Dive")

    st.markdown("""
    ## Understanding SHAP: The Gold Standard for Explainable AI in AML

    **SHAP (SHapley Additive exPlanations)** is based on cooperative game theory and provides
    the most theoretically sound method for explaining machine learning predictions.
    """)

    # What is SHAP
    st.markdown("---")
    st.subheader("🎯 What is SHAP?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **SHAP** assigns each feature an **importance value** for a particular prediction.

        **Key Concepts:**
        - **Shapley Values**: From Nobel Prize-winning game theory
        - **Additive**: All SHAP values sum to the final prediction
        - **Consistent**: If a feature helps, it gets positive credit
        - **Local**: Explains individual predictions, not just the model

        **Why Shapley Values?**
        - Fair attribution of "credit" to each feature
        - Considers all possible feature combinations
        - Mathematically proven to be the only fair method
        """)

    with col2:
        st.info("""
        **Example:**

        **Prediction:** Suspicious (0.82 probability)

        **SHAP Breakdown:**
        - Base rate: 0.10 (10% of alerts are suspicious)
        - Wire count (+0.28): 8 wires is very high
        - Burst score (+0.22): Rapid transactions
        - Amount (+0.12): Just below $10K threshold
        - Time (-0.05): Business hours is normal
        - Weekend (+0.15): Weekend activity unusual

        **Sum:** 0.10 + 0.28 + 0.22 + 0.12 - 0.05 + 0.15 = 0.82 ✓
        """)

    # Why SHAP for AML
    st.markdown("---")
    st.subheader("🏦 Why SHAP for AML Compliance?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔍 Regulatory Compliance**

        - **SR 11-7 (Fed)**: Model validation
        - **GDPR**: Right to explanation
        - **FCRA**: Adverse action notices
        - **OCC Bulletin**: Model risk management

        SHAP provides auditable, defensible explanations.
        """)

    with col2:
        st.markdown("""
        **👨‍💼 Investigator Trust**

        - Shows *why* an alert fired
        - Highlights key risk factors
        - Supports investigation prioritization
        - Documents decision trail

        Investigators can see the model's "reasoning".
        """)

    with col3:
        st.markdown("""
        **📊 Model Validation**

        - Verify correct features used
        - Detect bias or errors
        - Monitor for drift
        - Identify false positives

        SHAP helps validate the model is working correctly.
        """)

    # How to read SHAP values
    st.markdown("---")
    st.subheader("📖 How to Read SHAP Values")

    st.markdown("""
    **SHAP values show how much each feature pushed the prediction up or down.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.success("""
        **Positive SHAP Value** 🔴

        - **Increases** suspicion
        - Pushes prediction toward "Suspicious"
        - Higher value = stronger effect

        **Example:**
        - `wire_count: +0.28`
        - This feature makes the alert **more suspicious**
        - The high wire count is a red flag
        """)

    with col2:
        st.info("""
        **Negative SHAP Value** 🟢

        - **Decreases** suspicion
        - Pushes prediction toward "Benign"
        - More negative = stronger effect

        **Example:**
        - `business_hours: -0.05`
        - This feature makes the alert **less suspicious**
        - Transactions during business hours are normal
        """)

    # Visual example
    st.markdown("---")
    st.subheader("📊 Visual Example")

    st.markdown("**Sample SHAP Waterfall Chart:**")

    # Create example data
    import plotly.graph_objects as go

    features = ['Base Rate', 'Wire Count', 'Burst Score', 'Round Amounts', 'Business Hours', 'Final Prediction']
    values = [0.10, 0.28, 0.22, 0.15, -0.05, 0.70]
    cumulative = [0.10, 0.38, 0.60, 0.75, 0.70, 0.70]

    fig = go.Figure(go.Waterfall(
        x=features,
        y=values,
        measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
        title="<b>SHAP Waterfall Chart Example</b><br>How features combine to make the final prediction",
        yaxis_title="Contribution to Suspicious Probability",
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Reading the chart:**
    1. Start at **Base Rate** (10% of alerts are suspicious)
    2. Each bar shows how a feature changes the prediction
    3. **Up arrows** (red) increase suspicion
    4. **Down arrows** (green) decrease suspicion
    5. End at **Final Prediction** (70% suspicious)
    """)

    # Comparison with other methods
    st.markdown("---")
    st.subheader("🆚 SHAP vs Other Explanation Methods")

    comparison_data = {
        'Method': ['SHAP', 'LIME', 'Feature Importance', 'Permutation', 'Attention'],
        'Theoretically Sound': ['✅ Yes', '❌ No', '❌ No', '❌ No', '❌ No'],
        'Local Explanations': ['✅ Yes', '✅ Yes', '❌ No', '❌ No', '✅ Yes'],
        'Consistent': ['✅ Yes', '❌ No', '❌ No', '✅ Yes', '❌ No'],
        'Model Agnostic': ['✅ Yes', '✅ Yes', '❌ No', '✅ Yes', '❌ No'],
        'Regulatory Accepted': ['✅ Yes', '⚠️ Maybe', '❌ No', '⚠️ Maybe', '❌ No']
    }

    st.table(pd.DataFrame(comparison_data))

    st.markdown("""
    **Why SHAP is preferred:**
    - Only method with solid mathematical foundation (Shapley values)
    - Provably fair and consistent
    - Widely accepted by regulators
    - Works with any model (tree-based, neural networks, etc.)
    """)

    # Practical tips
    st.markdown("---")
    st.subheader("💡 Practical Tips for Using SHAP")

    st.markdown("""
    **For Compliance Officers:**
    - Include SHAP explanations in SAR documentation
    - Use SHAP to demonstrate model fairness
    - Show SHAP outputs during audits
    - Keep SHAP values for audit trail

    **For Investigators:**
    - Focus on top 3-5 features with highest SHAP values
    - Red bars = investigate these patterns
    - Green bars = evidence of normal behavior
    - Use SHAP to prioritize investigation efforts

    **For Model Validators:**
    - Check that SHAP values make business sense
    - Verify important features align with AML typologies
    - Monitor SHAP distributions for drift
    - Use SHAP to identify model errors

    **For Data Scientists:**
    - Always provide SHAP explanations with predictions
    - Validate SHAP values against domain knowledge
    - Use SHAP for feature selection
    - Debug models using SHAP
    """)

    # References
    st.markdown("---")
    st.subheader("📚 Learn More")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Academic Papers:**
        - Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
        - Shapley (1953): "A Value for n-person Games"

        **Industry Standards:**
        - SR 11-7: Model Risk Management (Federal Reserve)
        - OCC Bulletin 2011-12: Sound Practices for Model Risk Management
        """)

    with col2:
        st.markdown("""
        **Implementation:**
        - SHAP Python Library: github.com/slundberg/shap
        - TreeExplainer: Fast SHAP for tree models
        - KernelExplainer: SHAP for any model

        **Tutorials:**
        - SHAP Documentation
        - Christoph Molnar: "Interpretable Machine Learning"
        """)


def show_analytics():
    st.header("📈 System Analytics")

    if not st.session_state.trained or st.session_state.windows is None:
        st.warning("⚠️ Please train model first")
        return

    window1, window2, window3 = st.session_state.windows

    # Performance over time
    st.subheader("Model Performance Over Time")

    metrics_w2 = st.session_state.system.evaluate_model(window2, "Window 2")
    metrics_w3 = st.session_state.system.evaluate_model(window3, "Window 3")

    performance_df = pd.DataFrame([
        {'Window': 'Window 2', 'Recall': metrics_w2['recall'], 'FPR': metrics_w2['fpr'],
         'Accuracy': metrics_w2['accuracy']},
        {'Window': 'Window 3', 'Recall': metrics_w3['recall'], 'FPR': metrics_w3['fpr'],
         'Accuracy': metrics_w3['accuracy']}
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=performance_df['Window'], y=performance_df['Recall'],
                             mode='lines+markers', name='Recall'))
    fig.add_trace(go.Scatter(x=performance_df['Window'], y=performance_df['FPR'],
                             mode='lines+markers', name='FPR'))
    fig.add_trace(go.Scatter(x=performance_df['Window'], y=performance_df['Accuracy'],
                             mode='lines+markers', name='Accuracy'))
    fig.update_layout(title="Metrics Across Time Windows", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions by Class")

    if st.session_state.features_df is not None:
        feature_cols = [col for col in st.session_state.features_df.columns
                       if col not in ['alert_id', 'is_suspicious', 'timestamp']]

        selected_feature = st.selectbox("Select feature", feature_cols)

        fig = px.box(
            st.session_state.features_df,
            x='is_suspicious',
            y=selected_feature,
            color='is_suspicious',
            title=f"{selected_feature} by Class"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
