"""
Quick Start Script for AML Drift Detection System
Run this to get started immediately with synthetic data
"""

import sys
from example_workflow import generate_synthetic_data, run_full_workflow


def print_menu():
    print("\n" + "=" * 80)
    print("AML DRIFT DETECTION SYSTEM - QUICK START")
    print("=" * 80)
    print("\nChoose an option:")
    print("  1. Run full workflow with synthetic data (recommended for first time)")
    print("  2. Generate sample CSV for your own testing")
    print("  3. Launch web interface")
    print("  4. Exit")
    print("=" * 80)


def generate_sample_csv():
    print("\nGenerating sample CSV...")
    df = generate_synthetic_data(n_alerts=500)
    output_path = "sample_transactions.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Sample data saved to: {output_path}")
    print(f"   - {len(df):,} transactions")
    print(f"   - {df['alert_id'].nunique():,} alerts")
    print(f"\nYou can now:")
    print("  1. Use this CSV with the web interface (streamlit run web_interface.py)")
    print("  2. Modify it to match your data format")
    print("  3. Use it as a template for your own data")


def launch_web_interface():
    print("\nLaunching web interface...")
    print("=" * 80)
    print("The web interface will open in your browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the server")
    print()

    import subprocess
    try:
        subprocess.run(["streamlit", "run", "web_interface.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        print("\nMake sure streamlit is installed:")
        print("  pip install streamlit plotly")


def main():
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            print("\n🚀 Running full workflow...")
            print("This will:")
            print("  1. Generate synthetic AML data")
            print("  2. Process and engineer features")
            print("  3. Train models")
            print("  4. Detect drift")
            print("  5. Classify and explain alerts")
            print("\nThis may take a few minutes...")
            input("\nPress Enter to continue...")
            run_full_workflow()
            input("\n\nPress Enter to return to menu...")

        elif choice == '2':
            generate_sample_csv()
            input("\nPress Enter to return to menu...")

        elif choice == '3':
            launch_web_interface()
            input("\nPress Enter to return to menu...")

        elif choice == '4':
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print("\n⚠️ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
