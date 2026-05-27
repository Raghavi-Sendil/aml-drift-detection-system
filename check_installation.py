#!/usr/bin/env python3
"""
Check if all dependencies are installed correctly
"""

import sys


def check_module(module_name, import_name=None):
    """Check if a module is installed"""
    if import_name is None:
        import_name = module_name

    try:
        __import__(import_name)
        print(f"✅ {module_name}")
        return True
    except ImportError:
        print(f"❌ {module_name} - NOT INSTALLED")
        return False


def main():
    print("\n" + "=" * 80)
    print("AML DRIFT DETECTION SYSTEM - DEPENDENCY CHECK")
    print("=" * 80)
    print("\nChecking required packages...\n")

    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'scipy': 'scipy'
    }

    optional = {
        'xgboost': 'xgboost',
        'shap': 'shap',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'jupyter': 'jupyter'
    }

    print("REQUIRED PACKAGES:")
    print("-" * 80)
    required_ok = all(check_module(name, import_name) for name, import_name in required.items())

    print("\nOPTIONAL PACKAGES:")
    print("-" * 80)
    optional_results = {name: check_module(name, import_name)
                       for name, import_name in optional.items()}

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if required_ok:
        print("\n✅ All required packages are installed!")
    else:
        print("\n❌ Some required packages are missing.")
        print("\nTo install all required packages, run:")
        print("  pip install numpy pandas scikit-learn scipy")

    print("\nOptional packages status:")
    for name, installed in optional_results.items():
        status = "✅" if installed else "❌"
        print(f"  {status} {name}")

    if not optional_results.get('xgboost'):
        print("\n⚠️  XGBoost not installed - only Logistic Regression will be available")
        print("    To install: pip install xgboost")

    if not optional_results.get('shap'):
        print("\n⚠️  SHAP not installed - explanations will use fallback method")
        print("    To install: pip install shap")

    if not optional_results.get('streamlit'):
        print("\n⚠️  Streamlit not installed - web interface will not work")
        print("    To install: pip install streamlit plotly")

    print("\n" + "=" * 80)
    print("To install ALL packages at once:")
    print("  pip install -r requirements.txt")
    print("=" * 80)

    # Version info
    print("\n\nPython version:")
    print(f"  {sys.version}")

    if required_ok:
        print("\n✅ You're ready to go! Try:")
        print("  python quick_start.py")
        print("  python example_workflow.py")
        print("  streamlit run web_interface.py")

    print()


if __name__ == "__main__":
    main()
