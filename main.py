"""
=============================================================================
MASTER EXECUTION PIPELINE
Project: Machine Learning Techniques for Financial Data
Activity Title: FA1 Group Activity — Stock Market Next-Day Closing Price Prediction
=============================================================================
This master script runs all 5 phases sequentially with full reproducibility.
"""

import os
import sys
import subprocess

# Ensure portable site-packages is in sys.path
PORTABLE_PACKAGES = r"C:\Users\Hp\PythonPortable\Lib\site-packages"
if os.path.exists(PORTABLE_PACKAGES) and PORTABLE_PACKAGES not in sys.path:
    sys.path.insert(0, PORTABLE_PACKAGES)

def run_phase(phase_num, script_name, description):
    print("\n" + "="*80)
    print(f"RUNNING PHASE {phase_num}: {description}")
    print(f"Executing: {script_name}")
    print("="*80)
    python_exe = sys.executable
    result = subprocess.run([python_exe, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Phase {phase_num} failed with return code {result.returncode}!")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Phase {phase_num} completed successfully.")

def main():
    print("*"*80)
    print("STARTING COMPLETE END-TO-END FINANCIAL MODELING PIPELINE")
    print("Asset: Reliance Industries Limited (RELIANCE.NS) | Period: 2021-2025")
    print("*"*80)

    # Phase 1
    run_phase(1, "phase1_data_sourcing.py", "Setup, Data Sourcing & Problem Confirmation")

    # Phase 2
    run_phase(2, "phase2_feature_engineering.py", "Data Cleaning, Technical Feature Engineering & Target Alignment")

    # Phase 3
    run_phase(3, "phase3_model_training.py", "Model Building, Time-Series Split & Prediction")

    # Phase 4
    run_phase(4, "phase4_evaluation_visualization.py", "Evaluation Metrics & Complete Visualization Suite")

    # Phase 5
    run_phase(5, "phase5_report_assembly.py", "Final Report Assembly & Evaluation Criteria Check")

    print("\n" + "*"*80)
    print("[ALL PHASES COMPLETE] End-to-End Pipeline Finished Successfully!")
    print("Outputs generated:")
    print("  - Tables:  outputs/tables/ (raw_data.csv, processed_data.csv, predictions.csv, metrics_summary.csv)")
    print("  - Charts:  outputs/charts/ (Charts 1-6 + Correlation Heatmap + Residual Diagnostic)")
    print("  - Models:  outputs/models/ (linear_regression.pkl, random_forest.pkl, scaler.pkl)")
    print("  - Report:  outputs/Final_Report.md")
    print("*"*80)

if __name__ == "__main__":
    main()
