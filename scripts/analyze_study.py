"""
==============================================================================
Optuna Study Analysis Script
==============================================================================

This script analyzes a completed Optuna study from its SQLite database and
generates a summary report. It identifies the best trial for each unique
target column based on the highest Sharpe ratio.

Usage:
    python scripts/analyze_study.py [path_to_study.sqlite]

Example:
    python scripts/analyze_study.py output/optuna_stage/study_a.sqlite

If no path is provided, it will try to find the database in the default
output directory.
"""

import sys
import pandas as pd
import optuna
from pathlib import Path

def analyze_study(db_path: Path):
    """
    Loads an Optuna study, analyzes it, and prints a summary report.

    Args:
        db_path (Path): The path to the SQLite database file.
    """
    if not db_path.exists():
        print(f"❌ Error: Database file not found at '{db_path}'")
        sys.exit(1)

    study_name = "stageA_preprocess_selection"
    storage_url = f"sqlite:///{db_path}"

    try:
        print(f"🔗 Loading study '{study_name}' from '{storage_url}'...")
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"❌ Error: Study '{study_name}' not found in the database.")
        print("Please ensure the study name matches the one used during the run.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading the study: {e}")
        sys.exit(1)

    # Get all completed trials into a pandas DataFrame
    try:
        df = study.trials_dataframe()
        df = df[df.state == 'COMPLETE']
    except Exception as e:
        print(f"Could not fetch trials dataframe: {e}")
        return

    if df.empty:
        print("No completed trials found in the study.")
        return

    print(f"✅ Study loaded successfully. Analyzing {len(df)} completed trials...")

    # --- Analysis: Best trial per target column ---
    target_attr = 'user_attrs_target_column'
    value_metric = 'value' # 'value' is the Sharpe Ratio in a single-objective study

    if target_attr not in df.columns:
        print(f"❌ Error: The attribute '{target_attr}' was not found in the trial results.")
        print("Please ensure trials are saving the 'target_column' user attribute.")
        return

    # Find the index of the best trial for each target
    best_trials_idx = df.loc[df.groupby(target_attr)[value_metric].idxmax()]

    # Sort the results by the Sharpe Ratio in descending order
    best_trials_sorted = best_trials_idx.sort_values(by=value_metric, ascending=False)

    print("\n" + "="*80)
    print("🏆 BEST TRIAL PER TARGET (Ranked by Sharpe Ratio) 🏆")
    print("="*80)

    if best_trials_sorted.empty:
        print("Could not determine best trials. No trials with target attribute found.")
        return

    # Define columns to display in the report
    display_columns = [
        'number',
        value_metric,
        target_attr,
        'user_attrs_n_features_selected',
        'user_attrs_entry_threshold',
    ]
    
    # Filter out columns that might not exist
    existing_display_columns = [col for col in display_columns if col in best_trials_sorted.columns]

    # Rename columns for better readability
    report = best_trials_sorted[existing_display_columns].rename(columns={
        'number': 'Trial',
        value_metric: 'Sharpe Ratio',
        target_attr: 'Target',
        'user_attrs_n_features_selected': 'Features',
        'user_attrs_entry_threshold': 'Entry Threshold',
    })

    # Format the report for printing
    report['Sharpe Ratio'] = report['Sharpe Ratio'].map('{:.4f}'.format)
    if 'Entry Threshold' in report.columns:
        report['Entry Threshold'] = report['Entry Threshold'].map('{:.6f}'.format)

    print(report.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    # Determine the database path
    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])
    else:
        # Default path if none is provided
        db_path = Path("output/optuna_stage/study_a.sqlite")
        print(f"⚠️ No database path provided. Using default: '{db_path}'")

    analyze_study(db_path)
