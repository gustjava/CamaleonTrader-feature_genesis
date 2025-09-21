#!/usr/bin/env python3
"""
Test script to verify the optimized trial logging system.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_optimized_logging():
    """Test the optimized trial logging functions."""
    print("🧪 Testing optimized trial logging system...")
    
    # Create mock configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = MagicMock()
        cfg.study.outputs.dir = temp_dir
        
        # Create mock study with multiple trials
        study = MagicMock()
        study.study_name = "test_preprocess_selection"
        study.trials = []
        
        # Create 12 mock trials
        for i in range(12):
            trial = MagicMock()
            trial.number = i
            trial.state.name = "COMPLETE"
            trial.values = [0.5 + i * 0.1]  # Increasing values
            trial.value = 0.5 + i * 0.1
            trial.params = {"test_param": i}
            trial.user_attrs = {"test_attr": f"value_{i}"}
            trial.datetime_start = pd.Timestamp.now()
            trial.datetime_complete = pd.Timestamp.now() + pd.Timedelta(minutes=5)
            trial.duration = pd.Timedelta(minutes=5)
            study.trials.append(trial)
        
        # Set best trial (highest value)
        study.best_trial = study.trials[-1]  # Last trial has highest value
        
        # Import the function (this will test if our changes compile)
        try:
            from orchestration.main import _save_trial_progress, _save_final_study_report
            print("✅ Successfully imported logging functions")
        except ImportError as e:
            print(f"❌ Failed to import functions: {e}")
            return False
        
        # Test progress logging for trial 10 (should trigger snapshot)
        trial_10 = study.trials[10]
        try:
            _save_trial_progress(cfg, study, trial_10, {})
            print("✅ Progress logging function executed without errors")
            
            # Check if snapshot file was created
            snapshot_files = list(Path(temp_dir).glob("study_snapshot_*.json"))
            if snapshot_files:
                with open(snapshot_files[0], 'r') as f:
                    snapshot_data = json.load(f)
                
                # Check structure
                assert "recent_trials" in snapshot_data
                assert "summary" in snapshot_data
                assert "note" in snapshot_data
                
                # Should have at most 6 trials (last 5 + best if different)
                assert len(snapshot_data["recent_trials"]) <= 6
                
                # Check if best trial is marked
                best_marked = any(t.get("is_best", False) for t in snapshot_data["recent_trials"])
                assert best_marked, "Best trial should be marked"
                
                print(f"✅ Snapshot contains {len(snapshot_data['recent_trials'])} trials (expected ≤ 6)")
                print(f"✅ Best trial properly marked: {best_marked}")
            else:
                print("❌ No snapshot file created")
                return False
                
        except Exception as e:
            print(f"❌ Progress logging failed: {e}")
            return False
        
        # Test final report
        try:
            _save_final_study_report(cfg, study)
            print("✅ Final report function executed without errors")
            
            # Check if final report was created
            final_report_files = list(Path(temp_dir).glob("final_study_report_*.json"))
            if final_report_files:
                with open(final_report_files[0], 'r') as f:
                    final_data = json.load(f)
                
                # Check structure
                assert "all_trials" in final_data
                assert "summary" in final_data
                assert "best_trial" in final_data
                assert "study_type" in final_data
                
                # Should have all 12 trials
                assert len(final_data["all_trials"]) == 12
                
                print(f"✅ Final report contains all {len(final_data['all_trials'])} trials")
                print(f"✅ Study type detected: {final_data['study_type']}")
            else:
                print("❌ No final report file created")
                return False
                
        except Exception as e:
            print(f"❌ Final report failed: {e}")
            return False
    
    print("🎉 All logging optimization tests passed!")
    return True

if __name__ == "__main__":
    success = test_optimized_logging()
    sys.exit(0 if success else 1)
