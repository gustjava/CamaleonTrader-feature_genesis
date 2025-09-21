#!/usr/bin/env python3
"""
Quick test to verify target selection and denylist fixes.
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_target_preservation():
    """Test that target is preserved correctly during denylist filtering."""
    print("🧪 Testing target preservation during denylist filtering...")
    
    # Create mock data with forward return targets
    np.random.seed(42)
    n_samples = 1000
    
    # Create features (some that should be blocked)
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'm1_prediction': np.random.randn(n_samples),  # Should be blocked
        'm5_signal': np.random.randn(n_samples),      # Should be blocked
        'y_ret_fwd_1m': np.random.randn(n_samples),   # Target candidate
        'y_ret_fwd_5m': np.random.randn(n_samples),   # Target candidate
        'y_ret_fwd_20m': np.random.randn(n_samples),  # Target candidate
        'is_bullish': np.random.choice([0, 1], n_samples),  # Should be blocked
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save to parquet
        df.to_parquet(temp_path)
        
        # Mock configuration
        dataset_cfg = {
            'path': temp_path,
            'format': 'parquet',
            'target': 'y_ret_fwd_20m',  # This should work now
            'target_candidates': ['y_ret_fwd_1m', 'y_ret_fwd_5m', 'y_ret_fwd_20m']
        }
        
        # Import the function
        try:
            from orchestration.objectives import _load_dataset
            print("✅ Successfully imported _load_dataset function")
        except ImportError as e:
            print(f"❌ Failed to import function: {e}")
            return False
        
        # Test the function
        try:
            X, y = _load_dataset(dataset_cfg)
            print(f"✅ Dataset loaded successfully: X shape {X.shape}, y shape {y.shape}")
            
            # Check that problematic features were removed
            blocked_features = [col for col in X.columns if 
                               col.startswith('m1_') or col.startswith('m5_') or 
                               col.startswith('y_ret_fwd_') or col.startswith('is_')]
            
            if blocked_features:
                print(f"❌ Found {len(blocked_features)} blocked features in X: {blocked_features}")
                return False
            else:
                print("✅ All problematic features correctly removed from X")
            
            # Check that we have the target
            if len(y) == len(df):
                print("✅ Target y has correct length")
            else:
                print(f"❌ Target y has wrong length: {len(y)} vs {len(df)}")
                return False
            
            # Check that target values match original
            original_target = df['y_ret_fwd_20m'].values
            if np.allclose(y.values, original_target, equal_nan=True):
                print("✅ Target values correctly preserved")
            else:
                print("❌ Target values do not match original")
                return False
            
            # Check remaining features
            expected_features = ['feature_1', 'feature_2']
            actual_features = list(X.columns)
            if set(actual_features) == set(expected_features):
                print(f"✅ Correct features remaining: {actual_features}")
            else:
                print(f"❌ Unexpected features: expected {expected_features}, got {actual_features}")
                return False
                
        except Exception as e:
            print(f"❌ Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("🎉 All target preservation tests passed!")
    return True

if __name__ == "__main__":
    success = test_target_preservation()
    sys.exit(0 if success else 1)
