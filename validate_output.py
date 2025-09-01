#!/usr/bin/env python3
"""
Validation script for processed feature files.
"""

import pandas as pd
import sys
from pathlib import Path

def validate_file(file_path):
    """Validate a processed feature file."""
    print(f"Validating: {file_path}")
    
    try:
        df = pd.read_feather(file_path)
        print(f"Shape: {df.shape}")
        
        # Check original features
        original_features = [col for col in df.columns if col.startswith('y_')]
        print(f"Original features (y_*): {len(original_features)}")
        
        # Check features by engine
        print("\n" + "="*60)
        print("FEATURE ENGINE VALIDATION")
        print("="*60)
        
        # 1. StationarizationEngine features
        stationarization_features = [col for col in df.columns if any(prefix in col for prefix in ['fracdiff', 'log_', 'diff_', 'rolling_'])]
        print(f"\n1. STATIONARIZATION ENGINE:")
        print(f"   Features found: {len(stationarization_features)}")
        if stationarization_features:
            print(f"   Feature names: {stationarization_features[:10]}{'...' if len(stationarization_features) > 10 else ''}")
            # Check non-NaN values
            for col in stationarization_features[:5]:  # Check first 5
                non_nan_count = df[col].notna().sum()
                print(f"   {col}: {non_nan_count}/{len(df)} ({non_nan_count/len(df)*100:.1f}%)")
        else:
            print("   ❌ NO STATIONARIZATION FEATURES FOUND!")
        
        # 2. SignalProcessor features (Baxter-King)
        signal_features = [col for col in df.columns if col.startswith('bk_filter')]
        print(f"\n2. SIGNAL PROCESSOR (Baxter-King):")
        print(f"   Features found: {len(signal_features)}")
        if signal_features:
            print(f"   Feature names: {signal_features}")
            for col in signal_features:
                non_nan_count = df[col].notna().sum()
                print(f"   {col}: {non_nan_count}/{len(df)} ({non_nan_count/len(df)*100:.1f}%)")
        else:
            print("   ❌ NO SIGNAL PROCESSING FEATURES FOUND!")
        
        # 3. StatisticalTests features
        statistical_features = [col for col in df.columns if any(prefix in col for prefix in ['adf_', 'dcor_', 'stationarity_', 'test_'])]
        print(f"\n3. STATISTICAL TESTS:")
        print(f"   Features found: {len(statistical_features)}")
        if statistical_features:
            print(f"   Feature names: {statistical_features[:10]}{'...' if len(statistical_features) > 10 else ''}")
            for col in statistical_features[:5]:  # Check first 5
                non_nan_count = df[col].notna().sum()
                print(f"   {col}: {non_nan_count}/{len(df)} ({non_nan_count/len(df)*100:.1f}%)")
        else:
            print("   ❌ NO STATISTICAL TEST FEATURES FOUND!")
        
        # 4. GARCH Models features
        garch_features = [col for col in df.columns if col.startswith('garch_')]
        print(f"\n4. GARCH MODELS:")
        print(f"   Features found: {len(garch_features)}")
        if garch_features:
            print(f"   Feature names: {garch_features}")
            for col in garch_features:
                non_nan_count = df[col].notna().sum()
                print(f"   {col}: {non_nan_count}/{len(df)} ({non_nan_count/len(df)*100:.1f}%)")
        else:
            print("   ❌ NO GARCH FEATURES FOUND!")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        total_new_features = len(stationarization_features) + len(signal_features) + len(statistical_features) + len(garch_features)
        print(f"Total new features created: {total_new_features}")
        print(f"Feature expansion: {len(original_features)} -> {len(df.columns)} (+{len(df.columns) - len(original_features)})")
        
        # Check if all engines contributed
        engines_working = []
        if len(stationarization_features) > 0:
            engines_working.append("Stationarization")
        if len(signal_features) > 0:
            engines_working.append("Signal Processing")
        if len(statistical_features) > 0:
            engines_working.append("Statistical Tests")
        if len(garch_features) > 0:
            engines_working.append("GARCH Models")
        
        print(f"Engines working: {engines_working}")
        print(f"Engines expected: 4")
        
        if len(engines_working) == 4:
            print("✅ ALL ENGINES WORKING CORRECTLY!")
        else:
            print(f"❌ ONLY {len(engines_working)}/4 ENGINES WORKING!")
        
        return True
        
    except Exception as e:
        print(f"Error validating {file_path}: {e}")
        return False

def main():
    """Main validation function."""
    base_path = Path("/workspace/feature_genesis/data/processed_features")
    
    if not base_path.exists():
        print(f"Base path does not exist: {base_path}")
        return 1
    
    # Find all feather files
    feather_files = list(base_path.rglob("*.feather"))
    
    if not feather_files:
        print("No feather files found!")
        return 1
    
    print(f"Found {len(feather_files)} feather files")
    
    # Validate first few files
    for i, file_path in enumerate(feather_files[:3]):
        print(f"\n{'='*60}")
        print(f"FILE {i+1}/{min(3, len(feather_files))}")
        print(f"{'='*60}")
        validate_file(file_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
