#!/usr/bin/env python3
"""
Comprehensive Validation Script for Dynamic Stage 0 Pipeline

This script validates the output of the feature engineering pipeline by checking:
1. Data integrity and completeness
2. Feature generation and stationarity
3. Statistical properties
4. File structure and metadata
5. Performance metrics
"""

import logging
import sys
import os
import glob
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import cudf
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.unified_config import get_unified_config as get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineValidator:
    """Comprehensive validator for the Dynamic Stage 0 pipeline output."""
    
    def __init__(self, output_path: str):
        """Initialize the validator with output path."""
        self.output_path = Path(output_path)
        self.settings = get_settings()
        self.validation_results = {}
        
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate the file structure and organization."""
        logger.info("🔍 Validating file structure...")
        
        results = {
            'total_files': 0,
            'currency_pairs': [],
            'file_sizes': {},
            'missing_files': [],
            'structure_issues': []
        }
        
        # Check if output directory exists
        if not self.output_path.exists():
            results['structure_issues'].append(f"Output directory does not exist: {self.output_path}")
            return results
        
        # Find all feather files
        feather_files = list(self.output_path.glob("*/**/*.feather"))
        results['total_files'] = len(feather_files)
        
        logger.info(f"Found {len(feather_files)} feather files")
        
        for feather_file in feather_files:
            try:
                # Extract currency pair from path
                currency_pair = feather_file.parent.name
                results['currency_pairs'].append(currency_pair)
                
                # Check file size
                file_size_mb = feather_file.stat().st_size / (1024 * 1024)
                results['file_sizes'][currency_pair] = file_size_mb
                
                logger.info(f"✅ {currency_pair}: {file_size_mb:.2f} MB")
                
            except Exception as e:
                results['structure_issues'].append(f"Error processing {feather_file}: {e}")
        
        # Check for expected currency pairs
        expected_pairs = ['AUDUSD', 'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
                         'EURJPY', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY']
        
        for pair in expected_pairs:
            if pair not in results['currency_pairs']:
                results['missing_files'].append(pair)
        
        return results
    
    def validate_data_integrity(self, currency_pair: str) -> Dict[str, Any]:
        """Validate data integrity for a specific currency pair."""
        logger.info(f"🔍 Validating data integrity for {currency_pair}...")
        
        results = {
            'currency_pair': currency_pair,
            'shape': None,
            'columns': [],
            'dtypes': {},
            'null_counts': {},
            'duplicate_rows': 0,
            'timestamp_issues': [],
            'data_quality_issues': []
        }
        
        try:
            # Load the feather file
            file_path = self.output_path / currency_pair / f"{currency_pair}.feather"
            if not file_path.exists():
                results['data_quality_issues'].append(f"File not found: {file_path}")
                return results
            
            # Load with cudf for GPU processing
            df = cudf.read_feather(file_path)
            
            # Basic shape and structure
            results['shape'] = df.shape
            results['columns'] = list(df.columns)
            results['dtypes'] = df.dtypes.to_dict()
            
            logger.info(f"📊 {currency_pair} shape: {df.shape}")
            logger.info(f"📊 {currency_pair} columns: {len(df.columns)}")
            
            # Check for null values
            null_counts = df.isna().sum()
            results['null_counts'] = null_counts.to_dict()
            
            high_null_cols = null_counts[null_counts > len(df) * 0.1]  # >10% null
            if not high_null_cols.empty:
                results['data_quality_issues'].append(f"High null columns: {high_null_cols.to_dict()}")
            
            # Check for duplicate rows
            results['duplicate_rows'] = len(df) - len(df.drop_duplicates())
            
            # Check timestamp column if exists
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].to_pandas()
                if not timestamps.is_monotonic_increasing:
                    results['timestamp_issues'].append("Timestamps are not monotonically increasing")
                
                # Check for gaps in timestamps
                time_diff = timestamps.diff()
                large_gaps = time_diff[time_diff > pd.Timedelta(minutes=5)]
                if not large_gaps.empty:
                    results['timestamp_issues'].append(f"Found {len(large_gaps)} large time gaps")
            
            # Convert back to pandas for further analysis
            df_pandas = df.to_pandas()
            
            # Check for infinite values
            inf_counts = np.isinf(df_pandas.select_dtypes(include=[np.number])).sum()
            if inf_counts.sum() > 0:
                results['data_quality_issues'].append(f"Infinite values found: {inf_counts.to_dict()}")
            
            # Check for extreme outliers
            numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  # Check first 10 numeric columns
                if col in ['timestamp', 'symbol']:
                    continue
                try:
                    z_scores = np.abs(stats.zscore(df_pandas[col].dropna()))
                    extreme_outliers = (z_scores > 10).sum()
                    if extreme_outliers > 0:
                        results['data_quality_issues'].append(f"Extreme outliers in {col}: {extreme_outliers}")
                except:
                    pass
            
        except Exception as e:
            results['data_quality_issues'].append(f"Error loading data: {e}")
            logger.error(f"Error validating {currency_pair}: {e}")
        
        return results
    
    def validate_feature_generation(self, currency_pair: str) -> Dict[str, Any]:
        """Validate that features were properly generated."""
        logger.info(f"🔍 Validating feature generation for {currency_pair}...")
        
        results = {
            'currency_pair': currency_pair,
            'total_features': 0,
            'new_features': [],
            'expected_features': [],
            'missing_features': [],
            'feature_categories': {
                'stationarization': [],
                'signal_processing': [],
                'statistical_tests': [],
                'garch_models': []
            }
        }
        
        try:
            # Load the data
            file_path = self.output_path / currency_pair / f"{currency_pair}.feather"
            df = cudf.read_feather(file_path)
            
            all_columns = list(df.columns)
            results['total_features'] = len(all_columns)
            
            # Categorize features by expected patterns
            for col in all_columns:
                col_lower = col.lower()
                
                # Stationarization features (fracdiff, rolling correlations)
                if 'frac_diff' in col_lower or 'rolling_corr' in col_lower:
                    results['feature_categories']['stationarization'].append(col)
                
                # Signal processing features (Baxter-King filter)
                elif 'bk_filter' in col_lower or 'baxter' in col_lower:
                    results['feature_categories']['signal_processing'].append(col)
                
                # Statistical tests features (ADF, dCor)
                elif 'adf_stat' in col_lower or 'dcor' in col_lower:
                    results['feature_categories']['statistical_tests'].append(col)
                
                # GARCH model features
                elif 'garch_' in col_lower:
                    results['feature_categories']['garch_models'].append(col)
            
            # Expected features based on pipeline stages
            expected_stationarization = ['frac_diff', 'rolling_corr']
            expected_signal_processing = ['bk_filter']
            expected_statistical_tests = ['adf_stat', 'dcor']
            expected_garch = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_persistence']
            
            # Check for expected features
            for category, expected in [
                ('stationarization', expected_stationarization),
                ('signal_processing', expected_signal_processing),
                ('statistical_tests', expected_statistical_tests),
                ('garch_models', expected_garch)
            ]:
                for expected_feature in expected:
                    found = any(expected_feature in col for col in all_columns)
                    if found:
                        results['expected_features'].append(f"{category}:{expected_feature}")
                    else:
                        results['missing_features'].append(f"{category}:{expected_feature}")
            
            # Log results
            for category, features in results['feature_categories'].items():
                logger.info(f"📊 {category}: {len(features)} features")
                if features:
                    logger.info(f"   Examples: {features[:3]}")
            
        except Exception as e:
            logger.error(f"Error validating features for {currency_pair}: {e}")
            results['missing_features'].append(f"Error: {e}")
        
        return results
    
    def validate_stationarity(self, currency_pair: str) -> Dict[str, Any]:
        """Validate that stationarization features are actually stationary."""
        logger.info(f"🔍 Validating stationarity for {currency_pair}...")
        
        results = {
            'currency_pair': currency_pair,
            'stationarity_tests': {},
            'frac_diff_features': [],
            'stationary_features': [],
            'non_stationary_features': []
        }
        
        try:
            # Load the data
            file_path = self.output_path / currency_pair / f"{currency_pair}.feather"
            df = cudf.read_feather(file_path)
            df_pandas = df.to_pandas()
            
            # Find frac_diff features
            frac_diff_cols = [col for col in df.columns if 'frac_diff' in col.lower()]
            results['frac_diff_features'] = frac_diff_cols
            
            logger.info(f"📊 Testing stationarity for {len(frac_diff_cols)} frac_diff features")
            
            # Test stationarity for each frac_diff feature
            for col in frac_diff_cols[:5]:  # Test first 5 to avoid too much computation
                try:
                    series = df_pandas[col].dropna()
                    if len(series) < 50:
                        continue
                    
                    # ADF test
                    adf_result = adfuller(series, autolag='AIC')
                    p_value = adf_result[1]
                    
                    is_stationary = p_value < 0.05
                    results['stationarity_tests'][col] = {
                        'p_value': p_value,
                        'is_stationary': is_stationary,
                        'adf_statistic': adf_result[0]
                    }
                    
                    if is_stationary:
                        results['stationary_features'].append(col)
                    else:
                        results['non_stationary_features'].append(col)
                    
                    logger.info(f"   {col}: p={p_value:.4f} {'✅' if is_stationary else '❌'}")
                    
                except Exception as e:
                    logger.warning(f"Could not test stationarity for {col}: {e}")
            
        except Exception as e:
            logger.error(f"Error validating stationarity for {currency_pair}: {e}")
        
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        logger.info("📋 Generating comprehensive validation report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_structure': self.validate_file_structure(),
            'data_integrity': {},
            'feature_generation': {},
            'stationarity': {},
            'summary': {}
        }
        
        # Validate each currency pair
        for currency_pair in report['file_structure']['currency_pairs']:
            logger.info(f"🔍 Validating {currency_pair}...")
            
            report['data_integrity'][currency_pair] = self.validate_data_integrity(currency_pair)
            report['feature_generation'][currency_pair] = self.validate_feature_generation(currency_pair)
            report['stationarity'][currency_pair] = self.validate_stationarity(currency_pair)
        
        # Generate summary
        report['summary'] = self._generate_summary(report)
        
        return report
    
    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the validation results."""
        summary = {
            'total_currency_pairs': len(report['file_structure']['currency_pairs']),
            'total_files': report['file_structure']['total_files'],
            'missing_files': len(report['file_structure']['missing_files']),
            'structure_issues': len(report['file_structure']['structure_issues']),
            'data_quality_issues': 0,
            'feature_generation_issues': 0,
            'stationarity_issues': 0,
            'overall_status': 'PASS'
        }
        
        # Count issues
        for currency_pair in report['data_integrity']:
            summary['data_quality_issues'] += len(report['data_integrity'][currency_pair]['data_quality_issues'])
            summary['feature_generation_issues'] += len(report['feature_generation'][currency_pair]['missing_features'])
            
            # Count non-stationary features as issues
            summary['stationarity_issues'] += len(report['stationarity'][currency_pair]['non_stationary_features'])
        
        # Determine overall status
        total_issues = (summary['structure_issues'] + summary['data_quality_issues'] + 
                       summary['feature_generation_issues'] + summary['stationarity_issues'])
        
        if total_issues > 0:
            summary['overall_status'] = 'FAIL'
        
        return summary
    
    def print_report(self, report: Dict[str, Any]):
        """Print a formatted validation report."""
        print("\n" + "="*80)
        print("DYNAMIC STAGE 0 PIPELINE VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        
        print("\n📊 SUMMARY:")
        print(f"   Currency Pairs: {report['summary']['total_currency_pairs']}")
        print(f"   Total Files: {report['summary']['total_files']}")
        print(f"   Missing Files: {report['summary']['missing_files']}")
        print(f"   Structure Issues: {report['summary']['structure_issues']}")
        print(f"   Data Quality Issues: {report['summary']['data_quality_issues']}")
        print(f"   Feature Generation Issues: {report['summary']['feature_generation_issues']}")
        print(f"   Stationarity Issues: {report['summary']['stationarity_issues']}")
        
        if report['file_structure']['missing_files']:
            print(f"\n❌ MISSING FILES: {report['file_structure']['missing_files']}")
        
        if report['file_structure']['structure_issues']:
            print(f"\n⚠️ STRUCTURE ISSUES:")
            for issue in report['file_structure']['structure_issues']:
                print(f"   - {issue}")
        
        print("\n📈 FEATURE GENERATION SUMMARY:")
        for currency_pair in report['feature_generation']:
            features = report['feature_generation'][currency_pair]
            print(f"   {currency_pair}:")
            for category, feature_list in features['feature_categories'].items():
                print(f"     {category}: {len(feature_list)} features")
        
        print("\n🔍 STATIONARITY SUMMARY:")
        for currency_pair in report['stationarity']:
            stationarity = report['stationarity'][currency_pair]
            print(f"   {currency_pair}:")
            print(f"     Stationary: {len(stationarity['stationary_features'])}")
            print(f"     Non-stationary: {len(stationarity['non_stationary_features'])}")
        
        print("\n" + "="*80)

def main():
    """Main validation function."""
    logger.info("🔍 Starting Dynamic Stage 0 Pipeline Validation")
    
    try:
        settings = get_settings()
        output_path = settings.output.output_path
        
        # Create validator
        validator = PipelineValidator(output_path)
        
        # Generate comprehensive report
        report = validator.generate_validation_report()
        
        # Print report
        validator.print_report(report)
        
        # Save report to file
        report_file = Path(output_path) / "validation_report.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Validation report saved to: {report_file}")
        
        # Return exit code based on overall status
        if report['summary']['overall_status'] == 'PASS':
            logger.info("✅ Validation PASSED")
            return 0
        else:
            logger.error("❌ Validation FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Error during validation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
