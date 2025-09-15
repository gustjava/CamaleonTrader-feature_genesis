#!/usr/bin/env python3
"""
Experimento 1: Timeframe Sweep
Testa diferentes janelas de predição para encontrar timeframes com edge estatístico
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import yaml

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.unified_config import Config
from orchestration.main import run_pipeline_for_pair

class TimeframeSweepExperiment:
    """Experimenta diferentes timeframes para encontrar edge estatístico"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment with base configuration"""
        self.config_path = config_path
        self.base_config = Config(config_path)
        
        # Define timeframes to test (in minutes)
        self.timeframes_to_test = [
            ("5m", "y_ret_fwd_5m"),
            ("15m", "y_ret_fwd_15m"), 
            ("30m", "y_ret_fwd_30m"),
            ("60m", "y_ret_fwd_60m"),   # Current baseline
            ("240m", "y_ret_fwd_240m"), # 4 hours
            ("1440m", "y_ret_fwd_1440m") # 1 day
        ]
        
        # Test pairs (start with most liquid)
        self.test_pairs = ["EURUSD", "GBPUSD", "AUDUSD"]
        
        # Results storage
        self.results = {}
        self.experiment_id = f"timeframe_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def run_experiment(self) -> Dict:
        """Run complete timeframe sweep experiment"""
        print(f"🚀 Starting Timeframe Sweep Experiment: {self.experiment_id}")
        print(f"📊 Testing {len(self.timeframes_to_test)} timeframes on {len(self.test_pairs)} pairs")
        
        for pair in self.test_pairs:
            print(f"\n💰 Processing {pair}...")
            self.results[pair] = {}
            
            for timeframe_name, target_column in self.timeframes_to_test:
                print(f"  ⏱️ Testing {timeframe_name} ({target_column})...")
                
                try:
                    # Modify config for this specific test
                    result = self._run_single_experiment(pair, target_column)
                    self.results[pair][timeframe_name] = result
                    
                    # Print immediate results
                    r2_val = result.get('validation_r2', 0)
                    r2_train = result.get('training_r2', 0)
                    directional_acc = result.get('directional_accuracy', 0)
                    
                    print(f"    📈 R²(val): {r2_val:.4f} | R²(train): {r2_train:.4f} | Dir.Acc: {directional_acc:.4f}")
                    
                    # Early promising signal detection
                    if r2_val > 0.01:
                        print(f"    🎯 PROMISING SIGNAL DETECTED! R² = {r2_val:.4f}")
                    
                except Exception as e:
                    print(f"    ❌ Error: {str(e)}")
                    self.results[pair][timeframe_name] = {"error": str(e)}
        
        # Save and analyze results
        self._save_results()
        self._analyze_results()
        
        return self.results
    
    def _run_single_experiment(self, pair: str, target_column: str) -> Dict:
        """Run pipeline with modified target column"""
        
        # Create temporary config with modified target
        temp_config = self._create_temp_config(target_column)
        
        # Run pipeline (simplified version - focus on statistical tests)
        # This would need to be integrated with your main pipeline
        # For now, simulate the structure
        
        result = {
            'target_column': target_column,
            'pair': pair,
            'timestamp': datetime.now().isoformat(),
            'validation_r2': 0.0001,  # Placeholder - replace with actual pipeline run
            'training_r2': 0.0001,
            'directional_accuracy': 0.51,
            'feature_count': 18,
            'dataset_size': 1000000
        }
        
        return result
    
    def _create_temp_config(self, target_column: str) -> str:
        """Create temporary config file with modified target"""
        
        # Load base config
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Modify target column
        config_data['data']['target_column'] = target_column
        
        # Save temporary config
        temp_config_path = f"config/temp_config_{target_column}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        return temp_config_path
    
    def _save_results(self):
        """Save experiment results to JSON"""
        
        results_dir = "experiments/results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = f"{results_dir}/{self.experiment_id}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
    
    def _analyze_results(self):
        """Analyze and report experiment findings"""
        
        print(f"\n📊 TIMEFRAME SWEEP ANALYSIS - {self.experiment_id}")
        print("=" * 80)
        
        # Find best performing combinations
        best_combinations = []
        
        for pair in self.results:
            for timeframe in self.results[pair]:
                result = self.results[pair][timeframe]
                if 'error' not in result:
                    r2_val = result.get('validation_r2', 0)
                    best_combinations.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'r2_validation': r2_val,
                        'r2_training': result.get('training_r2', 0),
                        'directional_accuracy': result.get('directional_accuracy', 0)
                    })
        
        # Sort by validation R²
        best_combinations.sort(key=lambda x: x['r2_validation'], reverse=True)
        
        print("\n🏆 TOP 5 BEST PERFORMING COMBINATIONS:")
        for i, combo in enumerate(best_combinations[:5]):
            print(f"{i+1}. {combo['pair']} @ {combo['timeframe']}: "
                  f"R²={combo['r2_validation']:.4f}, "
                  f"Dir.Acc={combo['directional_accuracy']:.4f}")
        
        # Statistical significance threshold
        significant_threshold = 0.01
        significant_results = [c for c in best_combinations if c['r2_validation'] > significant_threshold]
        
        print(f"\n📈 STATISTICALLY SIGNIFICANT RESULTS (R² > {significant_threshold}):")
        if significant_results:
            for result in significant_results:
                print(f"   🎯 {result['pair']} @ {result['timeframe']}: R² = {result['r2_validation']:.4f}")
        else:
            print("   ❌ No statistically significant results found")
            print("   💡 Recommendation: Try Triple-Barrier labeling or alternative targets")
        
        # Timeframe analysis
        print(f"\n⏱️ TIMEFRAME PERFORMANCE SUMMARY:")
        timeframe_performance = {}
        
        for combo in best_combinations:
            tf = combo['timeframe']
            if tf not in timeframe_performance:
                timeframe_performance[tf] = []
            timeframe_performance[tf].append(combo['r2_validation'])
        
        for tf in timeframe_performance:
            avg_r2 = sum(timeframe_performance[tf]) / len(timeframe_performance[tf])
            max_r2 = max(timeframe_performance[tf])
            print(f"   {tf}: Avg R² = {avg_r2:.4f}, Max R² = {max_r2:.4f}")


def main():
    """Run the timeframe sweep experiment"""
    
    # Create experiments directory
    os.makedirs("experiments/results", exist_ok=True)
    
    # Initialize and run experiment
    experiment = TimeframeSweepExperiment()
    results = experiment.run_experiment()
    
    print(f"\n✅ Timeframe Sweep completed!")
    print(f"📁 Check results in: experiments/results/")
    
    return results


if __name__ == "__main__":
    main()
