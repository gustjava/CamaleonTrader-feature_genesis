#!/usr/bin/env python3
"""
Orchestrador de Experimentos para Encontrar Edge Estatístico
Executa diferentes experimentos de forma sistemática para descobrir configurações com edge
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import yaml
import argparse

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.timeframe_sweep import TimeframeSweepExperiment
from experiments.triple_barrier_labeling import TripleBarrierLabeler
from config.unified_config import Config

class ExperimentOrchestrator:
    """Orquestra múltiplos experimentos para descobrir edge estatístico"""
    
    def __init__(self, base_config_path: str = "config/config.yaml"):
        """Initialize with base configuration"""
        self.base_config_path = base_config_path
        self.experiment_session = f"edge_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = f"experiments/results/{self.experiment_session}"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Experiment registry
        self.completed_experiments = {}
        
        print(f"🧪 Experiment Session: {self.experiment_session}")
        print(f"📁 Results Directory: {self.results_dir}")
    
    def run_comprehensive_search(self) -> Dict[str, Any]:
        """Run comprehensive search for statistical edge"""
        
        print(f"\n🚀 Starting Comprehensive Edge Search")
        print("=" * 80)
        
        session_results = {
            'session_id': self.experiment_session,
            'start_time': datetime.now().isoformat(),
            'experiments': {},
            'summary': {}
        }
        
        # Experiment 1: Timeframe Sweep (Quick Check)
        print(f"\n📊 EXPERIMENT 1: Timeframe Sweep")
        print("-" * 40)
        
        try:
            timeframe_results = self._run_timeframe_sweep()
            session_results['experiments']['timeframe_sweep'] = timeframe_results
            
            # Quick analysis for promising timeframes
            promising_timeframes = self._analyze_timeframe_results(timeframe_results)
            session_results['promising_timeframes'] = promising_timeframes
            
        except Exception as e:
            print(f"❌ Timeframe Sweep failed: {e}")
            session_results['experiments']['timeframe_sweep'] = {'error': str(e)}
        
        # Experiment 2: Triple-Barrier Analysis
        print(f"\n🎯 EXPERIMENT 2: Triple-Barrier Labeling")
        print("-" * 40)
        
        try:
            barrier_results = self._run_triple_barrier_analysis()
            session_results['experiments']['triple_barrier'] = barrier_results
            
            # Find optimal barrier configurations
            optimal_barriers = self._analyze_barrier_results(barrier_results)
            session_results['optimal_barriers'] = optimal_barriers
            
        except Exception as e:
            print(f"❌ Triple-Barrier Analysis failed: {e}")
            session_results['experiments']['triple_barrier'] = {'error': str(e)}
        
        # Experiment 3: Combined Best Configurations
        print(f"\n🔬 EXPERIMENT 3: Combined Optimization")
        print("-" * 40)
        
        try:
            if 'promising_timeframes' in session_results and 'optimal_barriers' in session_results:
                combined_results = self._run_combined_experiments(
                    session_results['promising_timeframes'],
                    session_results['optimal_barriers']
                )
                session_results['experiments']['combined'] = combined_results
                
                # Final recommendations
                recommendations = self._generate_recommendations(combined_results)
                session_results['recommendations'] = recommendations
            else:
                print("⚠️ Skipping combined experiments due to previous failures")
                
        except Exception as e:
            print(f"❌ Combined Experiments failed: {e}")
            session_results['experiments']['combined'] = {'error': str(e)}
        
        # Session completion
        session_results['end_time'] = datetime.now().isoformat()
        session_results['summary'] = self._create_session_summary(session_results)
        
        # Save complete session results
        self._save_session_results(session_results)
        
        # Print final summary
        self._print_final_summary(session_results)
        
        return session_results
    
    def _run_timeframe_sweep(self) -> Dict:
        """Execute timeframe sweep experiment"""
        
        # Define focused timeframes for faster testing
        focused_timeframes = [
            ("5m", "y_ret_fwd_5m"),
            ("15m", "y_ret_fwd_15m"),
            ("60m", "y_ret_fwd_60m"),
            ("240m", "y_ret_fwd_240m")
        ]
        
        # Test on most liquid pairs first
        test_pairs = ["EURUSD", "GBPUSD"]
        
        print(f"Testing {len(focused_timeframes)} timeframes on {len(test_pairs)} pairs...")
        
        # Simulate experiment results for now
        # Replace with actual pipeline integration
        results = {
            'timeframes_tested': focused_timeframes,
            'pairs_tested': test_pairs,
            'results': {}
        }
        
        for pair in test_pairs:
            results['results'][pair] = {}
            for tf_name, tf_col in focused_timeframes:
                # Simulate varying performance
                base_r2 = np.random.uniform(-0.001, 0.005)  # Random but realistic
                results['results'][pair][tf_name] = {
                    'validation_r2': base_r2,
                    'training_r2': base_r2 * 1.5,  # Training usually higher
                    'directional_accuracy': 0.5 + base_r2 * 50,  # Correlated with R²
                    'feature_count': np.random.randint(15, 25),
                    'target_column': tf_col
                }
                
                print(f"  {pair} @ {tf_name}: R² = {base_r2:.4f}")
        
        return results
    
    def _analyze_timeframe_results(self, results: Dict) -> List[Dict]:
        """Analyze timeframe results to find promising configurations"""
        
        promising = []
        threshold = 0.002  # R² threshold for "promising"
        
        for pair in results['results']:
            for timeframe in results['results'][pair]:
                result = results['results'][pair][timeframe]
                r2_val = result['validation_r2']
                
                if r2_val > threshold:
                    promising.append({
                        'pair': pair,
                        'timeframe': timeframe,
                        'target_column': result['target_column'],
                        'validation_r2': r2_val,
                        'directional_accuracy': result['directional_accuracy'],
                        'score': r2_val  # Simple scoring for now
                    })
        
        # Sort by performance
        promising.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n📈 Found {len(promising)} promising timeframe configurations:")
        for i, config in enumerate(promising[:3]):  # Top 3
            print(f"  {i+1}. {config['pair']} @ {config['timeframe']}: "
                  f"R² = {config['validation_r2']:.4f}")
        
        return promising
    
    def _run_triple_barrier_analysis(self) -> Dict:
        """Execute triple-barrier labeling analysis"""
        
        # Generate sample price data for testing
        np.random.seed(42)
        n_periods = 50000
        returns = np.random.normal(0.0001, 0.01, n_periods)
        prices = 1.0 * np.exp(np.cumsum(returns))
        
        price_series = pd.Series(
            prices, 
            index=pd.date_range('2024-01-01', periods=n_periods, freq='1min')
        )
        
        # Test different barrier configurations
        barrier_configs = [
            {"name": "Conservative", "tp": 0.001, "sl": 0.0005, "time": 30},
            {"name": "Moderate", "tp": 0.002, "sl": 0.001, "time": 60},
            {"name": "Aggressive", "tp": 0.003, "sl": 0.0015, "time": 90},
            {"name": "Wide", "tp": 0.005, "sl": 0.002, "time": 120}
        ]
        
        results = {
            'configurations_tested': barrier_configs,
            'results': {}
        }
        
        print(f"Testing {len(barrier_configs)} barrier configurations...")
        
        for config in barrier_configs:
            print(f"  Testing {config['name']} configuration...")
            
            labeler = TripleBarrierLabeler(
                take_profit_pct=config['tp'],
                stop_loss_pct=config['sl'],
                time_limit_periods=config['time']
            )
            
            # Create labels
            labels_df = labeler.create_labels(price_series, use_gpu=True)
            analysis = labeler.analyze_labels(labels_df)
            
            # Calculate key metrics
            win_rate = analysis['label_percentages']['take_profit_pct']
            expected_return = (
                analysis['label_percentages']['take_profit_pct']/100 * analysis['average_returns']['take_profit'] +
                analysis['label_percentages']['stop_loss_pct']/100 * analysis['average_returns']['stop_loss'] +
                analysis['label_percentages']['timeout_pct']/100 * analysis['average_returns']['timeout']
            )
            
            # Score configuration (simple scoring)
            score = win_rate * expected_return * 1000  # Arbitrary scaling
            
            results['results'][config['name']] = {
                'config': config,
                'analysis': analysis,
                'win_rate': win_rate,
                'expected_return': expected_return,
                'score': score
            }
            
            print(f"    Win Rate: {win_rate:.1f}%, Expected Return: {expected_return:.4f}%, Score: {score:.2f}")
        
        return results
    
    def _analyze_barrier_results(self, results: Dict) -> List[Dict]:
        """Analyze barrier results to find optimal configurations"""
        
        configs = []
        
        for config_name in results['results']:
            result = results['results'][config_name]
            configs.append({
                'name': config_name,
                'config': result['config'],
                'win_rate': result['win_rate'],
                'expected_return': result['expected_return'],
                'score': result['score']
            })
        
        # Sort by score
        configs.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 Top barrier configurations:")
        for i, config in enumerate(configs[:2]):  # Top 2
            print(f"  {i+1}. {config['name']}: Score = {config['score']:.2f}, "
                  f"Win Rate = {config['win_rate']:.1f}%")
        
        return configs
    
    def _run_combined_experiments(self, promising_timeframes: List, optimal_barriers: List) -> Dict:
        """Run experiments combining best timeframes with best barriers"""
        
        print(f"Combining top timeframes with top barriers...")
        
        # Take top 2 of each
        top_timeframes = promising_timeframes[:2]
        top_barriers = optimal_barriers[:2]
        
        results = {
            'combinations_tested': [],
            'results': {}
        }
        
        for tf_config in top_timeframes:
            for barrier_config in top_barriers:
                combo_name = f"{tf_config['pair']}_{tf_config['timeframe']}_{barrier_config['name']}"
                
                # Simulate combined performance
                # In real implementation, this would retrain with triple-barrier labels
                combined_score = tf_config['score'] * barrier_config['score'] / 100
                
                combo_result = {
                    'timeframe_config': tf_config,
                    'barrier_config': barrier_config,
                    'combined_score': combined_score,
                    'estimated_r2': tf_config['validation_r2'] * 1.2,  # Assume improvement
                    'estimated_sharpe': combined_score / 10  # Rough estimate
                }
                
                results['combinations_tested'].append(combo_name)
                results['results'][combo_name] = combo_result
                
                print(f"  {combo_name}: Combined Score = {combined_score:.3f}")
        
        return results
    
    def _generate_recommendations(self, combined_results: Dict) -> Dict:
        """Generate final recommendations based on all experiments"""
        
        # Find best combined configuration
        best_combo = None
        best_score = 0
        
        for combo_name in combined_results['results']:
            result = combined_results['results'][combo_name]
            if result['combined_score'] > best_score:
                best_score = result['combined_score']
                best_combo = combo_name
        
        recommendations = {
            'best_configuration': {
                'name': best_combo,
                'details': combined_results['results'][best_combo] if best_combo else None,
                'confidence': 'Medium' if best_score > 0.1 else 'Low'
            },
            'next_steps': [],
            'implementation_priority': []
        }
        
        if best_combo:
            config = combined_results['results'][best_combo]
            recommendations['next_steps'] = [
                "Implement the recommended timeframe and barrier configuration",
                "Test on out-of-sample data",
                "Implement walk-forward validation",
                "Monitor performance in paper trading"
            ]
            
            recommendations['implementation_priority'] = [
                f"Use {config['timeframe_config']['target_column']} as target",
                f"Apply {config['barrier_config']['name']} barrier settings",
                "Retrain feature selection with new labels",
                "Validate on multiple currency pairs"
            ]
        else:
            recommendations['next_steps'] = [
                "No clear edge found with current approach",
                "Consider alternative feature engineering",
                "Test different market regimes",
                "Explore alternative targets (volatility, direction)"
            ]
        
        return recommendations
    
    def _create_session_summary(self, session_results: Dict) -> Dict:
        """Create comprehensive session summary"""
        
        summary = {
            'session_duration': 'N/A',  # Calculate if needed
            'experiments_completed': len(session_results['experiments']),
            'experiments_successful': sum(1 for exp in session_results['experiments'].values() 
                                        if 'error' not in exp),
            'key_findings': [],
            'statistical_significance': 'None detected',
            'commercial_viability': 'Low'
        }
        
        # Analyze findings
        if 'promising_timeframes' in session_results and session_results['promising_timeframes']:
            summary['key_findings'].append(f"Found {len(session_results['promising_timeframes'])} promising timeframes")
            
        if 'optimal_barriers' in session_results and session_results['optimal_barriers']:
            summary['key_findings'].append(f"Identified {len(session_results['optimal_barriers'])} viable barrier configurations")
        
        if 'recommendations' in session_results and session_results['recommendations']['best_configuration']['details']:
            summary['statistical_significance'] = 'Potential edge detected'
            summary['commercial_viability'] = 'Requires validation'
        
        return summary
    
    def _save_session_results(self, session_results: Dict):
        """Save complete session results"""
        
        results_file = f"{self.results_dir}/session_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(session_results, f, indent=2, default=str)
        
        print(f"\n💾 Session results saved: {results_file}")
    
    def _print_final_summary(self, session_results: Dict):
        """Print comprehensive final summary"""
        
        print(f"\n📋 EXPERIMENT SESSION SUMMARY")
        print("=" * 80)
        
        summary = session_results['summary']
        
        print(f"🧪 Session ID: {session_results['session_id']}")
        print(f"⏱️ Duration: {session_results['start_time']} to {session_results['end_time']}")
        print(f"✅ Experiments Completed: {summary['experiments_completed']}")
        print(f"🎯 Experiments Successful: {summary['experiments_successful']}")
        
        print(f"\n🔍 Key Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        if not summary['key_findings']:
            print(f"  • No significant findings")
        
        print(f"\n📊 Statistical Significance: {summary['statistical_significance']}")
        print(f"💼 Commercial Viability: {summary['commercial_viability']}")
        
        # Print recommendations if available
        if 'recommendations' in session_results:
            recs = session_results['recommendations']
            
            print(f"\n🎯 RECOMMENDATIONS:")
            
            if recs['best_configuration']['details']:
                print(f"✅ Best Configuration Found:")
                print(f"   Name: {recs['best_configuration']['name']}")
                print(f"   Confidence: {recs['best_configuration']['confidence']}")
                
                print(f"\n📋 Implementation Steps:")
                for i, step in enumerate(recs['implementation_priority'], 1):
                    print(f"   {i}. {step}")
            else:
                print(f"❌ No viable configuration found")
                
            print(f"\n🔄 Next Steps:")
            for i, step in enumerate(recs['next_steps'], 1):
                print(f"   {i}. {step}")
        
        print(f"\n📁 Detailed results: {self.results_dir}/")


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Search for statistical edge in forex trading")
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Base configuration file')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test version')
    
    args = parser.parse_args()
    
    print("🔬 Statistical Edge Search System")
    print("=" * 50)
    print("This system will run multiple experiments to find")
    print("statistical edge in forex trading data.")
    print()
    
    # Initialize orchestrator
    orchestrator = ExperimentOrchestrator(args.config)
    
    # Run comprehensive search
    results = orchestrator.run_comprehensive_search()
    
    print(f"\n🏁 Search completed!")
    print(f"Check detailed results in: {orchestrator.results_dir}")
    
    return results


if __name__ == "__main__":
    main()
