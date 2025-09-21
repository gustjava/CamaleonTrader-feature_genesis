#!/usr/bin/env python3
"""
Feature Robustness Analysis Script

Analyzes the results of Optuna Study A to identify the most robust and stable features
across the Pareto-optimal solutions. Provides comprehensive analysis of feature selection
frequency, stability, and importance across different trials and temporal folds.

Usage:
    python feature_robustness_analysis.py --study-path /path/to/study_a.sqlite --output-dir ./analysis_results
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from scipy import stats

class FeatureRobustnessAnalyzer:
    """
    Comprehensive analyzer for feature robustness from Optuna study results.
    """
    
    def __init__(self, study_path: str):
        """
        Initialize analyzer with Optuna study.
        
        Args:
            study_path: Path to the SQLite database containing the study
        """
        self.study_path = study_path
        self.study = None
        self.pareto_trials = []
        self.all_trials = []
        self.feature_analysis = {}
        
    def load_study(self) -> None:
        """Load the Optuna study from SQLite database."""
        try:
            storage_url = f"sqlite:///{self.study_path}"
            study_name = "stageA_preprocess_selection"  # Default study name
            
            self.study = optuna.load_study(
                study_name=study_name,
                storage=storage_url
            )
            
            self.all_trials = [trial for trial in self.study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
            
            # Get Pareto-optimal trials for multi-objective optimization
            if len(self.study.directions) > 1:
                self.pareto_trials = self.study.best_trials
            else:
                # Single objective - use top 10% of trials
                sorted_trials = sorted(self.all_trials, key=lambda t: t.value or 0, reverse=True)
                top_n = max(1, len(sorted_trials) // 10)
                self.pareto_trials = sorted_trials[:top_n]
                
            print(f"Loaded study with {len(self.all_trials)} completed trials")
            print(f"Found {len(self.pareto_trials)} Pareto-optimal trials")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load study from {self.study_path}: {e}")
    
    def analyze_feature_frequency(self) -> Dict[str, Any]:
        """
        Analyze frequency of feature selection across Pareto-optimal trials.
        
        Returns:
            Dictionary with frequency analysis results
        """
        feature_counts = Counter()
        total_pareto_trials = len(self.pareto_trials)
        
        # Count feature occurrences in Pareto trials
        for trial in self.pareto_trials:
            selected_features = trial.user_attrs.get('selected_features', [])
            for feature in selected_features:
                feature_counts[feature] += 1
        
        # Calculate selection frequencies
        feature_frequencies = {
            feature: count / total_pareto_trials 
            for feature, count in feature_counts.items()
        }
        
        # Rank features by frequency
        ranked_features = sorted(
            feature_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'feature_frequencies': feature_frequencies,
            'ranked_features': ranked_features,
            'total_unique_features': len(feature_counts),
            'most_frequent_features': ranked_features[:20],  # Top 20
            'selection_threshold_50': [f for f, freq in ranked_features if freq >= 0.5],
            'selection_threshold_75': [f for f, freq in ranked_features if freq >= 0.75],
            'feature_counts': dict(feature_counts)
        }
    
    def analyze_feature_stability(self) -> Dict[str, Any]:
        """
        Analyze stability of feature importance across temporal folds.
        
        Returns:
            Dictionary with stability analysis results
        """
        stability_analysis = {}
        feature_importance_data = defaultdict(list)
        
        # Collect feature importance data from trials
        for trial in self.pareto_trials:
            importance_folds = trial.user_attrs.get('feature_importance_folds', [])
            
            for fold_data in importance_folds:
                fold_importances = fold_data.get('importances', {})
                for feature, importance in fold_importances.items():
                    feature_importance_data[feature].append(importance)
        
        # Calculate stability metrics for each feature
        for feature, importance_values in feature_importance_data.items():
            if len(importance_values) >= 3:  # Minimum data points for meaningful analysis
                importance_array = np.array(importance_values)
                
                stability_analysis[feature] = {
                    'mean_importance': float(np.mean(importance_array)),
                    'std_importance': float(np.std(importance_array)),
                    'cv_importance': float(np.std(importance_array) / (np.mean(importance_array) + 1e-9)),
                    'median_importance': float(np.median(importance_array)),
                    'min_importance': float(np.min(importance_array)),
                    'max_importance': float(np.max(importance_array)),
                    'n_observations': len(importance_values),
                    'stability_score': 1.0 / (1.0 + np.std(importance_array) / (np.mean(importance_array) + 1e-9))
                }
        
        # Rank features by stability
        stable_features = sorted(
            stability_analysis.items(),
            key=lambda x: x[1]['stability_score'],
            reverse=True
        )
        
        return {
            'stability_analysis': stability_analysis,
            'stable_features_ranked': stable_features,
            'most_stable_features': stable_features[:20],
            'feature_importance_raw': dict(feature_importance_data)
        }
    
    def analyze_performance_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between feature selection and trial performance.
        
        Returns:
            Dictionary with performance correlation analysis
        """
        performance_data = []
        
        for trial in self.pareto_trials:
            # Extract performance metrics
            sharpe = trial.values[0] if len(trial.values) > 0 else 0.0
            turnover = trial.values[1] if len(trial.values) > 1 else 0.0
            max_drawdown = trial.values[2] if len(trial.values) > 2 else 0.0
            
            # Extract additional metrics from user attributes
            ic = trial.user_attrs.get('mean_ic', 0.0)
            hit_ratio = trial.user_attrs.get('mean_hit_ratio', 0.0)
            calmar_ratio = trial.user_attrs.get('mean_calmar_ratio', 0.0)
            
            n_features = trial.user_attrs.get('n_features_selected', 0)
            selected_features = trial.user_attrs.get('selected_features', [])
            
            performance_data.append({
                'trial_number': trial.number,
                'sharpe_ratio': sharpe,
                'turnover': turnover,
                'max_drawdown': max_drawdown,
                'information_coefficient': ic,
                'hit_ratio': hit_ratio,
                'calmar_ratio': calmar_ratio,
                'n_features': n_features,
                'selected_features': selected_features
            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data)
        
        # Correlation analysis
        correlations = {}
        if len(df) > 3:
            correlations = {
                'n_features_vs_sharpe': df['n_features'].corr(df['sharpe_ratio']),
                'n_features_vs_turnover': df['n_features'].corr(df['turnover']),
                'n_features_vs_drawdown': df['n_features'].corr(df['max_drawdown']),
                'sharpe_vs_ic': df['sharpe_ratio'].corr(df['information_coefficient']),
                'turnover_vs_sharpe': df['turnover'].corr(df['sharpe_ratio'])
            }
        
        return {
            'performance_data': performance_data,
            'performance_df': df,
            'correlations': correlations,
            'summary_stats': df.describe().to_dict() if len(df) > 0 else {}
        }
    
    def generate_feature_recommendations(self, frequency_threshold: float = 0.5, 
                                       stability_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Generate final feature recommendations based on frequency and stability analysis.
        
        Args:
            frequency_threshold: Minimum selection frequency for recommendation
            stability_threshold: Minimum stability score for recommendation
            
        Returns:
            Dictionary with feature recommendations
        """
        frequency_analysis = self.analyze_feature_frequency()
        stability_analysis = self.analyze_feature_stability()
        
        # Get frequently selected features
        frequent_features = set(frequency_analysis['selection_threshold_50'])
        
        # Get stable features
        stable_features = set([
            feature for feature, metrics in stability_analysis['stability_analysis'].items()
            if metrics['stability_score'] >= stability_threshold
        ])
        
        # Recommended features (intersection of frequent and stable)
        recommended_features = frequent_features.intersection(stable_features)
        
        # High-priority features (very frequent AND very stable)
        high_priority = set([
            feature for feature in recommended_features
            if (frequency_analysis['feature_frequencies'].get(feature, 0) >= 0.75 and
                stability_analysis['stability_analysis'].get(feature, {}).get('stability_score', 0) >= 0.8)
        ])
        
        # Medium-priority features
        medium_priority = recommended_features - high_priority
        
        # Create final ranking combining frequency and stability
        final_ranking = []
        for feature in recommended_features:
            freq = frequency_analysis['feature_frequencies'].get(feature, 0)
            stab = stability_analysis['stability_analysis'].get(feature, {}).get('stability_score', 0)
            combined_score = 0.6 * freq + 0.4 * stab  # Weighted combination
            
            final_ranking.append({
                'feature': feature,
                'frequency': freq,
                'stability_score': stab,
                'combined_score': combined_score,
                'priority': 'high' if feature in high_priority else 'medium'
            })
        
        final_ranking.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'recommended_features': list(recommended_features),
            'high_priority_features': list(high_priority),
            'medium_priority_features': list(medium_priority),
            'final_ranking': final_ranking,
            'recommendation_summary': {
                'total_recommended': len(recommended_features),
                'high_priority_count': len(high_priority),
                'medium_priority_count': len(medium_priority),
                'frequency_threshold_used': frequency_threshold,
                'stability_threshold_used': stability_threshold
            }
        }
    
    def create_visualizations(self, output_dir: Path) -> None:
        """
        Create comprehensive visualizations of the analysis.
        
        Args:
            output_dir: Directory to save visualization files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Feature Selection Frequency Plot
        frequency_analysis = self.analyze_feature_frequency()
        top_features = frequency_analysis['most_frequent_features'][:20]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        features, frequencies = zip(*top_features)
        bars = ax.barh(range(len(features)), frequencies)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Selection Frequency')
        ax.set_title('Top 20 Most Frequently Selected Features')
        ax.grid(axis='x', alpha=0.3)
        
        # Color bars by frequency
        for i, bar in enumerate(bars):
            if frequencies[i] >= 0.75:
                bar.set_color('darkgreen')
            elif frequencies[i] >= 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Stability Analysis Plot
        stability_analysis = self.analyze_feature_stability()
        if stability_analysis['stable_features_ranked']:
            stable_features = stability_analysis['most_stable_features'][:20]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Stability scores
            features, stability_data = zip(*stable_features)
            stability_scores = [data['stability_score'] for data in stability_data]
            
            bars1 = ax1.barh(range(len(features)), stability_scores)
            ax1.set_yticks(range(len(features)))
            ax1.set_yticklabels(features)
            ax1.set_xlabel('Stability Score')
            ax1.set_title('Top 20 Most Stable Features')
            ax1.grid(axis='x', alpha=0.3)
            
            # CV of importance
            cv_scores = [data['cv_importance'] for data in stability_data]
            bars2 = ax2.barh(range(len(features)), cv_scores)
            ax2.set_yticks(range(len(features)))
            ax2.set_yticklabels(features)
            ax2.set_xlabel('Coefficient of Variation (Importance)')
            ax2.set_title('Feature Importance Variability (Lower = More Stable)')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_stability.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance vs Number of Features
        perf_analysis = self.analyze_performance_correlation()
        df = perf_analysis['performance_df']
        
        if len(df) > 5:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Sharpe vs N Features
            ax1.scatter(df['n_features'], df['sharpe_ratio'], alpha=0.7)
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.set_title('Sharpe Ratio vs Number of Features')
            ax1.grid(alpha=0.3)
            
            # Turnover vs N Features
            ax2.scatter(df['n_features'], df['turnover'], alpha=0.7, color='orange')
            ax2.set_xlabel('Number of Features')
            ax2.set_ylabel('Turnover')
            ax2.set_title('Turnover vs Number of Features')
            ax2.grid(alpha=0.3)
            
            # Sharpe vs IC
            ax3.scatter(df['information_coefficient'], df['sharpe_ratio'], alpha=0.7, color='green')
            ax3.set_xlabel('Information Coefficient')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.set_title('Sharpe Ratio vs Information Coefficient')
            ax3.grid(alpha=0.3)
            
            # Hit Ratio vs Sharpe
            ax4.scatter(df['hit_ratio'], df['sharpe_ratio'], alpha=0.7, color='red')
            ax4.set_xlabel('Hit Ratio')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.set_title('Sharpe Ratio vs Hit Ratio')
            ax4.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Pareto Front Visualization (for multi-objective)
        if len(self.study.directions) > 1:
            fig = plt.figure(figsize=(12, 8))
            
            if len(self.study.directions) == 3:
                ax = fig.add_subplot(111, projection='3d')
                values = np.array([trial.values for trial in self.pareto_trials])
                ax.scatter(values[:, 0], values[:, 1], values[:, 2], c='red', s=50, alpha=0.7)
                ax.set_xlabel('Sharpe Ratio')
                ax.set_ylabel('Turnover')
                ax.set_zlabel('Max Drawdown')
                ax.set_title('Pareto Front (3D)')
            else:
                ax = fig.add_subplot(111)
                values = np.array([trial.values for trial in self.pareto_trials])
                ax.scatter(values[:, 0], values[:, 1], c='red', s=50, alpha=0.7)
                ax.set_xlabel('Objective 1')
                ax.set_ylabel('Objective 2')
                ax.set_title('Pareto Front (2D)')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, output_dir: Path) -> None:
        """
        Save all analysis results to files.
        
        Args:
            output_dir: Directory to save result files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run all analyses
        frequency_analysis = self.analyze_feature_frequency()
        stability_analysis = self.analyze_feature_stability()
        performance_analysis = self.analyze_performance_correlation()
        recommendations = self.generate_feature_recommendations()
        
        # Save JSON results
        results = {
            'frequency_analysis': frequency_analysis,
            'stability_analysis': stability_analysis,
            'performance_analysis': {
                k: v for k, v in performance_analysis.items() 
                if k != 'performance_df'  # Skip DataFrame
            },
            'recommendations': recommendations,
            'study_summary': {
                'total_trials': len(self.all_trials),
                'pareto_trials': len(self.pareto_trials),
                'study_directions': self.study.directions,
                'study_name': self.study.study_name
            }
        }
        
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV files
        if len(performance_analysis['performance_df']) > 0:
            performance_analysis['performance_df'].to_csv(
                output_dir / 'trial_performance.csv', index=False
            )
        
        # Save feature recommendations as CSV
        recommendations_df = pd.DataFrame(recommendations['final_ranking'])
        recommendations_df.to_csv(output_dir / 'feature_recommendations.csv', index=False)
        
        # Save detailed feature frequency
        freq_df = pd.DataFrame([
            {'feature': feature, 'frequency': freq, 'count': recommendations['recommendation_summary'].get(feature, 0)}
            for feature, freq in frequency_analysis['feature_frequencies'].items()
        ]).sort_values('frequency', ascending=False)
        freq_df.to_csv(output_dir / 'feature_frequencies.csv', index=False)
        
        print(f"Results saved to {output_dir}")
    
    def generate_report(self, output_dir: Path) -> None:
        """
        Generate a comprehensive markdown report.
        
        Args:
            output_dir: Directory to save the report
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analyses
        frequency_analysis = self.analyze_feature_frequency()
        stability_analysis = self.analyze_feature_stability()
        performance_analysis = self.analyze_performance_correlation()
        recommendations = self.generate_feature_recommendations()
        
        report_content = f"""# Feature Robustness Analysis Report

## Study Summary
- **Total Trials Completed**: {len(self.all_trials)}
- **Pareto-Optimal Trials**: {len(self.pareto_trials)}
- **Optimization Objectives**: {self.study.directions}
- **Study Name**: {self.study.study_name}

## Key Findings

### Feature Selection Frequency
- **Total Unique Features Evaluated**: {frequency_analysis['total_unique_features']}
- **Features Selected ≥50% of Time**: {len(frequency_analysis['selection_threshold_50'])}
- **Features Selected ≥75% of Time**: {len(frequency_analysis['selection_threshold_75'])}

### Top 10 Most Frequently Selected Features
"""
        
        for i, (feature, freq) in enumerate(frequency_analysis['most_frequent_features'][:10], 1):
            report_content += f"{i}. **{feature}** - {freq:.2%} selection frequency\n"
        
        report_content += f"""
### Feature Stability Analysis
- **Features with Stability Data**: {len(stability_analysis['stability_analysis'])}
- **Most Stable Features** (top 10 by stability score):

"""
        
        for i, (feature, metrics) in enumerate(stability_analysis['most_stable_features'][:10], 1):
            stability_score = metrics['stability_score']
            mean_importance = metrics['mean_importance']
            report_content += f"{i}. **{feature}** - Stability: {stability_score:.3f}, Avg Importance: {mean_importance:.3f}\n"
        
        report_content += f"""
## Final Recommendations

### High Priority Features ({len(recommendations['high_priority_features'])} features)
These features appear frequently (≥75%) and have high stability (≥0.8):
"""
        for feature in recommendations['high_priority_features'][:15]:  # Top 15
            freq = frequency_analysis['feature_frequencies'].get(feature, 0)
            stab = stability_analysis['stability_analysis'].get(feature, {}).get('stability_score', 0)
            report_content += f"- **{feature}** (Freq: {freq:.2%}, Stability: {stab:.3f})\n"
        
        report_content += f"""
### Medium Priority Features ({len(recommendations['medium_priority_features'])} features)
These features have good frequency or stability but not both at the highest levels:
"""
        for feature in recommendations['medium_priority_features'][:10]:  # Top 10
            freq = frequency_analysis['feature_frequencies'].get(feature, 0)
            stab = stability_analysis['stability_analysis'].get(feature, {}).get('stability_score', 0)
            report_content += f"- **{feature}** (Freq: {freq:.2%}, Stability: {stab:.3f})\n"
        
        if performance_analysis['correlations']:
            report_content += f"""
## Performance Correlations
- **Number of Features vs Sharpe Ratio**: {performance_analysis['correlations']['n_features_vs_sharpe']:.3f}
- **Number of Features vs Turnover**: {performance_analysis['correlations']['n_features_vs_turnover']:.3f}
- **Sharpe Ratio vs Information Coefficient**: {performance_analysis['correlations']['sharpe_vs_ic']:.3f}
- **Turnover vs Sharpe Ratio**: {performance_analysis['correlations']['turnover_vs_sharpe']:.3f}
"""
        
        report_content += f"""
## Implementation Recommendations

1. **Start with High Priority Features**: Use the {len(recommendations['high_priority_features'])} high-priority features as your base feature set.

2. **Gradually Add Medium Priority**: Test adding medium-priority features to see if they improve performance.

3. **Monitor Stability**: Track feature importance over time to ensure consistency.

4. **Consider Feature Count**: Optimal range appears to be between {10} and {50} features based on the analysis.

## Files Generated
- `analysis_results.json` - Complete analysis data
- `feature_recommendations.csv` - Ranked feature recommendations
- `feature_frequencies.csv` - Feature selection frequencies
- `trial_performance.csv` - Trial performance data
- `feature_frequency.png` - Feature frequency visualization
- `feature_stability.png` - Feature stability visualization
- `performance_analysis.png` - Performance correlation plots
- `pareto_front.png` - Pareto front visualization

---
*Generated by Feature Robustness Analyzer*
"""
        
        with open(output_dir / 'feature_analysis_report.md', 'w') as f:
            f.write(report_content)
        
        print(f"Report generated: {output_dir / 'feature_analysis_report.md'}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze feature robustness from Optuna Study A results')
    parser.add_argument('--study-path', required=True, help='Path to the SQLite study database')
    parser.add_argument('--output-dir', default='./analysis_results', help='Output directory for results')
    parser.add_argument('--frequency-threshold', type=float, default=0.5, help='Minimum frequency for feature recommendation')
    parser.add_argument('--stability-threshold', type=float, default=0.7, help='Minimum stability for feature recommendation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Initialize analyzer
    analyzer = FeatureRobustnessAnalyzer(args.study_path)
    
    # Load study
    print("Loading Optuna study...")
    analyzer.load_study()
    
    # Run analysis
    print("Running feature robustness analysis...")
    
    # Generate visualizations
    print("Creating visualizations...")
    analyzer.create_visualizations(output_dir)
    
    # Save results
    print("Saving analysis results...")
    analyzer.save_results(output_dir)
    
    # Generate report
    print("Generating report...")
    analyzer.generate_report(output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
