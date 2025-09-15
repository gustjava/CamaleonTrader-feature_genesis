#!/usr/bin/env python3
"""
Model Database Query Utility

Script para consultar e visualizar os modelos CatBoost salvos no banco de dados.
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.unified_config import get_unified_config
from data_io.db_handler import DatabaseHandler


class ModelDatabaseQuery:
    """Utility class for querying model database."""
    
    def __init__(self):
        """Initialize database connection."""
        self.config = get_unified_config()
        self.db_handler = DatabaseHandler(self.config.database)
    
    def list_all_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all models in the database."""
        try:
            with self.db_handler.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                where_clause = "WHERE is_active = TRUE" if active_only else ""
                
                query = f"""
                SELECT 
                    id, model_name, model_version, symbol, timeframe, task_type,
                    feature_count, training_samples, test_samples, train_score, test_score,
                    is_active, created_at, updated_at
                FROM final_models 
                {where_clause}
                ORDER BY symbol, timeframe, model_version DESC
                """
                
                cursor.execute(query)
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error querying models: {e}")
            return []
    
    def get_model_details(self, model_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific model."""
        try:
            with self.db_handler.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                # Get model info
                cursor.execute("SELECT * FROM final_models WHERE id = %s", (model_id,))
                model = cursor.fetchone()
                
                if not model:
                    return None
                
                # Get features
                cursor.execute("""
                    SELECT feature_name, selection_importance, final_importance, 
                           rank_selection, rank_final
                    FROM model_features 
                    WHERE model_id = %s 
                    ORDER BY rank_final
                """, (model_id,))
                features = cursor.fetchall()
                
                # Get metrics
                cursor.execute("""
                    SELECT metric_name, metric_value, metric_category
                    FROM model_metrics 
                    WHERE model_id = %s 
                    ORDER BY metric_category, metric_name
                """, (model_id,))
                metrics = cursor.fetchall()
                
                return {
                    'model': model,
                    'features': features,
                    'metrics': metrics
                }
                
        except Exception as e:
            print(f"Error getting model details: {e}")
            return None
    
    def get_models_by_symbol(self, symbol: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all models for a specific symbol."""
        try:
            with self.db_handler.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                where_clause = "WHERE symbol = %s"
                params = [symbol]
                
                if active_only:
                    where_clause += " AND is_active = TRUE"
                
                query = f"""
                SELECT 
                    id, model_name, model_version, symbol, timeframe, task_type,
                    feature_count, training_samples, test_samples, train_score, test_score,
                    is_active, created_at
                FROM final_models 
                {where_clause}
                ORDER BY timeframe, model_version DESC
                """
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            print(f"Error querying models for symbol {symbol}: {e}")
            return []
    
    def compare_models(self, model_ids: List[int]) -> Dict[str, Any]:
        """Compare multiple models."""
        models_data = []
        
        for model_id in model_ids:
            model_data = self.get_model_details(model_id)
            if model_data:
                models_data.append(model_data)
        
        if not models_data:
            return {}
        
        # Organize comparison data
        comparison = {
            'models': [],
            'feature_overlap': {},
            'metric_comparison': {}
        }
        
        all_features = set()
        all_metrics = set()
        
        for model_data in models_data:
            model = model_data['model']
            features = model_data['features']
            metrics = model_data['metrics']
            
            model_summary = {
                'id': model['id'],
                'name': model['model_name'],
                'symbol': model['symbol'],
                'timeframe': model['timeframe'],
                'test_score': model['test_score'],
                'feature_count': model['feature_count'],
                'top_features': [f['feature_name'] for f in features[:10]],
                'key_metrics': {m['metric_name']: m['metric_value'] for m in metrics 
                               if m['metric_name'] in ['test_r2', 'test_skill_score', 'test_information_coefficient']}
            }
            
            comparison['models'].append(model_summary)
            
            # Track features and metrics for overlap analysis
            model_features = {f['feature_name'] for f in features}
            all_features.update(model_features)
            all_metrics.update(m['metric_name'] for m in metrics)
        
        return comparison

    def print_model_summary(self, models: List[Dict[str, Any]]):
        """Print a formatted summary of models."""
        if not models:
            print("No models found.")
            return
        
        print(f"\n{'='*100}")
        print(f"{'MODEL SUMMARY':^100}")
        print(f"{'='*100}")
        
        print(f"{'ID':<5} {'Model Name':<25} {'Symbol':<8} {'TF':<5} {'Type':<12} {'Features':<8} {'Test Score':<10} {'Created':<20}")
        print("-" * 100)
        
        for model in models:
            created = model['created_at'].strftime('%Y-%m-%d %H:%M') if model['created_at'] else 'Unknown'
            active_indicator = "🟢" if model.get('is_active', False) else "🔴"
            
            print(f"{model['id']:<5} {model['model_name']:<25} {model['symbol']:<8} "
                  f"{model['timeframe']:<5} {model['task_type']:<12} {model['feature_count']:<8} "
                  f"{model['test_score']:<10.6f} {created:<20} {active_indicator}")

    def print_model_details(self, model_id: int):
        """Print detailed information for a specific model."""
        details = self.get_model_details(model_id)
        
        if not details:
            print(f"Model {model_id} not found.")
            return
        
        model = details['model']
        features = details['features']
        metrics = details['metrics']
        
        print(f"\n{'='*80}")
        print(f"MODEL DETAILS - ID: {model['id']}")
        print(f"{'='*80}")
        
        print(f"Model Name: {model['model_name']}")
        print(f"Version: {model['model_version']}")
        print(f"Symbol: {model['symbol']}")
        print(f"Timeframe: {model['timeframe']}")
        print(f"Task Type: {model['task_type']}")
        print(f"Feature Count: {model['feature_count']}")
        print(f"Training Samples: {model['training_samples']:,}")
        print(f"Test Samples: {model['test_samples']:,}")
        print(f"Train Score: {model['train_score']:.6f}")
        print(f"Test Score: {model['test_score']:.6f}")
        print(f"Active: {'Yes' if model['is_active'] else 'No'}")
        print(f"Created: {model['created_at']}")
        
        # Top Features
        print(f"\n🏆 TOP 15 FEATURES (by final importance):")
        print("-" * 60)
        for i, feature in enumerate(features[:15], 1):
            print(f"{i:2d}. {feature['feature_name']:<40} {feature['final_importance']:.6f}")
        
        # Key Metrics
        key_metrics = ['test_r2', 'test_skill_score', 'test_information_coefficient', 
                      'test_rmse', 'test_mae', 'train_r2']
        
        print(f"\n📊 KEY METRICS:")
        print("-" * 40)
        
        metrics_dict = {m['metric_name']: m['metric_value'] for m in metrics}
        
        for metric_name in key_metrics:
            if metric_name in metrics_dict:
                print(f"{metric_name:<30} {metrics_dict[metric_name]:.6f}")
        
        # Model Configuration
        if model['model_config']:
            try:
                config = json.loads(model['model_config'])
                print(f"\n⚙️  MODEL CONFIGURATION:")
                print("-" * 40)
                for key, value in config.items():
                    print(f"{key:<25} {value}")
            except:
                pass


def main():
    """Main function for command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query CatBoost model database")
    parser.add_argument('--list', action='store_true', help='List all models')
    parser.add_argument('--list-all', action='store_true', help='List all models (including inactive)')
    parser.add_argument('--details', type=int, help='Show details for specific model ID')
    parser.add_argument('--symbol', type=str, help='Show models for specific symbol')
    parser.add_argument('--compare', nargs='+', type=int, help='Compare multiple models by ID')
    
    args = parser.parse_args()
    
    if not any([args.list, args.list_all, args.details, args.symbol, args.compare]):
        parser.print_help()
        return
    
    query = ModelDatabaseQuery()
    
    if args.list or args.list_all:
        models = query.list_all_models(active_only=not args.list_all)
        query.print_model_summary(models)
        
        if models:
            print(f"\nTotal: {len(models)} models")
            active_count = sum(1 for m in models if m.get('is_active', False))
            print(f"Active: {active_count}, Inactive: {len(models) - active_count}")
    
    elif args.details:
        query.print_model_details(args.details)
    
    elif args.symbol:
        models = query.get_models_by_symbol(args.symbol.upper())
        print(f"\nModels for {args.symbol.upper()}:")
        query.print_model_summary(models)
    
    elif args.compare:
        comparison = query.compare_models(args.compare)
        if comparison:
            print(f"\nModel Comparison:")
            for model in comparison['models']:
                print(f"\nModel {model['id']} ({model['name']}):")
                print(f"  Symbol: {model['symbol']}, Timeframe: {model['timeframe']}")
                print(f"  Test Score: {model['test_score']:.6f}")
                print(f"  Features: {model['feature_count']}")
                print(f"  Top Features: {', '.join(model['top_features'][:5])}")


if __name__ == "__main__":
    main()
