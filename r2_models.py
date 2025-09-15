#!/usr/bin/env python3
"""
R2 Model Management Utility

This utility provides command-line interface for managing CatBoost models
stored in Cloudflare R2 storage.
"""

import click
import json
import sys
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add project root to path for imports
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_io.r2_uploader import R2ModelUploader
from config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """R2 Model Management Utility"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option('--symbol', '-s', help='Filter by currency pair symbol')
@click.option('--format', 'output_format', 
              type=click.Choice(['table', 'json', 'summary']), 
              default='table',
              help='Output format')
def list_models(symbol: Optional[str], output_format: str):
    """List all models in R2 storage."""
    try:
        uploader = R2ModelUploader()
        models = uploader.list_uploaded_models(symbol=symbol)
        
        if not models:
            print(f"No models found" + (f" for symbol {symbol}" if symbol else ""))
            return
        
        if output_format == 'json':
            print(json.dumps(models, indent=2, default=str))
        elif output_format == 'summary':
            print(f"Found {len(models)} models" + (f" for {symbol}" if symbol else ""))
            symbols = set(m['symbol'] for m in models)
            print(f"Symbols: {', '.join(sorted(symbols))}")
        else:
            # Table format
            print(f"{'Model Name':<40} {'Symbol':<10} {'Size (KB)':<12} {'Last Modified':<20}")
            print("-" * 85)
            for model in sorted(models, key=lambda x: (x['symbol'], x['model_name'])):
                size_kb = model['size'] / 1024
                last_modified = datetime.fromisoformat(model['last_modified'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                print(f"{model['model_name']:<40} {model['symbol']:<10} {size_kb:<12.1f} {last_modified:<20}")
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


@cli.command()
@click.argument('symbol')
@click.argument('model_name')
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'summary']), 
              default='summary',
              help='Output format')
def get_metadata(symbol: str, model_name: str, output_format: str):
    """Get detailed metadata for a specific model."""
    try:
        uploader = R2ModelUploader()
        metadata = uploader.download_model_metadata(symbol, model_name)
        
        if not metadata:
            print(f"Metadata not found for model {model_name} (symbol: {symbol})")
            sys.exit(1)
        
        if output_format == 'json':
            print(json.dumps(metadata, indent=2, default=str))
        else:
            # Summary format
            print(f"Model: {metadata['model_name']}")
            print(f"Symbol: {metadata['symbol']}")
            print(f"Version: {metadata['model_version']}")
            print(f"Task Type: {metadata.get('model_type', 'Unknown')}")
            print(f"Created: {metadata['created_at']}")
            
            # Training config
            config = metadata.get('training_config', {})
            print(f"\nTraining Configuration:")
            print(f"  Iterations: {config.get('iterations', 'Unknown')}")
            print(f"  Learning Rate: {config.get('learning_rate', 'Unknown')}")
            print(f"  Depth: {config.get('depth', 'Unknown')}")
            print(f"  L2 Leaf Reg: {config.get('l2_leaf_reg', 'Unknown')}")
            
            # Features
            features_info = metadata.get('features', {})
            print(f"\nFeatures:")
            print(f"  Count: {features_info.get('num_features', 'Unknown')}")
            
            top_features = features_info.get('top_10_features', [])
            if top_features:
                print(f"  Top 10 Features:")
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"    {i:2d}. {feature:<30} {importance:.4f}")
            
            # Performance metrics
            metrics = metadata.get('evaluation_metrics', {})
            if metrics:
                print(f"\nPerformance Metrics:")
                for metric_name, value in sorted(metrics.items()):
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")
                    else:
                        print(f"  {metric_name}: {value}")
        
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        sys.exit(1)


@cli.command()
@click.argument('symbol')
@click.argument('model_name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_model(symbol: str, model_name: str, confirm: bool):
    """Delete a model from R2 storage."""
    try:
        if not confirm:
            click.confirm(f"Are you sure you want to delete model '{model_name}' for symbol '{symbol}'?", abort=True)
        
        uploader = R2ModelUploader()
        success = uploader.delete_model(symbol, model_name)
        
        if success:
            print(f"Successfully deleted model {model_name} for symbol {symbol}")
        else:
            print(f"Failed to delete model {model_name} for symbol {symbol}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        sys.exit(1)


@cli.command()
@click.option('--symbol', '-s', help='Filter by currency pair symbol')
def storage_stats(symbol: Optional[str]):
    """Show storage statistics."""
    try:
        uploader = R2ModelUploader()
        models = uploader.list_uploaded_models(symbol=symbol)
        
        if not models:
            print(f"No models found" + (f" for symbol {symbol}" if symbol else ""))
            return
        
        total_size = sum(model['size'] for model in models)
        total_size_mb = total_size / (1024 * 1024)
        
        # Group by symbol
        by_symbol = {}
        for model in models:
            sym = model['symbol']
            if sym not in by_symbol:
                by_symbol[sym] = {'count': 0, 'size': 0, 'latest': None}
            
            by_symbol[sym]['count'] += 1
            by_symbol[sym]['size'] += model['size']
            
            model_date = datetime.fromisoformat(model['last_modified'].replace('Z', '+00:00'))
            if by_symbol[sym]['latest'] is None or model_date > by_symbol[sym]['latest']:
                by_symbol[sym]['latest'] = model_date
        
        print(f"R2 Storage Statistics" + (f" for {symbol}" if symbol else ""))
        print("=" * 50)
        print(f"Total Models: {len(models)}")
        print(f"Total Size: {total_size_mb:.2f} MB")
        print(f"Average Size: {total_size_mb / len(models):.2f} MB")
        
        print(f"\nBy Symbol:")
        print(f"{'Symbol':<10} {'Count':<8} {'Size (MB)':<12} {'Latest':<20}")
        print("-" * 55)
        
        for sym in sorted(by_symbol.keys()):
            stats = by_symbol[sym]
            size_mb = stats['size'] / (1024 * 1024)
            latest_str = stats['latest'].strftime('%Y-%m-%d %H:%M') if stats['latest'] else 'Unknown'
            print(f"{sym:<10} {stats['count']:<8} {size_mb:<12.2f} {latest_str:<20}")
        
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        sys.exit(1)


@cli.command()
@click.option('--symbol', '-s', multiple=True, help='Symbols to compare (can be used multiple times)')
@click.option('--metric', '-m', default='test_score', help='Metric to compare (default: test_score)')
def compare_models(symbol: List[str], metric: str):
    """Compare models across symbols."""
    try:
        uploader = R2ModelUploader()
        
        if not symbol:
            # Get all symbols
            all_models = uploader.list_uploaded_models()
            symbols = list(set(m['symbol'] for m in all_models))
        else:
            symbols = list(symbol)
        
        print(f"Comparing models by '{metric}':")
        print("=" * 60)
        
        comparison_data = []
        
        for sym in sorted(symbols):
            models = uploader.list_uploaded_models(symbol=sym)
            
            for model in models:
                try:
                    metadata = uploader.download_model_metadata(sym, model['model_name'])
                    if metadata and 'evaluation_metrics' in metadata:
                        metrics = metadata['evaluation_metrics']
                        if metric in metrics:
                            comparison_data.append({
                                'symbol': sym,
                                'model_name': model['model_name'],
                                'metric_value': metrics[metric],
                                'created_at': metadata['created_at']
                            })
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {model['model_name']}: {e}")
        
        if not comparison_data:
            print(f"No models found with metric '{metric}'")
            return
        
        # Sort by metric value (descending)
        comparison_data.sort(key=lambda x: x['metric_value'], reverse=True)
        
        print(f"{'Rank':<6} {'Symbol':<10} {'Model':<35} {metric:<15} {'Created':<20}")
        print("-" * 90)
        
        for i, data in enumerate(comparison_data, 1):
            created_date = datetime.fromisoformat(data['created_at']).strftime('%Y-%m-%d %H:%M')
            print(f"{i:<6} {data['symbol']:<10} {data['model_name']:<35} {data['metric_value']:<15.4f} {created_date:<20}")
        
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        sys.exit(1)


@cli.command()
def validate_config():
    """Validate R2 configuration."""
    try:
        config = get_config()
        uploader = R2ModelUploader()
        
        print("R2 Configuration Validation:")
        print("=" * 40)
        
        # Check configuration fields
        required_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name', 'endpoint_url']
        for field in required_fields:
            value = config.r2.get(field)
            status = "✓" if value else "✗"
            display_value = value if field not in ['access_key', 'secret_key'] else "***hidden***"
            print(f"{status} {field}: {display_value}")
        
        # Test connection
        print(f"\nTesting R2 connection...")
        try:
            models = uploader.list_uploaded_models()
            print(f"✓ Connection successful - found {len(models)} models")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
