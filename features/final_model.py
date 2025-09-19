"""
Final Model Training and Evaluation Module

This module handles the construction, training, and evaluation of the final CatBoost model
using selected features, with comprehensive metrics logging to the database.
"""

import logging
import numpy as np
import cupy as cp
import cudf
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback as _tb

# Suppress GPU memory warnings globally for this module
warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')

from utils.logging_utils import get_logger
from utils.trading_metrics import TradingMetrics, GPUPostTrainingMetrics
from data_io.r2_uploader import R2ModelUploader

logger = get_logger(__name__, "features.final_model")


class FinalModelTrainer:
    """Class for training and evaluating the final CatBoost model with selected features."""
    
    def __init__(self, config, logger_instance=None):
        """Initialize the final model trainer."""
        self.config = config
        self.logger = logger_instance or logger
        self.trading_metrics = TradingMetrics(config=config)
        self.r2_uploader = R2ModelUploader()  # Initialize R2 uploader
        
    def _log_info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if self.logger:
            self.logger.info(f"FinalModel: {message}", extra=kwargs)
    
    def _log_warn(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if self.logger:
            self.logger.warning(f"FinalModel: {message}", extra=kwargs)
    
    def _log_error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if self.logger:
            self.logger.error(f"FinalModel: {message}", extra=kwargs)
    
    def _critical_error(self, message: str, **kwargs):
        """Log critical error and raise exception."""
        self._log_error(message, **kwargs)
        raise RuntimeError(f"FinalModel Critical Error: {message}")
    
    def _cleanup_model_memory(self, model_results: Dict[str, Any], symbol: str, timeframe: str) -> None:
        """
        Clean up GPU memory after model training to prevent accumulation between currencies.
        
        Args:
            model_results: Results containing model and predictions
            symbol: Currency symbol for logging
            timeframe: Timeframe for logging
        """
        try:
            # 1. Clear CatBoost model from memory
            if 'model' in model_results:
                try:
                    # Clear model reference
                    del model_results['model']
                except:
                    pass
            
            # 2. Clear prediction arrays
            for key in ['train_predictions', 'test_predictions']:
                if key in model_results:
                    try:
                        del model_results[key]
                    except:
                        pass
            
            # 3. Force garbage collection
            import gc
            gc.collect()
            
            # 4. Clear GPU memory pools (CuPy)
            try:
                import cupy as cp
                # Free all unused memory blocks
                cp.get_default_memory_pool().free_all_blocks()
                # Free all unused pinned memory
                cp.get_default_pinned_memory_pool().free_all_blocks()
                
                # Get memory usage after cleanup
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                used_mb = used_bytes / (1024 * 1024)
                
                self._log_info('GPU memory cleanup completed',
                              symbol=symbol,
                              timeframe=timeframe,
                              gpu_memory_used_mb=f"{used_mb:.1f}")
                
            except ImportError:
                # CuPy not available, skip GPU cleanup
                self._log_info('CPU memory cleanup completed (CuPy not available)',
                              symbol=symbol,
                              timeframe=timeframe)
            except Exception as cupy_err:
                self._log_warn('GPU memory cleanup failed',
                              symbol=symbol,
                              timeframe=timeframe,
                              error=str(cupy_err))
                
        except Exception as e:
            self._log_warn('Memory cleanup failed',
                          symbol=symbol,
                          timeframe=timeframe,
                          error=str(e))

    def build_and_evaluate_final_model(self, 
                                     X_df,  # Accept both cudf.DataFrame and dask_cudf.DataFrame
                                     y_series,  # Accept both cudf.Series and dask_cudf.Series
                                     selected_features: List[str], 
                                     feature_importances: Dict[str, float],
                                     selection_metadata: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str) -> Dict[str, Any]:
        """
        Build, train and evaluate the final CatBoost model with selected features.
        
        Args:
            X_df: Full feature DataFrame
            y_series: Target series
            selected_features: List of selected feature names
            feature_importances: Dictionary of feature importances from selection
            selection_metadata: Metadata from feature selection process
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1h')
            
        Returns:
            Dictionary with model results and metadata
        """
        
        # Check if we need to convert dask_cudf to cudf
        is_dask = hasattr(X_df, 'compute') and not hasattr(X_df, 'to_arrow')
        if is_dask:
            self._log_info('Converting dask_cudf to cudf for final model training...')
            X_df = X_df.compute()
            y_series = y_series.compute()
            self._log_info('Conversion completed', 
                          samples=len(X_df),
                          features=len(X_df.columns))
        
        self._log_info('Starting final model training', 
                       symbol=symbol, 
                       timeframe=timeframe,
                       selected_features=len(selected_features),
                       total_samples=len(X_df))
        
        try:
            # 1. Prepare data with selected features only
            if not selected_features:
                self._critical_error('No features selected for final model')
            
            # Ensure all selected features exist in the DataFrame
            missing_features = [f for f in selected_features if f not in X_df.columns]
            if missing_features:
                self._log_warn('Some selected features missing from DataFrame', 
                              missing=len(missing_features), 
                              examples=missing_features[:5])
                selected_features = [f for f in selected_features if f in X_df.columns]
                
            if not selected_features:
                self._critical_error('No valid selected features after filtering')
            
            # Create final dataset
            X_final = X_df[selected_features].copy()
            y_final = y_series.copy()
            
            # Remove any rows with NaN values
            combined_df = X_final.copy()
            combined_df['__target__'] = y_final
            combined_df = combined_df.dropna()
            
            if len(combined_df) < 100:
                self._critical_error('Insufficient data after cleaning for final model', 
                                   samples=len(combined_df))
            
            y_clean = combined_df.pop('__target__')
            X_clean = combined_df
            
            self._log_info('Data prepared for final model', 
                           features=len(selected_features),
                           samples=len(X_clean),
                           target_std=float(y_clean.std()),
                           target_mean=float(y_clean.mean()))
            
            # 2. Determine task type
            task_type = self._determine_task_type(y_clean)
            
            # 3. Split data for training and testing
            train_test_split = self._create_time_series_split(X_clean, y_clean)
            
            # 4. Build and train final model
            model_results = self._train_final_catboost_model(
                train_test_split, task_type, selected_features
            )
            
            # Capture frac_diff details before converting to numpy
            frac_diff_details = self._extract_frac_diff_details(X_clean, selected_features)

            # 5. Comprehensive evaluation
            evaluation_results = self._comprehensive_evaluation(
                model_results,
                train_test_split,
                task_type,
                symbol=symbol,
                timeframe=timeframe,
                frac_diff_details=frac_diff_details
            )
            
            # 6. Upload model to R2 cloud storage (skip database - using JSON metadata)
            r2_upload_success = self._upload_model_to_r2(
                symbol=symbol,
                timeframe=timeframe,
                selected_features=selected_features,
                feature_importances=feature_importances,
                model_results=model_results,
                evaluation_results=evaluation_results,
                task_type=task_type
            )
            
            # 7. Compile final results
            final_results = {
                'r2_upload_success': r2_upload_success,
                'symbol': symbol,
                'timeframe': timeframe,
                'task_type': task_type,
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'training_samples': len(train_test_split['X_train']),
                'test_samples': len(train_test_split['X_test']),
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'selection_metadata': selection_metadata,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            self._log_info('Final model training completed successfully', 
                           train_score=evaluation_results.get('train_primary_metric', 0.0),
                           test_score=evaluation_results.get('test_primary_metric', 0.0))
            
            # NOTE: Memory cleanup moved to AFTER R2 upload to avoid 
            # deleting model before upload completes
            
            return final_results
            
        except Exception as e:
            self._log_error('Final model training failed', 
                           error=str(e), 
                           error_type=type(e).__name__,
                           traceback=_tb.format_exc())
            
            # Clean up memory even on failure
            try:
                if 'model_results' in locals():
                    self._cleanup_model_memory(model_results, symbol, timeframe)
                else:
                    # Fallback cleanup without model_results
                    import gc
                    gc.collect()
                    try:
                        import cupy as cp
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except:
                        pass
            except Exception as cleanup_err:
                self._log_warn('Cleanup after error failed', error=str(cleanup_err))
            
            raise

    def _determine_task_type(self, y_series) -> str:
        """Determine if this is a regression or classification task."""
        try:
            y_vals = y_series.values if hasattr(y_series, 'values') else np.asarray(y_series)
            unique_values = len(np.unique(y_vals[np.isfinite(y_vals)]))
            
            # For forex trading data, ALWAYS use regression (targets are continuous returns)
            # This ensures trading metrics (IC, ICIR, Hit, Sharpe, etc.) are calculated
            task_type = 'regression'
            self._log_info('Task determined as regression (forex trading data)', 
                          unique_values=unique_values,
                          y_std=float(np.std(y_vals)),
                          y_min=float(np.min(y_vals)),
                          y_max=float(np.max(y_vals)))
            
            return task_type
            
        except Exception as e:
            self._log_warn('Failed to determine task type; defaulting to regression', error=str(e))
            return 'regression'

    def _create_time_series_split(self, X_df, y_series, test_ratio: float = 0.2) -> Dict[str, Any]:
        """Create time-series aware train/test split."""
        n_samples = len(X_df)
        split_idx = int(n_samples * (1 - test_ratio))
        
        # Convert to numpy for CatBoost
        X_train = X_df.iloc[:split_idx].to_numpy(dtype=np.float32)
        X_test = X_df.iloc[split_idx:].to_numpy(dtype=np.float32)
        y_train = y_series.iloc[:split_idx].to_numpy(dtype=np.float32)
        y_test = y_series.iloc[split_idx:].to_numpy(dtype=np.float32)
        
        feature_names = list(X_df.columns)
        
        self._log_info('Time series split created', 
                       train_samples=len(X_train),
                       test_samples=len(X_test),
                       split_ratio=f"{1-test_ratio:.1%}/{test_ratio:.1%}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'split_idx': split_idx
        }

    def _train_final_catboost_model(self, 
                                  train_test_split: Dict[str, Any], 
                                  task_type: str, 
                                  selected_features: List[str]) -> Dict[str, Any]:
        """Train the final CatBoost model with optimized parameters."""
        
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        import warnings
        
        # Get enhanced parameters from config
        iterations = int(getattr(self.config.features, 'stage3_catboost_iterations', 750))
        learning_rate = float(getattr(self.config.features, 'stage3_catboost_learning_rate', 0.025))
        depth = int(getattr(self.config.features, 'stage3_catboost_depth', 6))
        l2_leaf_reg = float(getattr(self.config.features, 'stage3_catboost_l2_leaf_reg', 10.0))
        bootstrap_type = str(getattr(self.config.features, 'stage3_catboost_bootstrap_type', 'Bernoulli'))
        subsample = float(getattr(self.config.features, 'stage3_catboost_subsample', 0.7))
        task_type_gpu = str(getattr(self.config.features, 'stage3_catboost_task_type', 'GPU'))
        devices = str(getattr(self.config.features, 'stage3_catboost_devices', '0'))
        early_stopping = int(getattr(self.config.features, 'stage3_catboost_early_stopping_rounds', 100))
        random_state = int(getattr(self.config.features, 'stage3_random_state', 42))
        
        self._log_info('Training final CatBoost model', 
                       task_type=task_type,
                       iterations=iterations,
                       learning_rate=learning_rate,
                       depth=depth,
                       l2_leaf_reg=l2_leaf_reg,
                       bootstrap_type=bootstrap_type)
        
        try:
            # Create data pools
            train_pool = Pool(
                train_test_split['X_train'], 
                train_test_split['y_train'], 
                feature_names=train_test_split['feature_names']
            )
            
            test_pool = Pool(
                train_test_split['X_test'], 
                train_test_split['y_test'], 
                feature_names=train_test_split['feature_names']
            )
            
            # Build model based on task type
            if task_type == 'classification':
                unique_y = np.unique(train_test_split['y_train'])
                is_binary = len(unique_y) == 2
                loss_function = 'Logloss' if is_binary else 'MultiClass'
                eval_metric = 'AUC' if is_binary else 'Accuracy'
                
                # Suppress GPU memory warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
                    model = CatBoostClassifier(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        random_seed=random_state,
                        task_type=task_type_gpu,
                        devices=devices,
                        loss_function=loss_function,
                        eval_metric=eval_metric,
                        verbose=100,  # Progress logging
                        l2_leaf_reg=l2_leaf_reg,
                        bootstrap_type=bootstrap_type,
                        subsample=subsample if bootstrap_type in ['Bernoulli', 'Poisson'] else None,
                    )
            else:
                loss_fn = str(getattr(self.config.features, 'stage3_catboost_loss_regression', 'RMSE'))
                
                # Suppress GPU memory warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
                    model = CatBoostRegressor(
                        iterations=iterations,
                        learning_rate=learning_rate,
                        depth=depth,
                        random_seed=random_state,
                        task_type=task_type_gpu,
                        devices=devices,
                        loss_function=loss_fn,
                        eval_metric=loss_fn,
                        verbose=100,  # Progress logging
                        l2_leaf_reg=l2_leaf_reg,
                        bootstrap_type=bootstrap_type,
                        subsample=subsample if bootstrap_type in ['Bernoulli', 'Poisson'] else None,
                    )
            
            # Training with validation
            fit_kwargs = {
                'eval_set': [test_pool],
                'use_best_model': True,
                'early_stopping_rounds': early_stopping,
                'plot': False
            }
            
            self._log_info('Starting CatBoost training')
            # Suppress GPU memory warnings during training
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*less than 75% GPU memory available.*')
                model.fit(train_pool, **fit_kwargs)
            
            # Get predictions
            train_pred = model.predict(train_pool)
            test_pred = model.predict(test_pool)
            
            # Get feature importances
            try:
                feature_importances = model.get_feature_importance(data=train_pool, type='PredictionValuesChange')
                importance_dict = dict(zip(selected_features, feature_importances.tolist()))
            except Exception as e:
                self._log_warn('Failed to get PredictionValuesChange importance; trying FeatureImportance', error=str(e))
                try:
                    feature_importances = model.get_feature_importance(type='FeatureImportance')
                    importance_dict = dict(zip(selected_features, feature_importances.tolist()))
                except Exception as e2:
                    self._log_warn('Failed to get feature importances', error=str(e2))
                    importance_dict = {f: 1.0 for f in selected_features}
            
            # Get model info
            try:
                best_iteration = int(model.get_best_iteration())
            except:
                best_iteration = iterations
            
            model_results = {
                'model': model,
                'train_predictions': train_pred,
                'test_predictions': test_pred,
                'feature_importances': importance_dict,
                'model_info': {
                    'task_type': task_type,
                    'iterations_used': best_iteration,
                    'total_iterations': iterations,
                    'learning_rate': learning_rate,
                    'depth': depth,
                    'l2_leaf_reg': l2_leaf_reg,
                    'bootstrap_type': bootstrap_type,
                    'subsample': subsample,
                    'loss_function': loss_function if task_type == 'classification' else loss_fn,
                    'eval_metric': eval_metric if task_type == 'classification' else loss_fn,
                    'early_stopping_used': best_iteration < iterations
                }
            }
            
            self._log_info('CatBoost training completed', 
                           best_iteration=best_iteration,
                           early_stopped=best_iteration < iterations)
            
            # Clean up training pools to free memory
            try:
                del train_pool, test_pool
            except:
                pass
            
            return model_results
            
        except Exception as e:
            self._log_error('CatBoost training failed', 
                           error=str(e), 
                           traceback=_tb.format_exc())
            raise

    def _comprehensive_evaluation(
        self,
        model_results: Dict[str, Any],
        train_test_split: Dict[str, Any],
        task_type: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        frac_diff_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation using both standard and trading metrics."""
        
        self._log_info('Starting comprehensive model evaluation', task_type=task_type)
        
        try:
            y_train = train_test_split['y_train']
            y_test = train_test_split['y_test']
            train_pred = model_results['train_predictions']
            test_pred = model_results['test_predictions']
            
            evaluation_results = {}
            
            if task_type == 'classification':
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                # Training metrics
                evaluation_results.update({
                    'train_accuracy': float(accuracy_score(y_train, train_pred)),
                    'train_precision': float(precision_score(y_train, train_pred, average='weighted', zero_division=0)),
                    'train_recall': float(recall_score(y_train, train_pred, average='weighted', zero_division=0)),
                    'train_f1': float(f1_score(y_train, train_pred, average='weighted', zero_division=0)),
                })
                
                # Test metrics
                evaluation_results.update({
                    'test_accuracy': float(accuracy_score(y_test, test_pred)),
                    'test_precision': float(precision_score(y_test, test_pred, average='weighted', zero_division=0)),
                    'test_recall': float(recall_score(y_test, test_pred, average='weighted', zero_division=0)),
                    'test_f1': float(f1_score(y_test, test_pred, average='weighted', zero_division=0)),
                })
                
                # AUC for binary classification
                if len(np.unique(y_train)) == 2:
                    try:
                        evaluation_results['train_auc'] = float(roc_auc_score(y_train, train_pred))
                        evaluation_results['test_auc'] = float(roc_auc_score(y_test, test_pred))
                    except:
                        pass
                
                # Primary metric for classification
                evaluation_results['train_primary_metric'] = evaluation_results['train_f1']
                evaluation_results['test_primary_metric'] = evaluation_results['test_f1']
                
            else:
                # Regression metrics - Standard
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                # Training metrics
                evaluation_results.update({
                    'train_r2': float(r2_score(y_train, train_pred)),
                    'train_mse': float(mean_squared_error(y_train, train_pred)),
                    'train_mae': float(mean_absolute_error(y_train, train_pred)),
                    'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred)))
                })
                
                # Test metrics
                evaluation_results.update({
                    'test_r2': float(r2_score(y_test, test_pred)),
                    'test_mse': float(mean_squared_error(y_test, test_pred)),
                    'test_mae': float(mean_absolute_error(y_test, test_pred)),
                    'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred)))
                })
                
                symbol_upper = str(symbol or '').upper()
                timeframe_lower = self._resolve_output_timeframe(timeframe)

                # Advanced trading metrics for regression
                try:
                    self._log_info('Starting advanced trading metrics calculation',
                                  symbol=symbol_upper, timeframe=timeframe_lower)
                    
                    # Create fresh TradingMetrics instance to avoid state issues
                    fresh_trading_metrics = TradingMetrics(config=self.config)
                    
                    # Training advanced metrics
                    self._log_info('Computing training advanced metrics',
                                  symbol=symbol_upper, timeframe=timeframe_lower)
                    train_advanced = fresh_trading_metrics.compute_comprehensive_metrics(
                        predictions=train_pred,
                        targets=y_train,
                        prefix="train"
                    )
                    evaluation_results.update(train_advanced)
                    self._log_info('Training advanced metrics completed',
                                  symbol=symbol_upper, timeframe=timeframe_lower, 
                                  metrics_count=len(train_advanced))
                    
                    # Test advanced metrics
                    self._log_info('Computing test advanced metrics',
                                  symbol=symbol_upper, timeframe=timeframe_lower)
                    test_advanced = fresh_trading_metrics.compute_comprehensive_metrics(
                        predictions=test_pred,
                        targets=y_test,
                        prefix="test"
                    )
                    evaluation_results.update(test_advanced)
                    self._log_info('Test advanced metrics completed',
                                  symbol=symbol_upper, timeframe=timeframe_lower,
                                  metrics_count=len(test_advanced))
                    
                    # Log advanced metrics
                    fresh_trading_metrics.log_comprehensive_metrics(train_advanced, "Final Model Training", logger=self.logger)
                    fresh_trading_metrics.log_comprehensive_metrics(test_advanced, "Final Model Testing", logger=self.logger)
                    
                    self._log_info('Advanced trading metrics logging completed',
                                  symbol=symbol_upper, timeframe=timeframe_lower)
                    
                    # Primary metric: use Skill Score or Information Coefficient if available
                    if 'test_skill_score' in test_advanced:
                        evaluation_results['test_primary_metric'] = float(test_advanced['test_skill_score'])
                    elif 'test_information_coefficient' in test_advanced:
                        evaluation_results['test_primary_metric'] = abs(float(test_advanced['test_information_coefficient']))
                    else:
                        evaluation_results['test_primary_metric'] = evaluation_results['test_r2']
                    
                    if 'train_skill_score' in train_advanced:
                        evaluation_results['train_primary_metric'] = float(train_advanced['train_skill_score'])
                    elif 'train_information_coefficient' in train_advanced:
                        evaluation_results['train_primary_metric'] = abs(float(train_advanced['train_information_coefficient']))
                    else:
                        evaluation_results['train_primary_metric'] = evaluation_results['train_r2']
                        
                except Exception as e:
                    self._log_error('Advanced trading metrics failed; using standard metrics', 
                                  error=str(e),
                                  error_type=type(e).__name__,
                                  symbol=symbol_upper,
                                  timeframe=timeframe_lower,
                                  traceback=_tb.format_exc())
                    evaluation_results['train_primary_metric'] = evaluation_results['train_r2']
                    evaluation_results['test_primary_metric'] = evaluation_results['test_r2']

                try:
                    self._log_info('Starting GPU post-training metrics calculation',
                                  symbol=symbol_upper, timeframe=timeframe_lower)

                    params = self._resolve_timeframe_params(timeframe_lower)
                    cost_per_trade = self._resolve_cost_per_trade(symbol_upper, timeframe_lower)

                    self._log_info('GPU metrics parameters resolved',
                                  symbol=symbol_upper, timeframe=timeframe_lower,
                                  cost_per_trade=cost_per_trade,
                                  annual_factor=params['annual_factor'],
                                  window_size=params['window_size'])

                    gpu_metrics_engine = GPUPostTrainingMetrics(
                        cost_per_trade=cost_per_trade,
                        annual_factor=params['annual_factor'],
                        window_size=params['window_size']
                    )

                    self._log_info('GPU metrics engine created, computing metrics',
                                  symbol=symbol_upper, timeframe=timeframe_lower,
                                  y_test_size=len(y_test), y_pred_size=len(test_pred))

                    gpu_metrics = gpu_metrics_engine.compute_metrics(
                        y_true=y_test,
                        y_pred=test_pred
                    )

                    self._log_info('GPU metrics computation completed',
                                  symbol=symbol_upper, timeframe=timeframe_lower,
                                  has_global=bool(gpu_metrics.get('global')),
                                  has_stability=bool(gpu_metrics.get('stability')),
                                  has_windows=bool(gpu_metrics.get('windows')))

                    evaluation_results['gpu_metrics_global'] = gpu_metrics['global']
                    evaluation_results['gpu_metrics_windows'] = gpu_metrics['windows']
                    evaluation_results['gpu_metrics_stability'] = gpu_metrics['stability']
                    evaluation_results['gpu_metrics_bucket_means'] = gpu_metrics['bucket_means']

                    stability = gpu_metrics['stability']
                    global_gpu = gpu_metrics['global']
                    ic_pct = stability.get('ic_positive_pct', 0.0)
                    sharpe_pct = stability.get('sharpe_positive_pct', 0.0)
                    self.logger.info(
                        "[FinalMetrics] IC=%.4f, ICIR=%.4f, Hit=%.4f, Sharpe(liq)=%.4f, Sortino(liq)=%.4f, "
                        "MDD(liq)=%.6f, Q5-Q1=%.6f, Estab_IC=%.2f%%, Estab_Sharpe=%.2f%%",
                        global_gpu.get('IC', 0.0),
                        global_gpu.get('ICIR', 0.0),
                        global_gpu.get('hit', 0.0),
                        global_gpu.get('sharpe_liq', 0.0),
                        global_gpu.get('sortino_liq', 0.0),
                        global_gpu.get('mdd_liq', 0.0),
                        global_gpu.get('q5_minus_q1', 0.0),
                        ic_pct * 100.0,
                        sharpe_pct * 100.0
                    )
                    self.logger.info(
                        "[FinalMetrics] Z=%.3f, Turnover=%.4f, Trades=%d, TStat(Q5-Q1)=%.3f",
                        global_gpu.get('z_score', 0.0),
                        global_gpu.get('turnover', 0.0),
                        global_gpu.get('trades_total', 0),
                        global_gpu.get('tstat_q5q1', 0.0)
                    )

                    bucket_means = gpu_metrics['bucket_means']
                    if bucket_means:
                        bucket_mono = global_gpu.get('bucket_monotonicity', 0.0)
                        self.logger.info(
                            "[FinalMetrics] BucketMonotonicity(Spearman)=%.3f",
                            bucket_mono
                        )

                    self._log_info('GPU metrics logging completed successfully',
                                  symbol=symbol_upper, timeframe=timeframe_lower,
                                  q5_minus_q1=global_gpu.get('q5_minus_q1', 0.0),
                                  tstat_q5q1=global_gpu.get('tstat_q5q1', 0.0))
                except Exception as gpu_err:
                    self._log_error('GPU post-training metrics failed', 
                                  error=str(gpu_err),
                                  error_type=type(gpu_err).__name__,
                                  symbol=symbol_upper,
                                  timeframe=timeframe_lower,
                                  traceback=_tb.format_exc())

                    try:
                        import gc
                        import cupy as cp
                        gc.collect()
                        cp.get_default_memory_pool().free_all_blocks()
                        self._log_info('GPU memory cleanup attempted after metrics failure')
                    except Exception as cleanup_err:
                        self._log_warn('GPU memory cleanup failed', error=str(cleanup_err))
            
            # Add prediction statistics
            evaluation_results.update({
                'train_pred_mean': float(np.mean(train_pred)),
                'train_pred_std': float(np.std(train_pred)),
                'test_pred_mean': float(np.mean(test_pred)),
                'test_pred_std': float(np.std(test_pred)),
                'train_target_mean': float(np.mean(y_train)),
                'train_target_std': float(np.std(y_train)),
                'test_target_mean': float(np.mean(y_test)),
                'test_target_std': float(np.std(y_test))
            })

            if frac_diff_details:
                evaluation_results['frac_diff_details'] = frac_diff_details

            self._log_info('Comprehensive evaluation completed', 
                           train_primary=evaluation_results['train_primary_metric'],
                           test_primary=evaluation_results['test_primary_metric'])
            
            return evaluation_results
            
        except Exception as e:
            self._log_error('Comprehensive evaluation failed', 
                           error=str(e), 
                           traceback=_tb.format_exc())
            raise

    def _resolve_timeframe_params(self, timeframe: Optional[str]) -> Dict[str, float]:
        """Infer window size and annualization factor from timeframe string."""
        timeframe_str = str(timeframe or '').lower()
        bars_per_day = 1
        try:
            if timeframe_str.endswith('m'):
                minutes = max(int(timeframe_str[:-1] or 1), 1)
                bars_per_day = max(int((24 * 60) / minutes), 1)
            elif timeframe_str.endswith('h'):
                hours = max(int(timeframe_str[:-1] or 1), 1)
                bars_per_day = max(int(24 / hours), 1)
        except Exception:
            bars_per_day = 1

        window_size = bars_per_day * 20
        annual_factor = float((252 * bars_per_day) ** 0.5)
        return {
            'bars_per_day': float(bars_per_day),
            'window_size': int(max(window_size, 1)),
            'annual_factor': annual_factor
        }

    def _resolve_cost_per_trade(self, symbol: Optional[str], timeframe: Optional[str]) -> float:
        """Resolve cost per trade from configuration with sensible fallbacks."""
        symbol = (symbol or '').upper()
        timeframe_key = str(timeframe or '').lower()
        candidate_keys = [
            f"{symbol}_{timeframe_key}",
            symbol,
            timeframe_key,
            'default'
        ]

        sources = []
        try:
            features_attr = getattr(self.config.features, 'final_metrics_cost_per_trade', None)
            sources.append(features_attr)
        except Exception:
            pass

        try:
            trading_cfg = getattr(self.config, 'trading', None)
            if trading_cfg is not None:
                sources.append(getattr(trading_cfg, 'cost_per_trade', None))
        except Exception:
            pass

        for source in sources:
            value = self._extract_cost_from_container(source, candidate_keys)
            if value is not None:
                return value

        return 0.0

    def _extract_cost_from_container(self, container: Any, keys: List[str]) -> Optional[float]:
        """Attempt to extract a float cost from various container types."""
        if container is None:
            return None
        try:
            if isinstance(container, (int, float)):
                return float(container)
            if isinstance(container, dict):
                for key in keys:
                    if key in container:
                        return float(container[key])
            if hasattr(container, '__dict__'):
                for key in keys:
                    if hasattr(container, key):
                        return float(getattr(container, key))
        except Exception:
            return None
        return None

    def _resolve_output_timeframe(self, timeframe: Optional[str]) -> str:
        """Resolve timeframe used for artifact naming and uploads."""
        timeframe_str = str(timeframe or '').strip().lower()
        if timeframe_str and timeframe_str != 'unknown':
            return timeframe_str

        default_target = str(getattr(self.config.features, 'selection_target_column', '') or '').lower()
        if default_target:
            parts = default_target.split('_')
            candidate = parts[-1] if parts else default_target
            if candidate and candidate[-1] in ('m', 'h', 'd'):
                return candidate

        return '60m'

    def _extract_frac_diff_details(self, X_df, selected_features: List[str]) -> Dict[str, Any]:
        """Extract fractional differentiation metadata for selected features."""
        details: Dict[str, Any] = {}

        if X_df is None or not hasattr(X_df, 'columns'):
            return details

        for feature in selected_features:
            if not feature or 'frac_diff' not in feature:
                continue

            d_column = f"{feature}_d"
            if d_column not in X_df.columns:
                continue

            try:
                d_series = X_df[d_column].dropna()
                if len(d_series) == 0:
                    continue

                first_value = d_series.iloc[0]
                d_value = float(first_value)

                unique_values: List[float] = []
                try:
                    uniques = d_series.unique()
                    if hasattr(uniques, 'to_pandas'):
                        uniques = uniques.to_pandas()
                    unique_values = [float(v) for v in list(uniques) if v is not None]
                except Exception:
                    unique_values = [d_value]

                details[feature] = {
                    'd': d_value,
                    'd_column': d_column,
                    'unique_d_values': unique_values,
                }
            except Exception as err:
                self._log_warn('Failed to extract frac_diff value for feature',
                               feature=feature,
                               error=str(err))

        return details

    # Database methods removed - using JSON metadata in R2 instead
    
    def _upload_model_to_r2(self,
                           symbol: str,
                           timeframe: str,
                           selected_features: List[str],
                           feature_importances: Dict[str, float],
                           model_results: Dict[str, Any],
                           evaluation_results: Dict[str, Any],
                           task_type: str) -> bool:
        """
        Upload the trained model to R2 cloud storage with metadata and cleanup.
        
        Args:
            symbol: Currency pair symbol
            timeframe: Trading timeframe
            selected_features: List of selected features
            feature_importances: Feature importance scores
            model_results: Model training results
            evaluation_results: Model evaluation results
            task_type: 'regression' or 'classification'
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            self._log_info('Starting model upload to R2', 
                          symbol=symbol, 
                          timeframe=timeframe)
            
            # 1. Resolve naming components
            resolved_timeframe = self._resolve_output_timeframe(timeframe)
            symbol_upper = str(symbol or '').upper()
            symbol_lower = symbol_upper.lower()
            model_name_base = f"catboost_{symbol_lower}_{resolved_timeframe}"
            # Simple version using timestamp since no database tracking
            model_version = int(datetime.now().timestamp()) % 10000  # Simple version number
            
            # 2. Save model to temporary file
            import tempfile
            import os
            
            temp_dir = tempfile.mkdtemp()
            model_file_path = os.path.join(temp_dir, f"{model_name_base}_v{model_version}.cbm")
            
            try:
                # Save CatBoost model
                model_results['model'].save_model(model_file_path)
                self._log_info('Model saved to temporary file', file_path=model_file_path)
                
            except Exception as e:
                self._log_error('Failed to save model to file', 
                               file_path=model_file_path, 
                               error=str(e))
                return False
            
            # 3. Prepare model information
            model_info = {
                'model_name': f"{model_name_base}_v{model_version}",
                'model_version': model_version,
                'symbol': symbol_upper,
                'timeframe': resolved_timeframe,
                'task_type': task_type,
                'created_at': datetime.now().isoformat(),
                
                # Training configuration
                'iterations': model_results['model_info']['iterations_used'],
                'learning_rate': model_results['model_info']['learning_rate'],
                'depth': model_results['model_info']['depth'],
                'l2_leaf_reg': model_results['model_info']['l2_leaf_reg'],
                'bootstrap_type': model_results['model_info']['bootstrap_type'],
                'subsample': model_results['model_info']['subsample'],
                'random_seed': 42,
                
                # Data information
                'train_samples': evaluation_results.get('train_samples'),
                'test_samples': evaluation_results.get('test_samples'),
                'vol_scaling_enabled': getattr(self.config.features, 'enable_vol_scaling', False)
            }
            
            # 4. Upload to R2
            upload_success = self.r2_uploader.upload_model(
                model_file_path=model_file_path,
                model_info=model_info,
                features=selected_features,
                feature_importances=feature_importances,
                metrics=evaluation_results,
                cleanup_local=True  # This will delete the temporary files after upload
            )
            
            # 5. Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
                self._log_info('Temporary directory cleaned up', temp_dir=temp_dir)
            except Exception as e:
                self._log_warn('Failed to cleanup temporary directory', 
                              temp_dir=temp_dir, 
                              error=str(e))
            
            if upload_success:
                self._log_info('Model successfully uploaded to R2', 
                              model_name=model_info['model_name'],
                              symbol=symbol_upper,
                              r2_path=f"models/{symbol_upper}/{model_info['model_name']}.cbm")
                
                # Clean up GPU memory after successful upload
                self._cleanup_model_memory(model_results, symbol, timeframe)
                return True
            else:
                self._log_error('Failed to upload model to R2', 
                               model_name=model_info['model_name'],
                               symbol=symbol)
                
                # Clean up GPU memory even on upload failure
                self._cleanup_model_memory(model_results, symbol, timeframe)
                return False
            
        except Exception as e:
            self._log_error('R2 upload process failed', 
                           symbol=symbol, 
                           error=str(e),
                           traceback=_tb.format_exc())
            
            # Clean up GPU memory even on exception
            try:
                self._cleanup_model_memory(model_results, symbol, timeframe)
            except Exception as cleanup_err:
                self._log_warn('Memory cleanup failed after R2 upload exception', 
                              error=str(cleanup_err))
            
            return False
