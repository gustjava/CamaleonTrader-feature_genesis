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
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback as _tb

from utils.logging_utils import get_logger
from utils.trading_metrics import TradingMetrics
from data_io.db_handler import DatabaseHandler
from data_io.r2_uploader import R2ModelUploader

logger = get_logger(__name__, "features.final_model")


class FinalModelTrainer:
    """Class for training and evaluating the final CatBoost model with selected features."""
    
    def __init__(self, config, logger_instance=None):
        """Initialize the final model trainer."""
        self.config = config
        self.logger = logger_instance or logger
        self.trading_metrics = TradingMetrics(config)
        self.db_handler = DatabaseHandler()
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
    
    def _extract_fracdiff_parameters(self, selected_features: List[str], fracdiff_optimal_d: Dict[str, float] = None, X_df=None) -> Dict[str, float]:
        """Extract frac_diff optimal d parameters from stationarization metrics or DataFrame columns."""
        try:
            # Use provided parameters if available
            if fracdiff_optimal_d:
                self._log_info(f"Using provided fracdiff parameters: {fracdiff_optimal_d}")
                return fracdiff_optimal_d
            
            # Try to extract from DataFrame columns if available
            if X_df is not None:
                fracdiff_params = {}
                frac_diff_d_columns = [f for f in X_df.columns if f.startswith('frac_diff_') and f.endswith('_d')]
                
                if frac_diff_d_columns:
                    self._log_info(f"Found {len(frac_diff_d_columns)} frac_diff_d columns in DataFrame")
                    
                    for d_col in frac_diff_d_columns:
                        try:
                            # Extract the base column name
                            base_col = d_col.replace('frac_diff_', '').replace('_d', '')
                            
                            # Get the d value from the DataFrame
                            if hasattr(X_df, 'compute'):  # dask_cudf
                                d_value = X_df[d_col].compute().iloc[0]
                            else:  # cudf
                                d_value = X_df[d_col].iloc[0]
                            
                            # Convert to float if needed
                            if hasattr(d_value, 'item'):
                                d_value = d_value.item()
                            
                            fracdiff_params[base_col] = float(d_value)
                            self._log_info(f"Extracted d parameter for {base_col}: {d_value}")
                            
                        except Exception as e:
                            self._log_warn(f"Could not extract d parameter from {d_col}: {e}")
                            continue
                    
                    if fracdiff_params:
                        self._log_info(f"Successfully extracted fracdiff parameters from DataFrame: {fracdiff_params}")
                        return fracdiff_params
            
            # Fallback: extract from selected features
            fracdiff_params = {}
            frac_diff_features = [f for f in selected_features if f.startswith('frac_diff_') and not f.endswith('_d')]
            
            if frac_diff_features:
                self._log_info(f"Found {len(frac_diff_features)} frac_diff features in selected features")
                
                # Use default values for all frac_diff features
                for feature in frac_diff_features:
                    base_col = feature.replace('frac_diff_', '')
                    fracdiff_params[base_col] = 0.5  # Default value
                    
                self._log_info(f"Using default fracdiff parameters: {fracdiff_params}")
            else:
                self._log_info("No frac_diff features found in selected features")
                
            return fracdiff_params
            
        except Exception as e:
            self._log_error(f"Error extracting fracdiff parameters: {e}")
            return {}

    def build_and_evaluate_final_model(self, 
                                     X_df,  # Accept both cudf.DataFrame and dask_cudf.DataFrame
                                     y_series,  # Accept both cudf.Series and dask_cudf.Series
                                     selected_features: List[str], 
                                     feature_importances: Dict[str, float],
                                     selection_metadata: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str,
                                     fracdiff_optimal_d: Dict[str, float] = None) -> Dict[str, Any]:
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

        try:
            import cudf
            if not hasattr(X_df, 'to_cupy'):
                X_df = cudf.DataFrame.from_pandas(X_df)
            if not hasattr(y_series, 'to_cupy'):
                y_series = cudf.Series(y_series)
        except Exception:
            # If cudf unavailable (unexpected), proceed with existing types
            pass
        
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
            
            # 5. Comprehensive evaluation
            evaluation_results = self._comprehensive_evaluation(
                model_results, train_test_split, task_type
            )
            
            # 6. Save to database
            db_record_id = self._save_to_database(
                symbol=symbol,
                timeframe=timeframe,
                selected_features=selected_features,
                feature_importances=feature_importances,
                selection_metadata=selection_metadata,
                model_results=model_results,
                evaluation_results=evaluation_results,
                task_type=task_type
            )
            
            # 7. Upload model to R2 cloud storage
            r2_upload_success = self._upload_model_to_r2(
                symbol=symbol,
                timeframe=timeframe,
                selected_features=selected_features,
                feature_importances=feature_importances,
                model_results=model_results,
                evaluation_results=evaluation_results,
                task_type=task_type,
                db_record_id=db_record_id
            )
            
            # 8. Compile final results
            final_results = {
                'database_record_id': db_record_id,
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
                           database_id=db_record_id,
                           train_score=evaluation_results.get('train_primary_metric', 0.0),
                           test_score=evaluation_results.get('test_primary_metric', 0.0))
            
            return final_results
            
        except Exception as e:
            self._log_error('Final model training failed', 
                           error=str(e), 
                           error_type=type(e).__name__,
                           traceback=_tb.format_exc())
            raise

    def _determine_task_type(self, y_series) -> str:
        """Determine if this is a regression or classification task."""
        try:
            if hasattr(y_series, 'to_cupy'):
                import cupy as cp
                y_gpu = y_series.to_cupy()
                finite_mask = cp.isfinite(y_gpu)
                y_gpu_finite = y_gpu[finite_mask]
                uniq_gpu = cp.unique(y_gpu_finite)
                unique_values = int(uniq_gpu.shape[0])
                uniq_cpu = cp.asnumpy(uniq_gpu)
                y_std_val = float(cp.std(y_gpu_finite)) if y_gpu_finite.size > 0 else 0.0
                total_samples = int(y_gpu_finite.size)
            else:
                y_vals = np.asarray(y_series)
                finite_mask = np.isfinite(y_vals)
                y_vals_finite = y_vals[finite_mask]
                uniq_cpu = np.unique(y_vals_finite)
                unique_values = len(uniq_cpu)
                y_std_val = float(np.std(y_vals_finite)) if y_vals_finite.size > 0 else 0.0
                total_samples = int(y_vals_finite.size)

            max_classes = int(getattr(self.config.features, 'stage3_classification_max_classes', 10))

            if unique_values <= max_classes and (total_samples > 0 and unique_values <= max(2, int(total_samples * 0.1))):
                task_type = 'classification'
                self._log_info('Task determined as classification', 
                              unique_values=unique_values, 
                              max_classes=max_classes)
            else:
                task_type = 'regression'
                self._log_info('Task determined as regression', 
                              unique_values=unique_values,
                              y_std=y_std_val)
            
            return task_type
            
        except Exception as e:
            self._log_warn('Failed to determine task type; defaulting to regression', error=str(e))
            return 'regression'

    def _create_time_series_split(self, X_df, y_series, test_ratio: float = 0.2) -> Dict[str, Any]:
        """Create time-series aware train/test split."""
        n_samples = len(X_df)
        split_idx = int(n_samples * (1 - test_ratio))
        
        feature_names = list(X_df.columns) if hasattr(X_df, 'columns') else list(range(X_df.shape[1]))

        if hasattr(X_df, 'to_cupy') and hasattr(y_series, 'to_cupy'):
            import cupy as cp
            X_gpu = X_df.to_cupy()
            y_gpu = y_series.to_cupy()

            X_train_gpu = X_gpu[:split_idx]
            X_test_gpu = X_gpu[split_idx:]
            y_train_gpu = y_gpu[:split_idx]
            y_test_gpu = y_gpu[split_idx:]

            X_train = cp.asnumpy(X_train_gpu)
            X_test = cp.asnumpy(X_test_gpu)
            y_train = cp.asnumpy(y_train_gpu)
            y_test = cp.asnumpy(y_test_gpu)
        else:
            X_train = X_df.iloc[:split_idx].to_numpy(dtype=np.float32)
            X_test = X_df.iloc[split_idx:].to_numpy(dtype=np.float32)
            y_train = y_series.iloc[:split_idx].to_numpy(dtype=np.float32)
            y_test = y_series.iloc[split_idx:].to_numpy(dtype=np.float32)
        
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
            
            return model_results
            
        except Exception as e:
            self._log_error('CatBoost training failed', 
                           error=str(e), 
                           traceback=_tb.format_exc())
            raise

    def _comprehensive_evaluation(self, 
                                model_results: Dict[str, Any], 
                                train_test_split: Dict[str, Any],
                                task_type: str) -> Dict[str, Any]:
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
                xp = np
                use_gpu_metrics = False
                try:
                    import cupy as cp
                    y_train_gpu = cp.asarray(y_train)
                    y_test_gpu = cp.asarray(y_test)
                    train_pred_gpu = cp.asarray(train_pred)
                    test_pred_gpu = cp.asarray(test_pred)
                    xp = cp
                    use_gpu_metrics = True
                except Exception:
                    y_train_gpu = y_train
                    y_test_gpu = y_test
                    train_pred_gpu = train_pred
                    test_pred_gpu = test_pred

                def _mse(target, pred):
                    diff = target - pred
                    return float(xp.mean(diff * diff))

                def _mae(target, pred):
                    return float(xp.mean(xp.abs(target - pred)))

                def _r2(target, pred):
                    ss_res = xp.sum((target - pred) ** 2)
                    ss_tot = xp.sum((target - xp.mean(target)) ** 2)
                    if float(ss_tot) <= 0.0:
                        return 0.0
                    return float(1.0 - (ss_res / ss_tot))

                def _rmse(target, pred):
                    return float(xp.sqrt(xp.mean((target - pred) ** 2)))

                if use_gpu_metrics:
                    train_r2 = _r2(y_train_gpu, train_pred_gpu)
                    test_r2 = _r2(y_test_gpu, test_pred_gpu)
                    train_mse = _mse(y_train_gpu, train_pred_gpu)
                    test_mse = _mse(y_test_gpu, test_pred_gpu)
                    train_mae = _mae(y_train_gpu, train_pred_gpu)
                    test_mae = _mae(y_test_gpu, test_pred_gpu)
                    train_rmse = _rmse(y_train_gpu, train_pred_gpu)
                    test_rmse = _rmse(y_test_gpu, test_pred_gpu)
                else:
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    train_r2 = float(r2_score(y_train, train_pred))
                    test_r2 = float(r2_score(y_test, test_pred))
                    train_mse = float(mean_squared_error(y_train, train_pred))
                    test_mse = float(mean_squared_error(y_test, test_pred))
                    train_mae = float(mean_absolute_error(y_train, train_pred))
                    test_mae = float(mean_absolute_error(y_test, test_pred))
                    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_pred)))
                    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

                evaluation_results.update({
                    'train_r2': train_r2,
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'train_rmse': train_rmse,
                })

                evaluation_results.update({
                    'test_r2': test_r2,
                    'test_mse': test_mse,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                })
                
                # Advanced trading metrics for regression
                try:
                    # Training advanced metrics
                    train_advanced = self.trading_metrics.compute_comprehensive_metrics(
                        y_true=y_train,
                        y_pred=train_pred,
                        fold_id="train"
                    )
                    # Add prefix to all keys
                    train_advanced = {f"train_{k}": v for k, v in train_advanced.items()}
                    evaluation_results.update(train_advanced)
                    
                    # Test advanced metrics
                    test_advanced = self.trading_metrics.compute_comprehensive_metrics(
                        y_true=y_test,
                        y_pred=test_pred,
                        fold_id="test"
                    )
                    # Add prefix to all keys
                    test_advanced = {f"test_{k}": v for k, v in test_advanced.items()}
                    evaluation_results.update(test_advanced)
                    
                    # Log advanced metrics
                    from utils.trading_metrics import log_comprehensive_metrics
                    log_comprehensive_metrics(train_advanced, self.logger, "Final Model Training")
                    log_comprehensive_metrics(test_advanced, self.logger, "Final Model Testing")
                    
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
                    self._log_warn('Advanced trading metrics failed; using standard metrics', error=str(e))
                    evaluation_results['train_primary_metric'] = evaluation_results['train_r2']
                    evaluation_results['test_primary_metric'] = evaluation_results['test_r2']
            
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
            
            self._log_info('Comprehensive evaluation completed', 
                           train_primary=evaluation_results['train_primary_metric'],
                           test_primary=evaluation_results['test_primary_metric'])
            
            return evaluation_results
            
        except Exception as e:
            self._log_error('Comprehensive evaluation failed', 
                           error=str(e), 
                           traceback=_tb.format_exc())
            raise

    def _save_to_database(self, 
                         symbol: str,
                         timeframe: str,
                         selected_features: List[str],
                         feature_importances: Dict[str, float],
                         selection_metadata: Dict[str, Any],
                         model_results: Dict[str, Any],
                         evaluation_results: Dict[str, Any],
                         task_type: str) -> int:
        """Save all results to the database and return the record ID."""
        
        self._log_info('Saving results to database', symbol=symbol, timeframe=timeframe)
        
        try:
            # Prepare data for database
            timestamp = datetime.utcnow()
            
            # Create model name with symbol prefix
            model_name = f"catboost_{symbol.lower()}_{timeframe}"
            model_version = self._get_next_model_version(symbol, timeframe)
            
            # Main model record
            model_record = {
                'model_name': model_name,
                'model_version': model_version,
                'symbol': symbol,
                'timeframe': timeframe,
                'task_type': task_type,
                'feature_count': len(selected_features),
                'training_samples': len(model_results['train_predictions']),
                'test_samples': len(model_results['test_predictions']),
                'train_score': evaluation_results['train_primary_metric'],
                'test_score': evaluation_results['test_primary_metric'],
                'model_config': json.dumps(model_results['model_info']),
                'selection_metadata': json.dumps(selection_metadata),
                'evaluation_metrics': json.dumps(evaluation_results),
                'is_active': True,  # Mark as active model for this symbol
                'created_at': timestamp,
                'updated_at': timestamp
            }
            
            # Insert main record
            with self.db_handler.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                self._create_database_tables(cursor)
                
                # Deactivate previous models for this symbol/timeframe
                deactivate_sql = """
                UPDATE final_models 
                SET is_active = FALSE, updated_at = %s 
                WHERE symbol = %s AND timeframe = %s AND is_active = TRUE
                """
                cursor.execute(deactivate_sql, (timestamp, symbol, timeframe))
                
                # Insert main model record
                insert_model_sql = """
                INSERT INTO final_models (
                    model_name, model_version, symbol, timeframe, task_type, feature_count, 
                    training_samples, test_samples, train_score, test_score, model_config, 
                    selection_metadata, evaluation_metrics, is_active, created_at, updated_at
                ) VALUES (
                    %(model_name)s, %(model_version)s, %(symbol)s, %(timeframe)s, %(task_type)s, %(feature_count)s,
                    %(training_samples)s, %(test_samples)s, %(train_score)s, %(test_score)s, %(model_config)s,
                    %(selection_metadata)s, %(evaluation_metrics)s, %(is_active)s, %(created_at)s, %(updated_at)s
                )
                """
                
                cursor.execute(insert_model_sql, model_record)
                model_id = cursor.lastrowid
                
                self._log_info('Model record saved', 
                               model_id=model_id, 
                               model_name=model_name, 
                               version=model_version)
                
                # Insert selected features with their importances
                for rank, feature_name in enumerate(selected_features, 1):
                    importance_from_selection = feature_importances.get(feature_name, 0.0)
                    importance_from_final = model_results['feature_importances'].get(feature_name, 0.0)
                    
                    # Calculate final ranking based on final importance
                    final_ranking = sorted(model_results['feature_importances'].items(), 
                                         key=lambda x: x[1], reverse=True)
                    try:
                        rank_final = next(i for i, (fname, _) in enumerate(final_ranking, 1) 
                                        if fname == feature_name)
                    except StopIteration:
                        rank_final = rank
                    
                    feature_record = {
                        'model_id': model_id,
                        'feature_name': feature_name,
                        'selection_importance': importance_from_selection,
                        'final_importance': importance_from_final,
                        'rank_selection': rank,
                        'rank_final': rank_final,
                        'created_at': timestamp
                    }
                    
                    insert_feature_sql = """
                    INSERT INTO model_features (
                        model_id, feature_name, selection_importance, final_importance,
                        rank_selection, rank_final, created_at
                    ) VALUES (
                        %(model_id)s, %(feature_name)s, %(selection_importance)s, %(final_importance)s,
                        %(rank_selection)s, %(rank_final)s, %(created_at)s
                    )
                    """
                    
                    cursor.execute(insert_feature_sql, feature_record)
                
                # Insert detailed metrics
                metrics_saved = 0
                for metric_name, metric_value in evaluation_results.items():
                    if isinstance(metric_value, (int, float)) and np.isfinite(metric_value):
                        metric_record = {
                            'model_id': model_id,
                            'metric_name': metric_name,
                            'metric_value': float(metric_value),
                            'metric_category': self._get_metric_category(metric_name),
                            'created_at': timestamp
                        }
                        
                        insert_metric_sql = """
                        INSERT INTO model_metrics (
                            model_id, metric_name, metric_value, metric_category, created_at
                        ) VALUES (
                            %(model_id)s, %(metric_name)s, %(metric_value)s, %(metric_category)s, %(created_at)s
                        )
                        """
                        
                        cursor.execute(insert_metric_sql, metric_record)
                        metrics_saved += 1
                
                conn.commit()
                
                self._log_info('Results saved to database successfully', 
                               model_id=model_id,
                               model_name=model_name,
                               features_saved=len(selected_features),
                               metrics_saved=metrics_saved)
                
                return model_id
                
        except Exception as e:
            self._log_error('Failed to save results to database', 
                           error=str(e), 
                           traceback=_tb.format_exc())
            raise

    def _get_next_model_version(self, symbol: str, timeframe: str) -> int:
        """Get the next version number for this symbol/timeframe combination."""
        try:
            with self.db_handler.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COALESCE(MAX(model_version), 0) + 1 FROM final_models WHERE symbol = %s AND timeframe = %s",
                    (symbol, timeframe)
                )
                result = cursor.fetchone()
                return result[0] if result else 1
        except Exception as e:
            self._log_warn('Failed to get next model version; using 1', error=str(e))
            return 1

    def _get_metric_category(self, metric_name: str) -> str:
        """Categorize metrics for better organization."""
        if 'train_' in metric_name:
            return 'training'
        elif 'test_' in metric_name:
            return 'testing'
        elif 'skill' in metric_name.lower():
            return 'trading'
        elif 'information_coefficient' in metric_name.lower() or 'ic_' in metric_name.lower():
            return 'trading'
        elif 'diebold' in metric_name.lower():
            return 'statistical'
        elif any(x in metric_name.lower() for x in ['r2', 'mse', 'mae', 'rmse']):
            return 'regression'
        elif any(x in metric_name.lower() for x in ['accuracy', 'precision', 'recall', 'f1', 'auc']):
            return 'classification'
        else:
            return 'other'

    def _create_database_tables(self, cursor):
        """Create database tables if they don't exist."""
        
        # Main models table with model naming and versioning
        create_models_table = """
        CREATE TABLE IF NOT EXISTS final_models (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version INT NOT NULL DEFAULT 1,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            task_type VARCHAR(20) NOT NULL,
            feature_count INT NOT NULL,
            training_samples INT NOT NULL,
            test_samples INT NOT NULL,
            train_score FLOAT NOT NULL,
            test_score FLOAT NOT NULL,
            model_config TEXT,
            selection_metadata TEXT,
            evaluation_metrics TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_model_version (symbol, timeframe, model_version),
            INDEX idx_model_name (model_name),
            INDEX idx_symbol_timeframe (symbol, timeframe),
            INDEX idx_created_at (created_at),
            INDEX idx_test_score (test_score),
            INDEX idx_is_active (is_active)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        # Features table
        create_features_table = """
        CREATE TABLE IF NOT EXISTS model_features (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_id INT NOT NULL,
            feature_name VARCHAR(255) NOT NULL,
            selection_importance FLOAT NOT NULL,
            final_importance FLOAT NOT NULL,
            rank_selection INT NOT NULL,
            rank_final INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES final_models(id) ON DELETE CASCADE,
            INDEX idx_model_id (model_id),
            INDEX idx_feature_name (feature_name),
            INDEX idx_final_importance (final_importance),
            INDEX idx_rank_final (rank_final)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        # Metrics table with categorization
        create_metrics_table = """
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_id INT NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_category VARCHAR(50) DEFAULT 'other',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES final_models(id) ON DELETE CASCADE,
            INDEX idx_model_id (model_id),
            INDEX idx_metric_name (metric_name),
            INDEX idx_metric_value (metric_value),
            INDEX idx_metric_category (metric_category)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
        
        # Model performance summary view (optional)
        create_summary_view = """
        CREATE OR REPLACE VIEW model_performance_summary AS
        SELECT 
            m.id,
            m.model_name,
            m.model_version,
            m.symbol,
            m.timeframe,
            m.task_type,
            m.feature_count,
            m.test_score,
            m.is_active,
            m.created_at,
            GROUP_CONCAT(DISTINCT f.feature_name ORDER BY f.rank_final LIMIT 10) as top_features,
            (SELECT metric_value FROM model_metrics WHERE model_id = m.id AND metric_name = 'test_r2' LIMIT 1) as test_r2,
            (SELECT metric_value FROM model_metrics WHERE model_id = m.id AND metric_name = 'test_skill_score' LIMIT 1) as test_skill_score,
            (SELECT metric_value FROM model_metrics WHERE model_id = m.id AND metric_name = 'test_information_coefficient' LIMIT 1) as test_ic
        FROM final_models m
        LEFT JOIN model_features f ON m.id = f.model_id
        GROUP BY m.id, m.model_name, m.model_version, m.symbol, m.timeframe, m.task_type, m.feature_count, m.test_score, m.is_active, m.created_at
        """
        
        try:
            cursor.execute(create_models_table)
            cursor.execute(create_features_table)
            cursor.execute(create_metrics_table)
            cursor.execute(create_summary_view)
            
            self._log_info('Database tables and views created/verified successfully')
            
        except Exception as e:
            # If view creation fails, continue without it
            if 'view' not in str(e).lower():
                raise
            self._log_warn('Failed to create summary view; continuing without it', error=str(e))
    
    def _upload_model_to_r2(self,
                           symbol: str,
                           timeframe: str,
                           selected_features: List[str],
                           feature_importances: Dict[str, float],
                           model_results: Dict[str, Any],
                           evaluation_results: Dict[str, Any],
                           task_type: str,
                           db_record_id: int) -> bool:
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
            db_record_id: Database record ID
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            self._log_info('Starting model upload to R2', 
                          symbol=symbol, 
                          timeframe=timeframe,
                          db_record_id=db_record_id)
            
            # 1. Generate model name and version
            model_name = f"catboost_{symbol}_{timeframe}"
            model_version = self._get_next_model_version(symbol, timeframe)
            
            # 2. Save model to temporary file
            import tempfile
            import os
            
            temp_dir = tempfile.mkdtemp()
            model_file_path = os.path.join(temp_dir, f"{model_name}_v{model_version}.cbm")
            
            try:
                # Save CatBoost model
                model_results['model'].save_model(model_file_path)
                self._log_info('Model saved to temporary file', file_path=model_file_path)
                
                # Verify model file was created and has content
                if not os.path.exists(model_file_path):
                    self._log_error('Model file was not created', file_path=model_file_path)
                    return False
                
                file_size = os.path.getsize(model_file_path)
                if file_size == 0:
                    self._log_error('Model file is empty', file_path=model_file_path, file_size=file_size)
                    return False
                
                self._log_info('Model file verified', file_path=model_file_path, file_size_bytes=file_size)
                
            except Exception as e:
                self._log_error('Failed to save model to file', 
                               file_path=model_file_path, 
                               error=str(e))
                return False
            
            # 3. Prepare model information
            model_info = {
                'model_name': f"{model_name}_v{model_version}",
                'model_version': model_version,
                'symbol': symbol,
                'timeframe': timeframe,
                'task_type': task_type,
                'db_record_id': db_record_id,
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
                'vol_scaling_enabled': getattr(self.config.features, 'enable_vol_scaling', False),
                
                # Feature engineering parameters
                'fracdiff_optimal_d': self._extract_fracdiff_parameters(selected_features, fracdiff_optimal_d, X_df)
            }
            
            # Log fracdiff parameters specifically
            fracdiff_params = model_info.get('fracdiff_optimal_d', {})
            if fracdiff_params:
                self._log_info(f"🔧 FracDiff parameters included in R2 JSON:")
                for col, d_value in fracdiff_params.items():
                    self._log_info(f"  - {col}: d={d_value}")
            else:
                self._log_info("⚠️  No FracDiff parameters found for R2 JSON")
            
            # 4. Log complete JSON metadata before upload
            self._log_info("📋 Complete R2 JSON metadata:")
            try:
                import json as _json
                json_str = _json.dumps(model_info, indent=2, ensure_ascii=False)
                # Log in chunks to avoid truncation
                lines = json_str.split('\n')
                for i, line in enumerate(lines):
                    self._log_info(f"JSON[{i+1:3d}]: {line}")
                
                # Verify JSON is valid
                _json.loads(json_str)
                self._log_info("✅ JSON metadata validation passed")
                
            except Exception as e:
                self._log_warn(f"Could not log JSON metadata: {e}")
                # Fallback: log as string
                self._log_info(f"JSON metadata: {model_info}")
            
            # 5. Upload to R2
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
                              symbol=symbol,
                              r2_path=f"models/{symbol}/{model_info['model_name']}.cbm")
                return True
            else:
                self._log_error('Failed to upload model to R2', 
                               model_name=model_info['model_name'],
                               symbol=symbol)
                return False
            
        except Exception as e:
            self._log_error('R2 upload process failed', 
                           symbol=symbol, 
                           error=str(e),
                           traceback=_tb.format_exc())
            return False
