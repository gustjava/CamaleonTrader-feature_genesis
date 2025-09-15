"""
R2 Model Uploader for CatBoost Models

This module handles uploading trained CatBoost models and metadata
to Cloudflare R2 storage with cleanup functionality.
"""

import json
import logging
import os
import boto3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from botocore.exceptions import ClientError, NoCredentialsError

from config import get_config

logger = logging.getLogger(__name__)


class R2ModelUploader:
    """Handles uploading CatBoost models and metadata to Cloudflare R2."""
    
    def __init__(self):
        """Initialize the R2 model uploader with configuration."""
        self.config = get_config()
        self.s3_client = None
        self._initialize_s3_client()
        
    def _initialize_s3_client(self) -> None:
        """Initialize S3 client for R2 connectivity."""
        try:
            r2_config = self.config.r2
            
            self.s3_client = boto3.client(
                's3',
                endpoint_url=r2_config['endpoint_url'],
                aws_access_key_id=r2_config['access_key'],
                aws_secret_access_key=r2_config['secret_key'],
                region_name=r2_config['region']
            )
            
            # Test connection
            self.s3_client.list_objects_v2(
                Bucket=r2_config['bucket_name'],
                MaxKeys=1
            )
            
            logger.info(f"R2 S3 client initialized successfully for bucket: {r2_config['bucket_name']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize R2 S3 client: {e}")
            self.s3_client = None
            raise
    
    def _validate_r2_credentials(self) -> bool:
        """
        Validate that R2 credentials are properly configured.
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
            
        required_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name']
        
        for field in required_fields:
            if not self.config.r2.get(field):
                logger.error(f"Missing R2 configuration field: {field}")
                return False
        
        logger.debug("R2 credentials validation passed")
        return True
    
    def _generate_model_metadata(
        self,
        model_info: Dict[str, Any],
        features: List[str],
        feature_importances: Dict[str, float],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the model.
        
        Args:
            model_info: Model information (name, version, symbol, etc.)
            features: List of selected features
            feature_importances: Feature importance scores
            metrics: Model evaluation metrics
            
        Returns:
            Dict[str, Any]: Complete model metadata
        """
        metadata = {
            # Model identification
            "model_name": model_info["model_name"],
            "model_version": model_info["model_version"],
            "symbol": model_info["symbol"],
            "timeframe": model_info.get("timeframe", "1H"),
            "created_at": datetime.now().isoformat(),
            
            # Model configuration
            "model_type": "CatBoost",
            "training_config": {
                "iterations": model_info.get("iterations", 750),
                "learning_rate": model_info.get("learning_rate", 0.025),
                "depth": model_info.get("depth", 6),
                "l2_leaf_reg": model_info.get("l2_leaf_reg", 10),
                "bootstrap_type": model_info.get("bootstrap_type", "Bernoulli"),
                "subsample": model_info.get("subsample", 0.7),
                "random_seed": model_info.get("random_seed", 42),
                "task_type": "GPU",
                "loss_function": "RMSE"
            },
            
            # Features and importances
            "features": {
                "selected_features": features,
                "num_features": len(features),
                "feature_importances": feature_importances,
                "top_10_features": sorted(
                    feature_importances.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            },
            
            # Model performance
            "evaluation_metrics": metrics,
            
            # Data characteristics
            "data_info": {
                "train_samples": model_info.get("train_samples"),
                "test_samples": model_info.get("test_samples"),
                "train_period": model_info.get("train_period"),
                "test_period": model_info.get("test_period"),
                "vol_scaling_enabled": model_info.get("vol_scaling_enabled", False)
            },
            
            # Technical metadata
            "file_info": {
                "model_file": f"{model_info['model_name']}.cbm",
                "metadata_file": f"{model_info['model_name']}_metadata.json",
                "upload_timestamp": datetime.now().isoformat(),
                "r2_path": f"models/{model_info['symbol']}/",
                "size_bytes": None  # Will be filled after upload
            }
        }
        
        return metadata
    
    def _upload_file_to_r2(
        self,
        local_file_path: str,
        r2_key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upload a file to R2 storage.
        
        Args:
            local_file_path: Path to the local file to upload
            r2_key: Key (path) in R2 where the file will be stored
            metadata: Optional metadata to attach to the file
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            if not self._validate_r2_credentials():
                return False
            
            bucket_name = self.config.r2['bucket_name']
            
            # Prepare upload arguments
            upload_args = {}
            if metadata:
                upload_args['Metadata'] = {
                    k: str(v) for k, v in metadata.items()
                }
            
            # Upload file
            logger.info(f"Uploading {local_file_path} to R2: s3://{bucket_name}/{r2_key}")
            
            self.s3_client.upload_file(
                local_file_path,
                bucket_name,
                r2_key,
                ExtraArgs=upload_args
            )
            
            # Verify upload
            try:
                response = self.s3_client.head_object(
                    Bucket=bucket_name,
                    Key=r2_key
                )
                file_size = response['ContentLength']
                logger.info(f"Successfully uploaded {r2_key} ({file_size} bytes)")
                return True
                
            except ClientError as e:
                logger.error(f"Upload verification failed for {r2_key}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path} to R2: {e}")
            return False
    
    def upload_model(
        self,
        model_file_path: str,
        model_info: Dict[str, Any],
        features: List[str],
        feature_importances: Dict[str, float],
        metrics: Dict[str, Any],
        cleanup_local: bool = True
    ) -> bool:
        """
        Upload a trained CatBoost model to R2 with metadata.
        
        Args:
            model_file_path: Path to the saved CatBoost model file
            model_info: Model information dictionary
            features: List of selected features
            feature_importances: Feature importance scores
            metrics: Model evaluation metrics
            cleanup_local: Whether to delete local files after upload
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            logger.info(f"Starting upload for model: {model_info['model_name']}")
            
            # Validate model file exists
            if not os.path.exists(model_file_path):
                logger.error(f"Model file does not exist: {model_file_path}")
                return False
            
            # Generate metadata
            metadata = self._generate_model_metadata(
                model_info, features, feature_importances, metrics
            )
            
            # Define R2 paths
            symbol = model_info["symbol"]
            model_name = model_info["model_name"]
            
            model_r2_key = f"models/{symbol}/{model_name}.cbm"
            metadata_r2_key = f"models/{symbol}/{model_name}_metadata.json"
            
            # Create temporary metadata file
            metadata_file_path = f"{model_file_path}_metadata.json"
            with open(metadata_file_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Upload model file
            logger.info(f"Uploading model file: {model_file_path}")
            model_upload_success = self._upload_file_to_r2(
                model_file_path,
                model_r2_key,
                metadata={
                    'model_name': model_info['model_name'],
                    'symbol': symbol,
                    'version': str(model_info['model_version']),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            if not model_upload_success:
                logger.error(f"Failed to upload model file: {model_file_path}")
                return False
            
            # Upload metadata file
            logger.info(f"Uploading metadata file: {metadata_file_path}")
            metadata_upload_success = self._upload_file_to_r2(
                metadata_file_path,
                metadata_r2_key,
                metadata={
                    'type': 'model_metadata',
                    'model_name': model_info['model_name'],
                    'symbol': symbol
                }
            )
            
            if not metadata_upload_success:
                logger.error(f"Failed to upload metadata file: {metadata_file_path}")
                return False
            
            # Update metadata with file size information
            bucket_name = self.config.r2['bucket_name']
            try:
                model_response = self.s3_client.head_object(
                    Bucket=bucket_name,
                    Key=model_r2_key
                )
                metadata['file_info']['size_bytes'] = model_response['ContentLength']
                
                # Re-upload updated metadata
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self._upload_file_to_r2(metadata_file_path, metadata_r2_key)
                
            except Exception as e:
                logger.warning(f"Could not update metadata with file size: {e}")
            
            # Cleanup local files if requested
            if cleanup_local:
                try:
                    os.remove(model_file_path)
                    os.remove(metadata_file_path)
                    logger.info(f"Local files cleaned up: {model_file_path}, {metadata_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup local files: {e}")
            
            logger.info(f"Successfully uploaded model {model_info['model_name']} to R2")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model {model_info['model_name']}: {e}")
            return False
    
    def list_uploaded_models(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all uploaded models in R2, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List[Dict[str, Any]]: List of model information
        """
        try:
            if not self._validate_r2_credentials():
                return []
            
            bucket_name = self.config.r2['bucket_name']
            prefix = f"models/{symbol}/" if symbol else "models/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('.cbm'):
                        # Extract model info from key
                        parts = key.split('/')
                        if len(parts) >= 3:
                            symbol = parts[1]
                            model_file = parts[2]
                            model_name = model_file.replace('.cbm', '')
                            
                            models.append({
                                'model_name': model_name,
                                'symbol': symbol,
                                'r2_key': key,
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'metadata_key': key.replace('.cbm', '_metadata.json')
                            })
            
            logger.info(f"Found {len(models)} models in R2" + (f" for symbol {symbol}" if symbol else ""))
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models from R2: {e}")
            return []
    
    def download_model_metadata(self, symbol: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Download and parse model metadata from R2.
        
        Args:
            symbol: Currency pair symbol
            model_name: Name of the model
            
        Returns:
            Optional[Dict[str, Any]]: Model metadata if found, None otherwise
        """
        try:
            if not self._validate_r2_credentials():
                return None
            
            bucket_name = self.config.r2['bucket_name']
            metadata_key = f"models/{symbol}/{model_name}_metadata.json"
            
            response = self.s3_client.get_object(
                Bucket=bucket_name,
                Key=metadata_key
            )
            
            metadata_content = response['Body'].read().decode('utf-8')
            metadata = json.loads(metadata_content)
            
            logger.info(f"Downloaded metadata for model: {model_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to download metadata for {model_name}: {e}")
            return None
    
    def delete_model(self, symbol: str, model_name: str) -> bool:
        """
        Delete a model and its metadata from R2.
        
        Args:
            symbol: Currency pair symbol
            model_name: Name of the model to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if not self._validate_r2_credentials():
                return False
            
            bucket_name = self.config.r2['bucket_name']
            model_key = f"models/{symbol}/{model_name}.cbm"
            metadata_key = f"models/{symbol}/{model_name}_metadata.json"
            
            # Delete model file
            self.s3_client.delete_object(
                Bucket=bucket_name,
                Key=model_key
            )
            
            # Delete metadata file
            self.s3_client.delete_object(
                Bucket=bucket_name,
                Key=metadata_key
            )
            
            logger.info(f"Successfully deleted model {model_name} from R2")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} from R2: {e}")
            return False
