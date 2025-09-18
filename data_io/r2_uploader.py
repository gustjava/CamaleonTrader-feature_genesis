"""
Cloudflare R2 Uploader

Small utility to upload local files to Cloudflare R2 (S3-compatible)
using credentials and endpoint from the unified configuration.
"""

from __future__ import annotations

import os
import logging
from typing import Optional


logger = logging.getLogger(__name__)


class R2Uploader:
    """Uploads files to Cloudflare R2 using boto3 (S3-compatible API)."""

    def __init__(self):
        try:
            from config.unified_config import get_unified_config
            self.settings = get_unified_config()
        except Exception as e:
            raise RuntimeError(f"Failed to load unified config for R2 upload: {e}")

        self.bucket = self.settings.r2.bucket_name
        self.endpoint_url = self.settings.r2.endpoint_url
        self.access_key = self.settings.r2.access_key
        self.secret_key = self.settings.r2.secret_key
        self.region = getattr(self.settings.r2, 'region', 'auto') or 'auto'

        # Lazy client
        self._s3_client = None

    def _client(self):
        if self._s3_client is None:
            import boto3
            self._s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )
        return self._s3_client

    def upload_file(self, local_path: str, remote_key: str, content_type: Optional[str] = None) -> bool:
        """Upload a local file to R2 at the given key.

        Args:
            local_path: path to local file
            remote_key: key inside the bucket (e.g., 'AUDUSD_feature_model.parquet')
            content_type: optional mime type

        Returns:
            True on success, False otherwise
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f"R2 upload: local file does not exist: {local_path}")
                return False
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"R2 upload: {local_path} -> s3://{self.bucket}/{remote_key} ({size_mb:.2f} MB)")

            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            self._client().upload_file(local_path, self.bucket, remote_key, ExtraArgs=extra_args)
            logger.info(f"R2 upload completed: s3://{self.bucket}/{remote_key}")
            return True
        except Exception as e:
            logger.error(f"R2 upload failed for {local_path} -> {remote_key}: {e}")
            return False

