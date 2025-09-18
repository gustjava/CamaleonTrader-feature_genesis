#!/usr/bin/env python3
"""
Simple R2 Test Script

Tests R2 connectivity without GPU dependencies.
Run this from the same environment where the pipeline runs.
"""

import os
import sys
import tempfile

def test_r2_simple():
    """Test R2 connectivity with minimal dependencies."""
    
    print("🧪 Simple R2 Test (No GPU Dependencies)")
    print("=" * 50)
    
    try:
        # Test config loading
        print("📋 Testing configuration loading...")
        
        # Check environment variables
        env_vars = {
            'R2_ACCOUNT_ID': os.getenv('R2_ACCOUNT_ID'),
            'R2_ACCESS_KEY': os.getenv('R2_ACCESS_KEY'),
            'R2_SECRET_KEY': os.getenv('R2_SECRET_KEY'),
            'R2_BUCKET_NAME': os.getenv('R2_BUCKET_NAME'),
            'R2_ENDPOINT_URL': os.getenv('R2_ENDPOINT_URL')
        }
        
        print("🔧 Environment Variables:")
        for key, value in env_vars.items():
            status = "✅" if value else "❌"
            display_value = value if 'KEY' not in key else "***hidden***"
            print(f"{status} {key}: {display_value}")
        
        # Test boto3 import and basic S3 client
        print("\n📦 Testing boto3...")
        import boto3
        from botocore.exceptions import ClientError
        print("✅ boto3 imported successfully")
        
        # Create S3 client
        print("\n🔗 Creating S3 client...")
        account_id = env_vars['R2_ACCOUNT_ID']
        access_key = env_vars['R2_ACCESS_KEY']
        secret_key = env_vars['R2_SECRET_KEY']
        bucket_name = env_vars['R2_BUCKET_NAME']
        endpoint_url = env_vars['R2_ENDPOINT_URL']
        
        if not all([account_id, access_key, secret_key, bucket_name]):
            print("❌ Missing required R2 credentials in environment variables")
            return False
        
        # Fix endpoint URL if needed
        if not endpoint_url or "account_id.r2.cloudflarestorage.com" in endpoint_url:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
            print(f"🔧 Fixed endpoint URL: {endpoint_url}")
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
        print("✅ S3 client created successfully")
        
        # Test connection
        print("\n🌐 Testing R2 connection...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=1
            )
            print("✅ Connection successful!")
            
            # Count existing models
            models_response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix="models/",
                MaxKeys=1000
            )
            
            model_count = models_response.get('KeyCount', 0)
            print(f"📊 Found {model_count} objects in models/ prefix")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"❌ Connection failed: {error_code} - {error_message}")
            return False
        
        # Test upload capability
        print("\n📤 Testing upload capability...")
        test_content = "test-content-from-r2-test"
        test_key = "test/connectivity_test.txt"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            # Upload test file
            s3_client.upload_file(
                temp_file_path,
                bucket_name,
                test_key
            )
            print("✅ Upload test successful!")
            
            # Verify upload
            response = s3_client.head_object(
                Bucket=bucket_name,
                Key=test_key
            )
            file_size = response['ContentLength']
            print(f"✅ Upload verified: {test_key} ({file_size} bytes)")
            
            # Cleanup test file
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=test_key
            )
            print("✅ Test file cleaned up")
            
        except Exception as e:
            print(f"❌ Upload test failed: {e}")
            return False
        finally:
            # Cleanup local temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        print("\n🎉 All R2 tests passed! Upload should work correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_r2_simple()
    sys.exit(0 if success else 1)
