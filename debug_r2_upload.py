#!/usr/bin/env python3
"""
Debug R2 Upload - Teste específico para identificar problema no upload

Execute este script para testar se o problema é:
1. Inicialização do R2ModelUploader
2. Configuração/credenciais
3. Processo de upload
"""

import sys
import os
import tempfile
import traceback

def debug_r2_upload():
    """Debug detalhado do R2 upload."""
    
    print("🔍 DEBUG R2 UPLOAD")
    print("=" * 50)
    
    try:
        # 1. Test imports
        print("📦 Testing imports...")
        from config import get_settings
        print("✅ config imported")
        
        import boto3
        print("✅ boto3 imported")
        
        from data_io.r2_uploader import R2ModelUploader
        print("✅ R2ModelUploader imported")
        
        # 2. Test configuration loading
        print("\n🔧 Testing configuration...")
        config = get_settings()
        r2_config = getattr(config, 'r2', None)
        
        if r2_config is None:
            print("❌ R2 config is None")
            return False
        else:
            print("✅ R2 config loaded")
            
        # Check required fields
        required_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name', 'endpoint_url']
        missing_fields = []
        for field in required_fields:
            value = getattr(r2_config, field, None)
            if not value:
                missing_fields.append(field)
            else:
                display_value = value if 'key' not in field.lower() else "***hidden***"
                print(f"✅ {field}: {display_value}")
        
        if missing_fields:
            print(f"❌ Missing R2 fields: {missing_fields}")
            return False
        
        # 3. Test R2ModelUploader initialization
        print("\n🚀 Testing R2ModelUploader initialization...")
        uploader = R2ModelUploader()
        print("✅ R2ModelUploader initialized")
        
        # Check if S3 client was created
        if uploader.s3_client is None:
            print("❌ S3 client is None after initialization")
            return False
        else:
            print("✅ S3 client created successfully")
        
        # 4. Test credentials validation
        print("\n🔑 Testing credentials validation...")
        creds_valid = uploader._validate_r2_credentials()
        if creds_valid:
            print("✅ Credentials validation passed")
        else:
            print("❌ Credentials validation failed")
            return False
        
        # 5. Test basic connectivity
        print("\n🌐 Testing R2 connectivity...")
        try:
            response = uploader.s3_client.list_objects_v2(
                Bucket=r2_config.bucket_name,
                MaxKeys=1
            )
            print("✅ R2 connectivity test passed")
            
            # Count existing models
            models_response = uploader.s3_client.list_objects_v2(
                Bucket=r2_config.bucket_name,
                Prefix="models/",
                MaxKeys=100
            )
            model_count = models_response.get('KeyCount', 0)
            print(f"📊 Found {model_count} existing model objects")
            
        except Exception as conn_e:
            print(f"❌ R2 connectivity failed: {conn_e}")
            print(f"Error type: {type(conn_e).__name__}")
            return False
        
        # 6. Test file upload (create a dummy model file)
        print("\n📤 Testing file upload...")
        try:
            # Create a temporary file to upload
            with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as temp_file:
                temp_file.write(b"dummy model data for testing")
                temp_file_path = temp_file.name
            
            test_model_info = {
                'model_name': 'test_debug_model',
                'model_version': 1,
                'symbol': 'TEST',
                'timeframe': '1h',
                'task_type': 'regression',
                'db_record_id': 99999,
                'created_at': '2024-01-01T00:00:00',
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 10.0,
                'bootstrap_type': 'Bernoulli',
                'subsample': 0.7,
                'random_seed': 42,
                'train_samples': 1000,
                'test_samples': 200,
                'vol_scaling_enabled': False
            }
            
            # Test upload
            upload_success = uploader.upload_model(
                model_file_path=temp_file_path,
                model_info=test_model_info,
                features=['feature1', 'feature2'],
                feature_importances={'feature1': 0.6, 'feature2': 0.4},
                metrics={'test_r2': 0.5, 'test_rmse': 0.1},
                cleanup_local=True
            )
            
            if upload_success:
                print("✅ Test upload successful!")
                
                # Clean up test model
                try:
                    uploader.delete_model('TEST', 'test_debug_model')
                    print("✅ Test model cleaned up")
                except:
                    print("⚠️ Could not clean up test model (not critical)")
                
            else:
                print("❌ Test upload failed")
                return False
                
        except Exception as upload_e:
            print(f"❌ Upload test failed: {upload_e}")
            print(f"Error type: {type(upload_e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Clean up temp file
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
            except:
                pass
        
        print("\n🎉 All R2 upload tests passed!")
        print("✅ R2 upload should be working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = debug_r2_upload()
    
    if success:
        print("\n💡 Conclusion: R2 upload is working. Check for:")
        print("   - Model file creation issues")
        print("   - Exceptions in the main pipeline")
        print("   - Missing model data in final_model.py")
    else:
        print("\n💡 Conclusion: R2 upload has issues. Check above errors.")
    
    sys.exit(0 if success else 1)
