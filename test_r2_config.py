#!/usr/bin/env python3
"""
Quick R2 Configuration Test

Test script to validate R2 connectivity and configuration.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_r2_config():
    """Test R2 configuration and connectivity."""
    
    print("🧪 Testing R2 Configuration...")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from data_io.r2_uploader import R2ModelUploader
        from config import get_config
        print("✅ Imports successful")
        
        # Test configuration loading
        print("\n🔧 Testing configuration...")
        config = get_config()
        
        required_fields = ['account_id', 'access_key', 'secret_key', 'bucket_name', 'endpoint_url']
        for field in required_fields:
            value = getattr(config.r2, field, None)
            status = "✅" if value else "❌"
            display_value = value if field not in ['access_key', 'secret_key'] else "***hidden***"
            print(f"{status} {field}: {display_value}")
        
        # Test R2ModelUploader initialization
        print("\n🚀 Testing R2ModelUploader initialization...")
        uploader = R2ModelUploader()
        print("✅ R2ModelUploader initialized successfully")
        
        # Test connection
        print("\n🌐 Testing R2 connection...")
        models = uploader.list_uploaded_models()
        print(f"✅ Connection successful!")
        print(f"📊 Found {len(models)} existing models in R2")
        
        if models:
            print("\n📋 Sample models:")
            for i, model in enumerate(models[:3], 1):
                size_kb = model['size'] / 1024
                print(f"  {i}. {model['symbol']} - {model['model_name']} ({size_kb:.1f} KB)")
            
            if len(models) > 3:
                print(f"  ... and {len(models) - 3} more models")
        
        print("\n🎉 All tests passed! R2 configuration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(f"📋 Error details:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_r2_config()
    sys.exit(0 if success else 1)
