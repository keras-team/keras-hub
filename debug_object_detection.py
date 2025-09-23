#!/usr/bin/env python3
"""
Test script to understand object detection model outputs and investigate export issues.
"""

import keras_hub
import keras
import numpy as np

def test_object_detection_outputs():
    """Test what object detection models output in different modes."""
    
    print("🔍 Testing object detection model outputs...")
    
    # Load a simple object detection model
    model = keras_hub.models.DFineObjectDetector.from_preset(
        "dfine_nano_coco", 
        # Remove NMS post-processing to see raw outputs
        prediction_decoder=None  # This should give us raw logits and boxes
    )
    
    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"📏 Model inputs: {[inp.shape for inp in model.inputs] if hasattr(model, 'inputs') and model.inputs else 'Not built yet'}")
    
    # Create test input
    test_input = np.random.random((1, 640, 640, 3)).astype(np.float32)
    image_shape = np.array([[640, 640]], dtype=np.int32)
    
    print(f"🎯 Test input shapes:")
    print(f"   Images: {test_input.shape}")
    print(f"   Image shape: {image_shape.shape}")
    
    # Test raw outputs (without post-processing)
    print(f"\n🧠 Testing raw model outputs...")
    try:
        # Try dictionary input format
        raw_outputs = model({
            "images": test_input,
            "image_shape": image_shape
        }, training=False)
        
        print(f"✅ Raw outputs (dict input):")
        if isinstance(raw_outputs, dict):
            for key, value in raw_outputs.items():
                print(f"   {key}: {value.shape}")
        else:
            print(f"   Output type: {type(raw_outputs)}")
            if hasattr(raw_outputs, 'shape'):
                print(f"   Output shape: {raw_outputs.shape}")
                
    except Exception as e:
        print(f"❌ Dict input failed: {e}")
        
        # Try single tensor input
        try:
            raw_outputs = model(test_input, training=False)
            print(f"✅ Raw outputs (single tensor input):")
            if isinstance(raw_outputs, dict):
                for key, value in raw_outputs.items():
                    print(f"   {key}: {value.shape}")
            else:
                print(f"   Output type: {type(raw_outputs)}")
                if hasattr(raw_outputs, 'shape'):
                    print(f"   Output shape: {raw_outputs.shape}")
        except Exception as e2:
            print(f"❌ Single tensor input also failed: {e2}")
    
    # Now test with the default post-processing
    print(f"\n🎯 Testing with default NMS post-processing...")
    model_with_nms = keras_hub.models.DFineObjectDetector.from_preset("dfine_nano_coco")
    
    try:
        # Try dictionary input format
        nms_outputs = model_with_nms({
            "images": test_input,
            "image_shape": image_shape
        }, training=False)
        
        print(f"✅ NMS outputs (dict input):")
        if isinstance(nms_outputs, dict):
            for key, value in nms_outputs.items():
                print(f"   {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"   Output type: {type(nms_outputs)}")
            
    except Exception as e:
        print(f"❌ Dict input failed with NMS: {e}")
        
        # Try single tensor input
        try:
            nms_outputs = model_with_nms(test_input, training=False)
            print(f"✅ NMS outputs (single tensor input):")
            if isinstance(nms_outputs, dict):
                for key, value in nms_outputs.items():
                    print(f"   {key}: {value.shape} (dtype: {value.dtype})")
            else:
                print(f"   Output type: {type(nms_outputs)}")
        except Exception as e2:
            print(f"❌ Single tensor input also failed with NMS: {e2}")

def test_export_attempt():
    """Test the current export behavior that's failing."""
    print(f"\n🚀 Testing current export behavior...")
    
    try:
        model = keras_hub.models.DFineObjectDetector.from_preset("dfine_nano_coco")
        
        # Check what the export config expects
        from keras_hub.src.export.base import ExporterRegistry
        config = ExporterRegistry.get_config_for_model(model)
        
        print(f"📋 Export config:")
        print(f"   Model type: {config.MODEL_TYPE}")
        print(f"   Expected inputs: {config.EXPECTED_INPUTS}")
        
        # Try to get input signature
        signature = config.get_input_signature()
        print(f"   Input signature:")
        for name, spec in signature.items():
            print(f"     {name}: shape={spec.shape}, dtype={spec.dtype}")
            
        # Try to create the export wrapper to see what fails
        from keras_hub.src.export.lite_rt import LiteRTExporter
        exporter = LiteRTExporter(config, verbose=True)
        
        # Try to build the wrapper (this is where it might fail)
        print(f"\n🔧 Creating export wrapper...")
        wrapper = exporter._create_export_wrapper()
        print(f"✅ Export wrapper created successfully")
        print(f"   Wrapper inputs: {[inp.shape for inp in wrapper.inputs]}")
        
        # Try a forward pass through the wrapper
        print(f"\n🧪 Testing wrapper forward pass...")
        test_inputs = [
            np.random.random((1, 640, 640, 3)).astype(np.float32),
            np.array([[640, 640]], dtype=np.int32)
        ]
        
        wrapper_output = wrapper(test_inputs)
        print(f"✅ Wrapper forward pass successful:")
        if isinstance(wrapper_output, dict):
            for key, value in wrapper_output.items():
                print(f"   {key}: {value.shape}")
        else:
            print(f"   Output shape: {wrapper_output.shape}")
            
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_object_detection_outputs()
        test_export_attempt()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()