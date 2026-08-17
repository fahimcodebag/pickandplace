#!/usr/bin/env python3
"""
Convert PyTorch model to Float32 TFLite (no quantization)

This avoids ALL quantization issues and should give same performance as PC
"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import os

STATE_DIM = 46
ACTION_DIM = 7


class Actor(nn.Module):
    """PyTorch Actor matching your TD3 implementation"""
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, fc1_dims=512, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.output(x))
        return x


def convert_to_float32_tflite(pytorch_model_path, output_path='actor_float32.tflite'):
    """
    Convert PyTorch model to Float32 TFLite (no quantization)
    
    Args:
        pytorch_model_path: Path to PyTorch model (.pth or checkpoint)
        output_path: Output .tflite file path
        
    Returns:
        Path to generated TFLite model
    """
    print("="*70)
    print("Converting PyTorch Model to Float32 TFLite")
    print("(No Quantization - Full Precision)")
    print("="*70)
    
    # Load PyTorch model
    print(f"\n1. Loading PyTorch model from {pytorch_model_path}")
    model = Actor()
    checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    
    model.eval()
    print("   ✓ PyTorch model loaded")
    
    # Create equivalent Keras model
    print("\n2. Creating Keras model...")
    keras_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(STATE_DIM,)),
        tf.keras.layers.Dense(512, activation='relu', name='fc1'),
        tf.keras.layers.Dense(256, activation='relu', name='fc2'),
        tf.keras.layers.Dense(ACTION_DIM, activation='tanh', name='output')
    ])
    
    # Copy weights from PyTorch to Keras
    with torch.no_grad():
        keras_model.layers[0].set_weights([
            model.fc1.weight.T.numpy(),
            model.fc1.bias.numpy()
        ])
        keras_model.layers[1].set_weights([
            model.fc2.weight.T.numpy(),
            model.fc2.bias.numpy()
        ])
        keras_model.layers[2].set_weights([
            model.output.weight.T.numpy(),
            model.output.bias.numpy()
        ])
    
    print("   ✓ Keras model created")
    
    # Verify conversion accuracy
    print("\n3. Verifying conversion accuracy...")
    test_input = np.random.randn(10, STATE_DIM).astype(np.float32)
    pytorch_output = model(torch.FloatTensor(test_input)).detach().numpy()
    keras_output = keras_model.predict(test_input, verbose=0)
    conversion_error = np.abs(pytorch_output - keras_output).max()
    
    print(f"   Max conversion error: {conversion_error:.10f}")
    
    if conversion_error > 0.001:
        print("   ⚠ WARNING: High conversion error detected!")
    else:
        print("   ✓ Conversion accurate")
    
    # Convert to Float32 TFLite (NO quantization)
    print("\n4. Converting to Float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    
    # NO optimization = keep Float32
    # converter.optimizations = []  # Default is empty = Float32
    
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"   ✓ Successfully converted to Float32 TFLite!")
    print(f"   Output: {output_path}")
    print(f"   Size: {size_kb:.1f} KB")
    
    # Verify TFLite model
    print("\n5. Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"   Input dtype:  {input_details['dtype']}")
    print(f"   Output dtype: {output_details['dtype']}")
    
    if input_details['dtype'] != np.float32:
        print("   ⚠ WARNING: Input is not Float32!")
    else:
        print("   ✓ Input is Float32")
    
    if output_details['dtype'] != np.float32:
        print("   ⚠ WARNING: Output is not Float32!")
    else:
        print("   ✓ Output is Float32")
    
    # Test inference
    print("\n6. Testing inference...")
    test_input_single = test_input[0:1]
    interpreter.set_tensor(input_details['index'], test_input_single)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details['index'])
    
    keras_output_single = keras_output[0:1]
    tflite_error = np.abs(keras_output_single - tflite_output).max()
    
    print(f"   Keras output:  {keras_output_single[0][:4]}")
    print(f"   TFLite output: {tflite_output[0][:4]}")
    print(f"   Max error: {tflite_error:.10f}")
    
    if tflite_error < 1e-6:
        print("   ✓ TFLite inference matches Keras perfectly!")
    elif tflite_error < 1e-4:
        print("   ✓ TFLite inference close to Keras (acceptable)")
    else:
        print("   ⚠ WARNING: TFLite inference differs from Keras")
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print(f"Model: {output_path}")
    print(f"Size: {size_kb:.1f} KB")
    print(f"Type: Float32 (no quantization)")
    print(f"Input: Float32[{STATE_DIM}]")
    print(f"Output: Float32[{ACTION_DIM}]")
    print("\nThis model should give SAME performance as PC (no quantization loss)")
    print("="*70)
    
    return output_path


def test_float32_model(tflite_path, n_samples=100):
    """Quick test of Float32 model"""
    print("\n" + "="*70)
    print("Testing Float32 TFLite Model")
    print("="*70)
    
    import time
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Warmup
    test_input = np.random.randn(1, STATE_DIM).astype(np.float32)
    for _ in range(50):
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details['index'])
    
    # Benchmark
    times = []
    for _ in range(n_samples):
        test_input = np.random.randn(1, STATE_DIM).astype(np.float32)
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nInference speed:")
    print(f"  Average: {avg_time:.3f} ms")
    print(f"  Std dev: {std_time:.3f} ms")
    print(f"  FPS: {1000/avg_time:.1f}")
    print(f"  ESP32 suitable: {'✓ YES' if avg_time < 50 else '✗ NO'}")
    
    print(f"\nOutput sample: {output[0][:4]}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("="*70)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("Float32 TFLite Conversion (No Quantization)")
    print("="*70)
    
    # Check for PyTorch model
    pytorch_model_path = 'checkpoints/td3_builtin/actor_td3'
    if not os.path.exists(pytorch_model_path):
        print(f"✗ PyTorch model not found: {pytorch_model_path}")
        print("\nUsage:")
        print("  python convert_float32_tflite.py [pytorch_model_path] [output_path]")
        sys.exit(1)
    
    output_path = 'actor_float32.tflite'
    
    # Override with command line args if provided
    if len(sys.argv) >= 2:
        pytorch_model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    
    print(f"Input:  {pytorch_model_path}")
    print(f"Output: {output_path}")
    print("="*70)
    
    # Convert
    try:
        result_path = convert_to_float32_tflite(pytorch_model_path, output_path)
        
        # Test
        test_float32_model(result_path)
        
        print("\n✓ Conversion successful!")
        print("\nNext steps:")
        print("1. Copy actor_float32.tflite to your project")
        print("2. Run: python tflite_to_header.py actor_float32.tflite actor_model_float32.h")
        print("3. Update ESP32 code to use Float32 (no quantization needed)")
        print("4. Upload and test - should match PC performance!")
        
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
