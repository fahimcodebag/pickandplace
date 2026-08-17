#!/usr/bin/env python3
"""
Convert TensorFlow Lite model to C header file for ESP32
This script takes a .tflite file and generates a C array that can be included in Arduino sketches
"""

import os
import sys

def convert_tflite_to_header(tflite_path, output_path=None):
    """
    Convert TFLite model to C header file
    
    Args:
        tflite_path: Path to .tflite model file
        output_path: Output .h file path (optional)
    """
    if not os.path.exists(tflite_path):
        print(f"Error: File not found: {tflite_path}")
        return False
    
    # Read TFLite file
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()
    
    model_size = len(tflite_data)
    
    # Generate output filename if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(tflite_path))[0]
        output_path = f"{base_name}.h"
    
    # Extract model name from filename
    model_name = os.path.splitext(os.path.basename(output_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Converting TFLite Model to C Header")
    print(f"{'='*60}")
    print(f"Input:  {tflite_path}")
    print(f"Output: {output_path}")
    print(f"Size:   {model_size:,} bytes ({model_size/1024:.1f} KB)")
    print(f"{'='*60}\n")
    
    # Generate C header content
    header_content = f"""// Auto-generated TensorFlow Lite model
// Source: {os.path.basename(tflite_path)}
// Size: {model_size} bytes ({model_size/1024:.1f} KB)

#ifndef {model_name.upper()}_H
#define {model_name.upper()}_H

const unsigned char {model_name}_tflite[] = {{
"""
    
    # Write data as hex bytes, 12 per line
    bytes_per_line = 12
    for i in range(0, model_size, bytes_per_line):
        line_bytes = tflite_data[i:i+bytes_per_line]
        hex_bytes = ', '.join([f'0x{b:02x}' for b in line_bytes])
        header_content += f"  {hex_bytes}"
        
        if i + bytes_per_line < model_size:
            header_content += ","
        
        header_content += "\n"
    
    header_content += f"""}}; 

const unsigned int {model_name}_tflite_len = {model_size};

#endif  // {model_name.upper()}_H
"""
    
    # Write header file
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"✓ Successfully generated {output_path}")
    print(f"✓ Array name: {model_name}_tflite")
    print(f"✓ Size constant: {model_name}_tflite_len\n")
    
    return True


def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python tflite_to_header.py <model.tflite> [output.h]")
        print("\nExample:")
        print("  python tflite_to_header.py actor_int8.tflite actor_model.h")
        sys.exit(1)
    
    tflite_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = convert_tflite_to_header(tflite_path, output_path)
    
    if success:
        print("Next steps:")
        print("1. Copy the generated .h file to your Arduino project directory")
        print("2. Update the #include in your .ino file to match the header name")
        print("3. Upload to ESP32")
        print("\nNote: If you get memory errors, you may need to:")
        print("  - Reduce kTensorArenaSize in the .ino file")
        print("  - Use a smaller/more compressed model")
        print("  - Enable PSRAM if available on your ESP32")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
