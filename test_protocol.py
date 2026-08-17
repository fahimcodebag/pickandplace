#!/usr/bin/env python3
"""
ESP32 Protocol Test Script
Test communication before running full HIL simulation
"""

import serial
import time
import numpy as np
from protocol_float32 import ProtocolFloat32 as Protocol
import struct

def test_serial_connection(port='/dev/ttyUSB0', baudrate=921600):
    """Test basic serial connection"""
    print("\n" + "="*70)
    print("TEST 1: Serial Connection")
    print("="*70)
    
    try:
        ser = serial.Serial(port, baudrate, timeout=2.0)
        time.sleep(2)  # Wait for ESP32 reset
        print(f"✓ Connected to {port} @ {baudrate} baud")
        
        # Read startup messages from ESP32
        print("\nESP32 startup messages:")
        print("-" * 70)
        time.sleep(1)
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"  {line}")
        print("-" * 70)
        
        return ser
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None


def test_message_encoding():
    """Test message encoding"""
    print("\n" + "="*70)
    print("TEST 2: Message Encoding")
    print("="*70)
    
    # Create test state
    state = np.random.randn(46).astype(np.float32)
    seq_num = 42
    
    # Encode
    message = Protocol.encode_state(state, seq_num)
    
    print(f"Test state: {state[:4]} ... (46 values)")
    print(f"Sequence number: {seq_num}")
    print(f"Encoded message length: {len(message)} bytes")
    
    # Verify structure
    expected_length = 4 + 1 + 1 + 2 + 184 + 2  # sync + type + seq + len + state + crc
    if len(message) == expected_length:
        print(f"✓ Message length correct ({expected_length} bytes)")
    else:
        print(f"✗ Message length incorrect: {len(message)} vs {expected_length}")
        return False
    
    # Check sync pattern
    sync = message[:4]
    if sync == b'\xAA\x55\xAA\x55':
        print("✓ Sync pattern correct")
    else:
        print(f"✗ Sync pattern incorrect: {sync.hex()}")
        return False
    
    # Check message type
    msg_type = message[4]
    if msg_type == 0x01:
        print("✓ Message type correct (STATE_MSG)")
    else:
        print(f"✗ Message type incorrect: {msg_type}")
        return False
    
    # Check sequence number
    msg_seq = message[5]
    if msg_seq == seq_num:
        print(f"✓ Sequence number correct ({seq_num})")
    else:
        print(f"✗ Sequence number incorrect: {msg_seq}")
        return False
    
    return True


def test_single_inference(ser, test_num=1):
    """Test single inference cycle"""
    print(f"\n{'='*70}")
    print(f"TEST 3.{test_num}: Single Inference Cycle")
    print("="*70)
    
    # Create random state
    state = np.random.randn(46).astype(np.float32)
    seq_num = test_num
    
    print(f"Sending state (seq={seq_num})...")
    print(f"  State sample: [{state[0]:.4f}, {state[1]:.4f}, {state[2]:.4f}, ...]")
    
    # Clear buffer
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Send state
    message = Protocol.encode_state(state, seq_num)
    start_time = time.perf_counter()
    ser.write(message)
    ser.flush()
    
    print(f"  Sent {len(message)} bytes")
    
    # Wait for action
    try:
        action, phase = Protocol.decode_action(ser, timeout=3.0)
        cycle_time = (time.perf_counter() - start_time) * 1000

        print(f"✓ Received action in {cycle_time:.2f} ms")
        print(f"  FSM phase: {Protocol.PHASE_NAMES[phase]}")
        print(f"  Action: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, ...]")
        print(f"  Action range: [{action.min():.4f}, {action.max():.4f}]")
        
        # Verify action is in valid range
        if np.all((action >= -1.0) & (action <= 1.0)):
            print("✓ Action values in valid range [-1, 1]")
        else:
            print("⚠ Action values outside expected range!")
            
        return True, cycle_time
        
    except TimeoutError as e:
        print(f"✗ Timeout: {e}")
        print("  ESP32 did not respond")
        
        # Check if there's any data in buffer
        if ser.in_waiting > 0:
            print(f"  Buffer contains {ser.in_waiting} bytes:")
            data = ser.read(min(ser.in_waiting, 100))
            print(f"    {data.hex()}")
        
        return False, 0
        
    except ValueError as e:
        print(f"✗ Protocol error: {e}")
        return False, 0


def test_multiple_cycles(ser, n_cycles=10):
    """Test multiple inference cycles"""
    print(f"\n{'='*70}")
    print(f"TEST 4: Multiple Cycles ({n_cycles} cycles)")
    print("="*70)
    
    successes = 0
    failures = 0
    times = []
    
    for i in range(n_cycles):
        success, cycle_time = test_single_inference(ser, i+1)
        
        if success:
            successes += 1
            times.append(cycle_time)
        else:
            failures += 1
            print(f"  Cycle {i+1} failed, waiting 1s before retry...")
            time.sleep(1)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {successes}/{n_cycles} successful")
    print("="*70)
    
    if times:
        print(f"Average cycle time: {np.mean(times):.2f} ± {np.std(times):.2f} ms")
        print(f"Min: {np.min(times):.2f} ms")
        print(f"Max: {np.max(times):.2f} ms")
        print(f"Success rate: {successes/n_cycles*100:.1f}%")
        
        if successes == n_cycles:
            print("\n✓ ALL TESTS PASSED - Protocol working perfectly!")
            print("  You can now run hil_main.py")
        else:
            print(f"\n⚠ {failures} failures detected")
            print("  Check ESP32 serial monitor for errors")
    else:
        print("\n✗ NO SUCCESSFUL CYCLES")
        print("  Troubleshooting steps:")
        print("  1. Check ESP32 serial monitor output")
        print("  2. Verify correct .tflite model is loaded")
        print("  3. Check STATE_PAYLOAD_SIZE = 190 in .ino file")
        print("  4. Verify baudrate matches (921600)")


def main():
    print("="*70)
    print("ESP32 PROTOCOL TEST SUITE")
    print("="*70)
    print("\nThis will test communication with your ESP32 before HIL")
    print("Make sure:")
    print("  1. ESP32 is connected via USB")
    print("  2. Updated .ino code is uploaded")
    print("  3. Serial monitor is CLOSED (conflicts with this script)")
    
    # Test encoding
    if not test_message_encoding():
        print("\n✗ Encoding test failed - fix Python code first")
        return
    
    # Test connection
    port = input("\nSerial port [/dev/ttyUSB0]: ").strip() or '/dev/ttyUSB0'
    ser = test_serial_connection(port)
    
    if ser is None:
        print("\n✗ Connection failed")
        print("\nTroubleshooting:")
        print("  - Check port with: ls /dev/ttyUSB*")
        print("  - Check permissions: sudo usermod -a -G dialout $USER")
        print("  - Try: sudo chmod 666 /dev/ttyUSB0")
        return
    
    try:
        # Wait for ESP32 to be ready
        print("\nWaiting 3 seconds for ESP32 to initialize...")
        time.sleep(3)
        
        # Single test
        success, _ = test_single_inference(ser, 1)
        
        if success:
            # Multiple cycles
            test_multiple_cycles(ser, 10)
        else:
            print("\n✗ First test failed - not running multiple cycles")
            print("\nCheck ESP32 serial output for errors:")
            print("  arduino-cli monitor -p", port)
            
    finally:
        ser.close()
        print("\n✓ Disconnected")


if __name__ == "__main__":
    main()
