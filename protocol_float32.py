import struct
import numpy as np

class ProtocolFloat32:
    """
    Protocol for Float32 communication (no quantization)
    Simpler than INT8 version - direct float transmission
    """
    STATE_MSG = 0x01
    ACTION_MSG = 0x02
    RESET_MSG = 0x03
    # Observation PLUS the 12 detector features the residual corrector needs.
    # The object-pose block is sent UNCORRECTED, straight out of solvePnP, and
    # the ESP32 corrects it -- so part of perception runs on the MCU rather
    # than the host. Frame is 48 bytes longer than STATE_MSG.
    CORR_MSG  = 0x04
    IMG_MSG   = 0x05
    SYNC_PATTERN = b'\xAA\x55\xAA\x55'
    
    # FSM flag bits (PC -> ESP32). These are physics/contact queries the MCU
    # cannot compute from the 46-float observation, so the simulator (which owns
    # them) ships them alongside the state. Obs-only proxies were tried and
    # measured at 0-2/8 placements vs 93% with the true checks.
    FLAG_GRASPED = 0x01   # raw_env._check_grasp(...)
    FLAG_SUCCESS = 0x02   # raw_env._check_success()

    # FSM phases reported by the ESP32 (ESP32 -> PC), mirroring the `Phase`
    # enum in pick_and_place_INT8_FSM.ino. The PC uses these to run the
    # handoff-retry loop and to detect episode end.
    PHASE_GRASP     = 0
    PHASE_TEST_LIFT = 1
    PHASE_TRANSPORT = 2
    PHASE_RECENTER  = 3
    PHASE_DESCEND   = 4
    PHASE_OPEN      = 5
    PHASE_RETRACT   = 6
    PHASE_DONE_OK   = 7
    PHASE_DONE_FAIL = 8
    PHASE_NAMES = ["GRASP", "TEST_LIFT", "TRANSPORT", "RECENTER", "DESCEND",
                   "OPEN", "RETRACT", "DONE_OK", "DONE_FAIL"]

    @staticmethod
    def encode_state(state, seq_num=0, flags=0):
        """
        Encode state as binary message with FLOAT32 data + FSM flags
        Format: [SYNC:4][type:1][seq:1][len:2][state:184][flags:1][crc:2]
        Total: 199 bytes

        Args:
            state: 46D numpy array (float32)
            seq_num: Sequence number
            flags: bitfield of FLAG_* (true grasp/success from the simulator)

        Returns:
            bytes: Encoded message
        """
        # Header
        msg_type = ProtocolFloat32.STATE_MSG
        payload_len = len(state) * 4 + 1  # 46 floats * 4 bytes + 1 flags byte

        # Pack state as float32, then the flags byte
        state_bytes = state.astype(np.float32).tobytes() + bytes([flags & 0xFF])
        
        # Build message (without sync pattern first for checksum)
        header = struct.pack('BBH', msg_type, seq_num, payload_len)
        message_without_sync = header + state_bytes
        
        # Calculate checksum
        checksum = sum(message_without_sync) % 65536
        crc_bytes = struct.pack('H', checksum)
        
        # Add sync pattern at the beginning
        full_message = ProtocolFloat32.SYNC_PATTERN + message_without_sync + crc_bytes

        return full_message

    @staticmethod
    def encode_state_corr(state, feats, seq_num=0, flags=0):
        """CORR_MSG: the 46-float observation with an UNCORRECTED object pose,
        followed by the 12 corrector features. The board applies the residual
        and re-derives obj_to_eef_pos itself.

        feats must be in the order recorded in corrector_model.h:
          reproj, area_px, obliq_deg, cam_range, gripper_perp, gripper_dz,
          cx, cy, obj_yaw, det_x, det_y, det_z
        """
        assert len(feats) == 12, f"expected 12 features, got {len(feats)}"
        payload_len = len(state) * 4 + 1 + len(feats) * 4
        body = (np.asarray(state, np.float32).tobytes()
                + bytes([flags & 0xFF])
                + np.asarray(feats, np.float32).tobytes())
        header = struct.pack('BBH', ProtocolFloat32.CORR_MSG, seq_num, payload_len)
        msg = header + body
        return (ProtocolFloat32.SYNC_PATTERN + msg
                + struct.pack('H', sum(msg) % 65536))

    # ---- IMG_MSG: full on-device perception -------------------------------
    # CORR_MSG puts the DETECTION on the PC and only the corrector on the
    # board. IMG_MSG moves the whole front end across: the PC sends raw
    # pixels and the camera-to-world transform, and the board detects the
    # tag, solves the pose, applies the calibration and the corrector, and
    # writes the object-pose block of its own state vector.
    #
    # Only the deployment ROI is sent (180x160 = 28800 B), not the whole
    # 320x240 frame -- the board would have to crop to that anyway to fit the
    # detector in RAM, so sending the rest would just cost ~32 KB of link
    # time. At 921600 baud the ROI is ~0.47 s, which is affordable because
    # perception runs ONCE per episode, at t=0.
    #
    # T_world_cam is forward kinematics, not perception: on a real arm the
    # controller knows where the wrist camera is. It is sent as the 12 floats
    # of a 3x4 row-major transform, followed by the 3 floats of the gripper
    # position. The gripper position is part of the message rather than read
    # from the last state, because the image is sent immediately after reset()
    # -- before any state has arrived for this episode, so the board's copy
    # would be the PREVIOUS episode's (or zeros on the first).
    IMG_W, IMG_H = 180, 160
    IMG_X0, IMG_Y0 = 140, 0

    @staticmethod
    def encode_image(roi_bytes, T_world_cam, eef_pos, seq_num=0):
        """IMG_MSG: ROI pixels + the 3x4 camera-to-world transform + gripper pos."""
        npx = ProtocolFloat32.IMG_W * ProtocolFloat32.IMG_H
        assert len(roi_bytes) == npx, f"expected {npx} pixels, got {len(roi_bytes)}"
        T = np.asarray(T_world_cam, np.float32).reshape(4, 4)[:3, :4].ravel()
        e = np.asarray(eef_pos, np.float32).reshape(3)
        payload_len = npx + 15 * 4
        body = (bytes(roi_bytes) + T.astype(np.float32).tobytes()
                + e.astype(np.float32).tobytes())
        header = struct.pack('BBH', ProtocolFloat32.IMG_MSG, seq_num, payload_len)
        msg = header + body
        return (ProtocolFloat32.SYNC_PATTERN + msg
                + struct.pack('H', sum(msg) % 65536))

    @staticmethod
    def crop_roi(gray):
        """Cut the deployment ROI out of a full 320x240 frame."""
        x0, y0 = ProtocolFloat32.IMG_X0, ProtocolFloat32.IMG_Y0
        return np.ascontiguousarray(
            gray[y0:y0 + ProtocolFloat32.IMG_H,
                 x0:x0 + ProtocolFloat32.IMG_W], dtype=np.uint8).tobytes()

    @staticmethod
    def encode_reset(seq_num=0):
        """Force the ESP32's FSM back to GRASP (used between handoff attempts).

        Deliberately the SAME frame size as a state message so the sketch's
        fixed-size reader consumes it immediately instead of stalling on a
        short read. No action is returned for a reset frame.
        """
        zeros = np.zeros(46, dtype=np.float32)
        payload_len = len(zeros) * 4 + 1
        header = struct.pack('BBH', ProtocolFloat32.RESET_MSG, seq_num, payload_len)
        body = header + zeros.tobytes() + bytes([0])
        crc = struct.pack('H', sum(body) % 65536)
        return ProtocolFloat32.SYNC_PATTERN + body + crc

    # Bytes shifted out of the sync search below are not protocol -- they are
    # the sketch's own Serial.printf output (the boot self-test, and the
    # periodic "avg=X.XXms heap=NKB" line), which shares this one UART. Setting
    # this to a bytearray collects them, so a caller can read on-device
    # diagnostics WITHOUT a second serial connection -- the port is exclusive,
    # and under usbipd Windows cannot even see it while WSL holds it.
    debug_sink = None

    @staticmethod
    def decode_action(serial_port, timeout=1.0):
        """
        Decode action from serial port by finding sync pattern first
        Format: [SYNC:4][type:1][seq:1][len:2][action:28][status:1][crc:2]
        Total: 39 bytes (35 after the sync pattern)

        Args:
            serial_port: Serial port object to read from
            timeout: Maximum time to wait for sync pattern

        Returns:
            (action, status): 7D float32 numpy array, and the ESP32's current
            FSM phase (see PHASE_* below) so the PC knows when the handoff
            succeeded and when the episode has ended.
        """
        import time
        
        start_time = time.time()
        sync_buffer = bytearray(4)
        
        # Search for sync pattern
        while (time.time() - start_time) < timeout:
            if serial_port.in_waiting > 0:
                # Shift buffer and read new byte
                if ProtocolFloat32.debug_sink is not None and sync_buffer[0]:
                    ProtocolFloat32.debug_sink.append(sync_buffer[0])
                sync_buffer[0] = sync_buffer[1]
                sync_buffer[1] = sync_buffer[2]
                sync_buffer[2] = sync_buffer[3]
                byte_read = serial_port.read(1)
                if len(byte_read) == 0:
                    continue
                sync_buffer[3] = byte_read[0]
                
                # Check if we found sync pattern
                if bytes(sync_buffer) == ProtocolFloat32.SYNC_PATTERN:
                    # Read the rest of the message (35 bytes after sync)
                    # type(1)+seq(1)+len(2)+action(28)+status(1)+crc(2) = 35
                    remaining_data = serial_port.read(35)

                    if len(remaining_data) != 35:
                        raise TimeoutError(f"Incomplete response after sync: {len(remaining_data)} bytes")
                    
                    # Extract header
                    msg_type, seq_num, payload_len = struct.unpack('BBH', remaining_data[:4])
                    
                    if msg_type != ProtocolFloat32.ACTION_MSG:
                        raise ValueError(f"Wrong message type: {msg_type}")
                    
                    # Verify checksum (on message without sync pattern)
                    message = remaining_data[:-2]
                    received_crc = struct.unpack('H', remaining_data[-2:])[0]
                    calculated_crc = sum(message) % 65536
                    
                    if received_crc != calculated_crc:
                        raise ValueError(f"Checksum mismatch: {received_crc} vs {calculated_crc}")
                    
                    # Extract action (already Float32) — 7 floats = 28 bytes,
                    # then the 1-byte FSM status (phase) the ESP32 reports.
                    action_bytes = remaining_data[4:-3]
                    action = struct.unpack('7f', action_bytes)
                    status = remaining_data[-3]

                    return np.array(action, dtype=np.float32), int(status)

        raise TimeoutError("Sync pattern not found within timeout")


# Backward compatibility: Keep original Protocol class for INT8
class Protocol:
    """Original INT8 quantized protocol"""
    STATE_MSG = 0x01
    ACTION_MSG = 0x02
    RESET_MSG = 0x03
    SYNC_PATTERN = b'\xAA\x55\xAA\x55'
    
    # Quantization parameters - MUST MATCH YOUR TFLITE MODEL!
    # Run: python diagnose_quantization.py actor_int8.tflite
    # to get the correct values for YOUR model
    INPUT_SCALE = 0.02731429971754551
    INPUT_ZERO_POINT = -5

    @staticmethod
    def quantize_state(state):
        """Quantize float32 state to int8"""
        quantized = np.round(state / Protocol.INPUT_SCALE + Protocol.INPUT_ZERO_POINT)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        return quantized
    
    @staticmethod
    def encode_state(state, seq_num=0):
        """Encode state with INT8 quantization"""
        quantized_state = Protocol.quantize_state(state)
        
        msg_type = Protocol.STATE_MSG
        payload_len = len(quantized_state)
        
        state_bytes = quantized_state.tobytes()
        
        header = struct.pack('BBH', msg_type, seq_num, payload_len)
        message_without_sync = header + state_bytes
        
        checksum = sum(message_without_sync) % 65536
        crc_bytes = struct.pack('H', checksum)
        
        full_message = Protocol.SYNC_PATTERN + message_without_sync + crc_bytes
        
        return full_message
    
    @staticmethod
    def decode_action(serial_port, timeout=1.0):
        """Decode action (same as Float32 version)"""
        import time
        
        start_time = time.time()
        sync_buffer = bytearray(4)
        
        while (time.time() - start_time) < timeout:
            if serial_port.in_waiting > 0:
                if ProtocolFloat32.debug_sink is not None and sync_buffer[0]:
                    ProtocolFloat32.debug_sink.append(sync_buffer[0])
                sync_buffer[0] = sync_buffer[1]
                sync_buffer[1] = sync_buffer[2]
                sync_buffer[2] = sync_buffer[3]
                byte_read = serial_port.read(1)
                if len(byte_read) == 0:
                    continue
                sync_buffer[3] = byte_read[0]
                
                if bytes(sync_buffer) == Protocol.SYNC_PATTERN:
                    remaining_data = serial_port.read(34)
                    
                    if len(remaining_data) != 34:
                        raise TimeoutError(f"Incomplete response: {len(remaining_data)} bytes")
                    
                    msg_type, seq_num, payload_len = struct.unpack('BBH', remaining_data[:4])
                    
                    if msg_type != Protocol.ACTION_MSG:
                        raise ValueError(f"Wrong message type: {msg_type}")
                    
                    message = remaining_data[:-2]
                    received_crc = struct.unpack('H', remaining_data[-2:])[0]
                    calculated_crc = sum(message) % 65536
                    
                    if received_crc != calculated_crc:
                        raise ValueError(f"Checksum mismatch")
                    
                    action_bytes = remaining_data[4:-2]
                    action = struct.unpack('7f', action_bytes)
                    
                    return np.array(action, dtype=np.float32)
        
        raise TimeoutError("Sync pattern not found")
