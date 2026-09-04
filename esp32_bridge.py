import serial
import time
import numpy as np
from protocol_float32 import ProtocolFloat32 as Protocol

class ESP32Bridge:
    def __init__(self, port='/dev/ttyUSB0', baudrate=921600, timeout=1.0,
                 debug_log=None):
        """
        Initialize ESP32 serial bridge

        Args:
            port: Serial port (in WSL2, will be /dev/ttyUSBx)
            baudrate: Communication speed
            timeout: Read timeout in seconds
            debug_log: path to write the ESP32's OWN stdout to. The sketch
                prints its boot self-test and a periodic
                "[n] ep=N phase=P avg=X.XXms heap=NKB" line on the same UART
                the binary protocol uses, and the sync search discards those
                bytes. Capturing them here is the only way to read on-device
                diagnostics during a run: the serial port is exclusive, so the
                Arduino monitor cannot be attached at the same time, and under
                usbipd Windows cannot see the device at all while WSL holds it.
                The "avg=" figure is inference + FSM timed with micros() on the
                board -- NOT the PC-side round trip that stats['avg_cycle_time']
                reports.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.seq_num = 0
        
        # Statistics
        self.stats = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'timeouts': 0,
            'checksum_errors': 0,
            'avg_cycle_time': 0.0
        }

        self.debug_log = debug_log
        self._dbg_buf = bytearray()
        if debug_log:
            Protocol.debug_sink = self._dbg_buf
            self._dbg_fh = open(debug_log, "w", buffering=1)
        else:
            self._dbg_fh = None

    def _drain_debug(self):
        """Move any complete lines the board printed into the log file."""
        if self._dbg_fh is None:
            return
        while b"\n" in self._dbg_buf:
            line, _, rest = self._dbg_buf.partition(b"\n")
            self._dbg_buf[:] = rest
            txt = line.decode("utf-8", "replace").strip("\r\x00")
            if txt.strip():
                self._dbg_fh.write(txt + "\n")
        
    def connect(self):
        """Establish serial connection to ESP32"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            time.sleep(2)  # Wait for ESP32 to reset after connection
            
            # Flush buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            print(f"✓ Connected to ESP32 on {self.port} @ {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("✓ Disconnected from ESP32")
        if self._dbg_fh is not None:
            self._drain_debug()
            self._dbg_fh.close()
            Protocol.debug_sink = None
            self._dbg_fh = None
            print(f"✓ On-device log written to {self.debug_log}")
    
    def get_action(self, state, flags=0, retries=3):
        """
        Send state (+ FSM flags) to ESP32 and receive action (+ FSM phase)

        Args:
            state: 46D numpy array
            flags: bitfield of Protocol.FLAG_* — the simulator's true
                   _check_grasp / _check_success, which the MCU cannot compute
                   from the observation alone
            retries: Number of retry attempts

        Returns:
            (action, phase): 7D numpy array and the ESP32's FSM phase
        """
        start_time = time.perf_counter()

        for attempt in range(retries):
            try:
                # Encode and send state
                message = Protocol.encode_state(state, self.seq_num, flags)
                self.serial.write(message)
                self.serial.flush()  # Ensure data is sent

                # Wait for action response (with sync pattern search)
                action, phase = Protocol.decode_action(self.serial, timeout=self.timeout)
                
                # Update statistics
                cycle_time = (time.perf_counter() - start_time) * 1000
                self.stats['total_cycles'] += 1
                self.stats['successful_cycles'] += 1
                self._drain_debug()
                self.stats['avg_cycle_time'] = (
                    (self.stats['avg_cycle_time'] * (self.stats['total_cycles'] - 1) + cycle_time)
                    / self.stats['total_cycles']
                )
                
                self.seq_num = (self.seq_num + 1) % 256
                return action, phase


            except TimeoutError as e:
                self.stats['timeouts'] += 1
                print(f"⚠ Timeout (attempt {attempt + 1}/{retries}): {e}")
                self.serial.reset_input_buffer()
                time.sleep(0.1)  # Small delay before retry
                
            except ValueError as e:
                self.stats['checksum_errors'] += 1
                print(f"⚠ Checksum error (attempt {attempt + 1}/{retries}): {e}")
                self.serial.reset_input_buffer()
                time.sleep(0.1)  # Small delay before retry
        
        # All retries failed
        self.stats['total_cycles'] += 1
        raise RuntimeError(f"Failed to get action after {retries} attempts")
    
    def send_reset(self):
        """Force the ESP32 FSM back to GRASP (between handoff attempts).

        Fire-and-forget: the sketch consumes the frame and returns no action,
        so we just drain any stale bytes afterwards.
        """
        self.serial.write(Protocol.encode_reset(self.seq_num))
        self.serial.flush()
        time.sleep(0.02)
        self.serial.reset_input_buffer()

    def print_stats(self):
        """Print communication statistics"""
        success_rate = (self.stats['successful_cycles'] / max(self.stats['total_cycles'], 1)) * 100
        print(f"\n{'='*60}")
        print(f"ESP32 Communication Statistics")
        print(f"{'='*60}")
        print(f"Total cycles: {self.stats['total_cycles']}")
        print(f"Successful: {self.stats['successful_cycles']} ({success_rate:.1f}%)")
        print(f"Timeouts: {self.stats['timeouts']}")
        print(f"Checksum errors: {self.stats['checksum_errors']}")
        print(f"Avg cycle time: {self.stats['avg_cycle_time']:.2f} ms")
        print(f"{'='*60}\n")
