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
        self.recent_lines = []          # last board prints, for send_image()
        self.perc = {"sent": 0, "detected": 0, "oom": 0, "short": 0, "crc": 0,
                     "ms": [], "used": []}
        if debug_log:
            Protocol.debug_sink = self._dbg_buf
            self._dbg_fh = open(debug_log, "w", buffering=1)
        else:
            self._dbg_fh = None

    def _pump_serial_debug(self):
        """Pull whatever the board has printed into the debug buffer.

        _drain_debug only moves bytes the PROTOCOL's sync search discarded.
        Anything the board printed while the PC was not mid-frame sits in the
        OS serial buffer, and send_reset/send_image then call
        reset_input_buffer() and throw it away -- which silently discarded
        every [perc] diagnostic. Only safe in the fire-and-forget windows,
        where no protocol reply is expected.
        """
        try:
            n = self.serial.in_waiting
            if n:
                self._dbg_buf.extend(self.serial.read(n))
        except Exception:
            pass

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
                self.recent_lines.append(txt)
                del self.recent_lines[:-200]
        
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
        self._pump_serial_debug()
        self._drain_debug()
        self.serial.reset_input_buffer()

    def send_image(self, gray, T_world_cam, eef_pos, settle=2.0):
        """Send one camera frame for ON-DEVICE perception (IMG_MSG).

        Fire-and-forget like send_reset: the sketch runs the detector and
        latches the result, and returns no action, so there is nothing to
        read back. Perception is expected ONCE per episode, at t=0.

        `settle` must cover the link time plus the detection. The ROI is
        42846 bytes, which at 921600 baud is ~0.47 s, and the detector itself
        is the unknown -- it has never been timed on hardware. The board
        prints "[perc] det=.. NN.N ms heap ..." for each frame, so the real
        figure lands in the debug log; shorten this once it is known.
        """
        roi = Protocol.crop_roi(gray)
        mark = len(self.recent_lines)
        self.serial.write(Protocol.encode_image(roi, T_world_cam, eef_pos,
                                               self.seq_num))
        self.serial.flush()
        # Poll rather than one long sleep, so a slow detection is captured
        # without waiting the full settle when it is fast.
        deadline = time.time() + settle
        while time.time() < deadline:
            self._pump_serial_debug()
            self._drain_debug()
            if any("[perc]" in l for l in self.recent_lines[mark:]):
                time.sleep(0.05)          # let the rest of the line land
                self._pump_serial_debug()
                self._drain_debug()
                break
            time.sleep(0.02)
        self.serial.reset_input_buffer()
        self.perc["sent"] += 1

        # Read back what the board actually did. Without this a failed
        # detection is INVISIBLE: the board simply does not latch, applyLatch
        # becomes a no-op, and the object pose the PC sent -- ground truth --
        # is used instead. The run then looks excellent while measuring
        # nothing about perception at all.
        for ln in self.recent_lines[mark:]:
            if "[perc]" not in ln:
                continue
            if "OUT OF MEMORY" in ln:
                self.perc["oom"] += 1
            elif "SHORT" in ln:
                self.perc["short"] += 1
            elif "CRC" in ln:
                self.perc["crc"] += 1
            elif "det=" in ln:
                try:
                    if int(ln.split("det=")[1].split()[0]) == 1:
                        self.perc["detected"] += 1
                    self.perc["ms"].append(
                        float(ln.split("det=")[1].split()[1]))
                    if "peak used" in ln:
                        self.perc["used"].append(
                            int(ln.split("peak used")[1].split()[0]))
                except (ValueError, IndexError):
                    pass
        return dict(self.perc)

    def perception_summary(self):
        """One line saying whether perception actually ran, and at what cost."""
        p = self.perc
        if not p["sent"]:
            return "perception: never invoked"
        seen = p["detected"] + p["oom"] + p["short"] + p["crc"]
        out = [f"perception: {p['sent']} frames sent, "
               f"{p['detected']} detected"]
        for k, lbl in (("oom", "OUT-OF-MEMORY"), ("short", "short reads"),
                       ("crc", "CRC errors")):
            if p[k]:
                out.append(f"{p[k]} {lbl}")
        if p["ms"]:
            out.append(f"detect {min(p['ms']):.0f}-{max(p['ms']):.0f} ms "
                       f"(mean {sum(p['ms'])/len(p['ms']):.0f})")
        if p["used"]:
            out.append(f"peak heap used {max(p['used'])/1024:.0f} KB")
        if seen == 0:
            out.append("!! the board printed NOTHING -- it is almost certainly "
                       "not running the perception build, so these episodes "
                       "used the PC's ground-truth pose")
        return " | ".join(out)

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
