"""
Hardware-in-the-loop driver for the two-model INT8 FSM on ESP32.

The ESP32 owns the policy stack: both INT8 models (grasp + transport), the FSM
phases, counters and the scripted P-controller release. The PC owns physics —
it steps robosuite, and supplies the two predicates the MCU cannot compute from
the 46-float observation (_check_grasp / _check_success) as flag bits.

HANDOFF RETRIES: place_env_wrapper.reset() retries a failed grasp/test-lift with
a fresh env.reset() (respawn), up to _MAX_GRASP_ATTEMPTS, BEFORE the episode is
scored — the validated 92% is conditional on that loop. Only the PC can respawn,
so the loop lives here. Retries are reported separately and are not counted as
episode failures, matching the Python pipeline's semantics exactly.
"""

import json
import os
import time

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper

from esp32_bridge import ESP32Bridge
from protocol_float32 import ProtocolFloat32 as Protocol

MAX_GRASP_ATTEMPTS = 8      # mirrors place_env_wrapper._MAX_GRASP_ATTEMPTS
HANDOFF_STEP_CAP = 260      # per-attempt cap (sketch gives up at 250)
EPISODE_STEP_CAP = 400      # post-handoff cap (sketch's place horizon is 300)

# Set True to trace the handoff: prints every FSM phase change reported by the
# ESP32 alongside the flags the PC sent, so a stuck handoff shows immediately
# whether (a) the PC never asserts FLAG_GRASPED, (b) the MCU never reacts to it,
# or (c) the MCU advances but the test-lift keeps failing.
DEBUG_HANDOFF = True


def compute_flags(raw_env):
    """The simulator's true contact/success checks, packed as protocol flags."""
    flags = 0
    try:
        if raw_env._check_grasp(gripper=raw_env.robots[0].gripper,
                                object_geoms=raw_env.objects[raw_env.object_id]):
            flags |= Protocol.FLAG_GRASPED
    except Exception:
        pass
    try:
        if raw_env._check_success():
            flags |= Protocol.FLAG_SUCCESS
    except Exception:
        pass
    return flags


def render_wrist_gray(raw_env, width=320, height=240,
                     camera="robot0_eye_in_hand"):
    """The 320x240 grayscale wrist frame the ESP32 will detect in.

    Rendered directly rather than through use_camera_obs, because GymWrapper
    flattens every observation into the state vector and an image would
    destroy the 46-float layout the board expects.

    The [::-1] flip matches TagDetector.frame(): MuJoCo renders bottom-up and
    every detection result in Results/ was measured on the flipped image.
    """
    import cv2
    img = raw_env.sim.render(width=width, height=height, camera_name=camera)
    return cv2.cvtColor(np.ascontiguousarray(img[::-1]), cv2.COLOR_RGB2GRAY)


def send_perception(raw_env, bridge, camera="robot0_eye_in_hand"):
    """Hand the board one frame and let IT do the perception.

    Called once per episode with the arm at its home pose -- the condition the
    residual corrector was calibrated under. T_world_cam is forward kinematics,
    not perception: on a real arm the controller knows where the wrist camera
    sits.
    """
    import robosuite.utils.camera_utils as CU
    gray = render_wrist_gray(raw_env, camera=camera)
    T = CU.get_camera_extrinsic_matrix(raw_env.sim, camera)
    eef = np.asarray(raw_env._observables["robot0_eef_pos"].obs, dtype=np.float32)
    bridge.send_image(gray, T, eef)


def make_env(render=True, random_spawn=False, on_device_perception=False):
    rs_env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"),
        has_renderer=render,
        # Offscreen rendering only when the board is doing its own perception
        # -- it costs time and every previous HIL number was measured without.
        has_offscreen_renderer=on_device_perception,
        use_camera_obs=False,
        horizon=700,              # handoff attempt + scored episode
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    if on_device_perception:
        # The board can only detect a tag that EXISTS. hil_main has always
        # built a plain PickPlace env, which has no tag on the bread -- so
        # without this the wrist frames are tagless and the board reports
        # det=0 forever, which looks exactly like a broken detector.
        #
        # Injection goes through robosuite's set_xml_processor hook, the same
        # way apriltag_sim.make_tagged_env does it, so the tag survives every
        # hard reset instead of being wiped by the next _load_model().
        import os as _os
        from apriltag_sim import inject_tag, generate_tag_png
        _png = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             "assets", "tag36h11_0.png")
        _half, _zoff = 0.022, 0.026
        _marker_size_m = 2.0 * _half * generate_tag_png(_png)
        # The board hardcodes AT_TAGSIZE; a mismatch silently rescales every
        # pose it reports, so fail loudly instead.
        assert abs(_marker_size_m - 0.034375) < 1e-6, (
            f"tag is {_marker_size_m:.6f} m but at32_perception.h assumes "
            f"0.034375 m -- update AT_TAGSIZE")
        rs_env.set_xml_processor(lambda xml: inject_tag(
            xml, _png, body_name="Bread_main",
            half_size=_half, z_offset=_zoff))
        rs_env.reset()

    # Spawn condition. Fixed pins the object at the tuned pose (x=y=0, no
    # rotation) and is what this rig has always run. Random uses robosuite's
    # native PickPlace box PLUS z-rotation -- the condition every headline
    # number in Results/ is measured under, and the harder one by ~5 points.
    _orig_gpi = rs_env._get_placement_initializer

    def _fixed_placement():
        _orig_gpi()
        s = rs_env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = np.array([0.0, 0.0])
        s.y_range = np.array([0.0, 0.0])
        s.rotation = 0.0
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False

    if not random_spawn:
        rs_env._get_placement_initializer = _fixed_placement
    return rs_env, GymWrapper(rs_env)


def do_handoff(env, raw_env, bridge, render=True, on_device_perception=False):
    """Respawn-and-retry until the ESP32 reports it reached TRANSPORT.

    Returns (obs, attempts, ok). Not scored — mirrors reset()'s attempt loop.
    """
    lo, hi = env.action_space.low, env.action_space.high
    obs = None
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        obs = env.reset()
        bridge.send_reset()                     # resync the MCU's FSM to GRASP
        if on_device_perception:
            # After send_reset, so the latch the board clears on reset is the
            # one this frame refills -- not the other way round.
            send_perception(raw_env, bridge)
        last_phase = None
        grasp_flag_steps = 0
        for t in range(HANDOFF_STEP_CAP):
            flags = compute_flags(raw_env)
            grasp_flag_steps += int(bool(flags & Protocol.FLAG_GRASPED))
            action, phase = bridge.get_action(obs, flags)
            if DEBUG_HANDOFF and phase != last_phase:
                print(f"      t={t:3d} flags={flags:#04x} "
                      f"phase={Protocol.PHASE_NAMES[phase]}")
                last_phase = phase
            obs, _, done, _ = env.step(np.clip(action, lo, hi))
            if render:
                raw_env.render()
            if phase == Protocol.PHASE_TRANSPORT:
                return obs, attempt, True       # grasp survived the test-lift
            if phase in (Protocol.PHASE_DONE_OK, Protocol.PHASE_DONE_FAIL):
                break                           # sketch gave up -> respawn
            if done:
                break
        print(f"    handoff attempt {attempt} failed — respawning "
              f"(PC asserted FLAG_GRASPED on {grasp_flag_steps} steps)")
    return obs, MAX_GRASP_ATTEMPTS, False


def run_episode(env, raw_env, bridge, obs, ep, render=True):
    """Scored portion: transport + scripted release, driven by the ESP32."""
    lo, hi = env.action_space.low, env.action_space.high
    log = {'states': [], 'actions': [], 'rewards': [], 'cycle_times': [],
           'phases': []}
    score = 0.0
    steps = 0
    phase = Protocol.PHASE_TRANSPORT

    for _ in range(EPISODE_STEP_CAP):
        t0 = time.perf_counter()
        action, phase = bridge.get_action(obs, compute_flags(raw_env))
        next_obs, reward, done, _ = env.step(np.clip(action, lo, hi))
        if render:
            raw_env.render()

        log['states'].append(np.asarray(obs).tolist())
        log['actions'].append(np.asarray(action).tolist())
        log['rewards'].append(float(reward))
        log['cycle_times'].append((time.perf_counter() - t0) * 1000)
        log['phases'].append(int(phase))

        score += reward
        steps += 1
        obs = next_obs

        if phase in (Protocol.PHASE_DONE_OK, Protocol.PHASE_DONE_FAIL):
            break
        if done:
            break

    try:
        placed = bool(raw_env._check_success())
    except Exception:
        placed = False

    with open(f'logs/episode_{ep:03d}.json', 'w') as f:
        json.dump(log, f)
    return score, steps, placed, phase


def main():
    import argparse
    p = argparse.ArgumentParser(description="Hardware-in-the-loop with the ESP32")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--on-device-perception", action="store_true",
                   help="send the wrist frame to the ESP32 (IMG_MSG) and let it "
                        "detect the tag, solve the pose and correct it, instead "
                        "of the PC passing ground truth. Needs the sketch built "
                        "with src/esp32_apriltag/. Prints [perc] lines to the "
                        "--debug-log.")
    p.add_argument("--random-spawn", action="store_true",
                   help="native PickPlace spawn box + z-rotation. This is the "
                        "condition Results/ reports (95.33%% INT8 in sim); the "
                        "default fixed spawn scores 99.83%%.")
    p.add_argument("--port", default=None, help="override the serial port")
    p.add_argument("--debug-log", default=None,
                   help="write the ESP32's own printf output here, including "
                        "its 'avg=X.XXms' on-device inference timing. The "
                        "serial port is exclusive, so this is the only way to "
                        "read it while HIL runs.")
    p.add_argument("--render", dest="render", action="store_true", default=None,
                   help="force the on-screen viewer on")
    p.add_argument("--no-render", dest="render", action="store_false",
                   help="disable the viewer. Worth using for long runs: "
                        "rendering every control step is pure overhead when "
                        "nobody is watching.")
    args = p.parse_args()
    SERIAL_PORT = '/dev/ttyUSB0'
    BAUDRATE = 921600
    N_EPISODES = args.episodes
    RENDER = True

    print("\n" + "=" * 62)
    print("Hardware-in-the-Loop: ESP32 two-model INT8 FSM")
    print("=" * 62)

    print("\n1. Creating RoboSuite environment...")
    # None = fall back to the module default; the flags force it either way.
    # (Previously this read `args.render or RENDER`, which could never be
    # False and so left no way to switch the viewer off.)
    RENDER = RENDER if args.render is None else args.render
    raw_env, env = make_env(render=RENDER, random_spawn=args.random_spawn,
                            on_device_perception=args.on_device_perception)
    print(f"   spawn: {'RANDOM (box + rotation)' if args.random_spawn else 'FIXED'}")
    print(f"✓ Environment ready: {env.observation_space.shape} → "
          f"{env.action_space.shape}")

    print(f"\n2. Connecting to ESP32 on {SERIAL_PORT}...")
    bridge = ESP32Bridge(port=args.port or SERIAL_PORT, baudrate=BAUDRATE,
                         timeout=2.0, debug_log=args.debug_log)
    if not bridge.connect():
        print("✗ Failed to connect to ESP32. Exiting.")
        return

    print(f"\n3. Running {N_EPISODES} episodes...")
    print("-" * 62)

    scores, placements, attempts_log = [], 0, []
    try:
        for i in range(1, N_EPISODES + 1):
            print(f"\nEpisode {i}/{N_EPISODES}")
            obs, attempts, ok = do_handoff(env, raw_env, bridge, RENDER,
                                           args.on_device_perception)
            attempts_log.append(attempts)
            if not ok:
                print(f"  ✗ handoff failed after {MAX_GRASP_ATTEMPTS} attempts "
                      f"— episode skipped (grasp stage, not transport)")
                continue

            score, steps, placed, phase = run_episode(
                env, raw_env, bridge, obs, i, RENDER)
            scores.append(score)
            placements += int(placed)
            print(f"  handoff attempts: {attempts} | steps: {steps} | "
                  f"score: {score:.2f} | end: {Protocol.PHASE_NAMES[phase]} | "
                  f"{'✓ PLACED' if placed else '✗ failed'}")

            if i % 5 == 0:
                bridge.print_stats()

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except RuntimeError as e:
        print(f"\n✗ Communication error: {e}")
    finally:
        bridge.disconnect()
        n = len(scores)
        print(f"\n{'=' * 62}")
        if args.on_device_perception:
        # Printed BEFORE the score, because the score is uninterpretable
        # without it: if nothing was detected the board used the PC's
        # ground-truth pose and the number says nothing about perception.
        print("-" * 62)
        print(bridge.perception_summary())
        print("-" * 62)
    print(f"Final Results ({n} scored episodes of {N_EPISODES})")
        print(f"{'=' * 62}")
        if n:
            print(f"Placements:            {placements}/{n} "
                  f"({placements / n * 100:.1f}%)")
            print(f"Average score:         {np.mean(scores):.2f} "
                  f"± {np.std(scores):.2f}")
        if attempts_log:
            print(f"Mean handoff attempts: {np.mean(attempts_log):.2f} "
                  f"(retries are NOT counted as episode failures — same "
                  f"convention as the 92% Python baseline)")
        print(f"{'=' * 62}\n")
        bridge.print_stats()


if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    main()
