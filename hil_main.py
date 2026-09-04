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


def make_env(render=True, random_spawn=False):
    rs_env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=700,              # handoff attempt + scored episode
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
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


def do_handoff(env, raw_env, bridge, render=True):
    """Respawn-and-retry until the ESP32 reports it reached TRANSPORT.

    Returns (obs, attempts, ok). Not scored — mirrors reset()'s attempt loop.
    """
    lo, hi = env.action_space.low, env.action_space.high
    obs = None
    for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
        obs = env.reset()
        bridge.send_reset()                     # resync the MCU's FSM to GRASP
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
    raw_env, env = make_env(render=RENDER, random_spawn=args.random_spawn)
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
            obs, attempts, ok = do_handoff(env, raw_env, bridge, RENDER)
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
