#!/usr/bin/env python3
"""Scripted demonstration collector for PickPlace (bread only).

Uses a proportional controller with the OSC_POSE action space to
generate reach-grasp-lift-place trajectories. Saves transitions as a
.npz file consumable by ``HERReplayBuffer.load_demos()``.

Usage:
    python demo_collector.py [--n_demos 200] [--out demos_bread.npz]
"""

import os
import argparse
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import robosuite as suite
from robosuite.wrappers import GymWrapper
from utils_rl import compute_reward, potential_reward_shaping


# ---------------------------------------------------------------------------
# Indices into the 46-dim GymWrapper observation
# ---------------------------------------------------------------------------
BREAD_POS = slice(0, 3)       # obs[0:3]
EEF_POS = slice(35, 38)       # obs[35:38]
GRIPPER_QPOS = slice(42, 44)  # obs[42:44]


# ---------------------------------------------------------------------------
# Proportional controller
# ---------------------------------------------------------------------------

def p_control_action(eef_pos, target_pos, gripper_cmd, gain=10.0):
    """OSC_POSE action: [dx, dy, dz, dax, day, daz, gripper] in [-1, 1]."""
    delta = (target_pos - eef_pos) * gain
    delta = np.clip(delta, -1.0, 1.0)
    # Keep orientation fixed (zeros) and set gripper
    action = np.zeros(7)
    action[0:3] = delta
    action[6] = gripper_cmd  # +1 = close, -1 = open
    return action


def collect_one_demo(env, gym_env, goal, max_steps=300):
    """Run a single scripted pick-and-place episode.

    Phases:
      1. REACH  – move gripper above bread
      2. LOWER  – descend to grasp height
      3. GRASP  – close gripper and wait
      4. LIFT   – lift bread up
      5. MOVE   – carry bread to target bin
      6. PLACE  – open gripper

    Returns list of transition dicts, or empty list on failure.
    """
    obs = gym_env.reset()
    transitions = []

    bread_pos = obs[BREAD_POS].copy()
    above_bread = bread_pos.copy()
    above_bread[2] += 0.10  # 10 cm above

    grasp_pos = bread_pos.copy()
    grasp_pos[2] += 0.01  # just above table

    lift_pos = bread_pos.copy()
    lift_pos[2] += 0.20  # lift 20 cm

    above_goal = goal.copy()
    above_goal[2] += 0.15  # above bin

    # Phase definitions: (target_pos, gripper_cmd, n_steps, dist_threshold)
    phases = [
        (above_bread, -1.0, 50, 0.02),   # REACH above bread, gripper open
        (grasp_pos,   -1.0, 40, 0.02),   # LOWER to bread, gripper open
        (grasp_pos,    1.0, 15, None),    # GRASP (close gripper, hold)
        (lift_pos,     1.0, 40, 0.02),   # LIFT bread
        (above_goal,   1.0, 60, 0.02),   # MOVE to bin
        (above_goal,  -1.0, 15, None),   # PLACE (open gripper)
    ]

    step_count = 0
    for target, grip, budget, thresh in phases:
        for _ in range(budget):
            if step_count >= max_steps:
                break
            eef = obs[EEF_POS]
            action = p_control_action(eef, target, grip)
            next_obs, env_reward, done, info = gym_env.step(action)

            achieved_goal = next_obs[BREAD_POS].copy()
            reward = float(compute_reward(achieved_goal, goal))
            reward += float(potential_reward_shaping(obs, next_obs, goal))

            transitions.append({
                'state': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_state': next_obs.copy(),
                'done': done,
                'goal': goal.copy(),
                'achieved_goal': achieved_goal,
            })

            obs = next_obs
            step_count += 1

            # Early exit from phase if close enough
            if thresh is not None:
                dist = np.linalg.norm(eef - target)
                if dist < thresh:
                    break

            if done:
                break

        if done or step_count >= max_steps:
            break

    return transitions


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

def collect_demos(n_demos=200, out_path='./demos_bread.npz'):
    env = suite.make(
        'PickPlace', robots='Panda',
        controller_configs=suite.load_controller_config(
            default_controller='OSC_POSE',
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type='bread',
    )
    gym_env = GymWrapper(env)

    # Goal: target bin for bread (index 1)
    goal = env.target_bin_placements[env.object_to_id['bread']].copy()
    print(f"Target bin (goal): {goal}")

    all_states, all_actions, all_rewards = [], [], []
    all_next_states, all_dones, all_goals = [], [], []
    total_transitions = 0
    successes = 0

    for i in range(n_demos):
        transitions = collect_one_demo(env, gym_env, goal)
        if not transitions:
            continue

        for tr in transitions:
            all_states.append(tr['state'])
            all_actions.append(tr['action'])
            all_rewards.append(tr['reward'])
            all_next_states.append(tr['next_state'])
            all_dones.append(tr['done'])
            all_goals.append(tr['goal'])

        # Check if bread ended near goal
        final_bread = transitions[-1]['achieved_goal']
        if np.linalg.norm(final_bread - goal) < 0.10:
            successes += 1

        total_transitions += len(transitions)

        if (i + 1) % 20 == 0 or (i + 1) == n_demos:
            print(
                f"  Demo {i+1:4d}/{n_demos} | "
                f"Transitions: {total_transitions:,} | "
                f"Successes: {successes}/{i+1} "
                f"({100*successes/(i+1):.0f}%)"
            )

    env.close()

    # Save
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(
        out_path,
        states=np.array(all_states),
        actions=np.array(all_actions),
        rewards=np.array(all_rewards),
        next_states=np.array(all_next_states),
        dones=np.array(all_dones),
        goals=np.array(all_goals),
        n_transitions=np.array([total_transitions]),
    )
    print(f"\nSaved {total_transitions:,} transitions to {out_path}")
    print(f"Success rate: {successes}/{n_demos} ({100*successes/max(n_demos,1):.0f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect scripted demos')
    parser.add_argument('--n_demos', type=int, default=200)
    parser.add_argument('--out', type=str, default='./demos_bread.npz')
    args = parser.parse_args()
    collect_demos(args.n_demos, args.out)
