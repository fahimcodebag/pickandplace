"""Periodic probe of agent internals — uniform across TD3 / SAC / PPO.

Reads the agent from the outside (samples its replay buffer, runs its own
networks) rather than instrumenting each `learn()`. That keeps `td3.py` — the
file every existing thesis result was produced with — completely untouched,
while still yielding the critic/actor curves a thesis needs.

Returns a flat dict of scalars; keys absent for an algorithm are simply omitted.
"""

import numpy as np
import torch as T


def probe(agent, batch=512):
    """Sample the agent's buffer and summarize critic/actor state."""
    out = {"grad_steps": float(getattr(agent, "learn_step_cntr", 0))}

    if hasattr(agent, "log_alpha"):                    # SAC temperature
        out["sac_alpha"] = float(agent.alpha)
    if hasattr(agent, "actor") and hasattr(agent.actor, "log_std"):
        ls = agent.actor.log_std
        if isinstance(ls, T.nn.Parameter):             # PPO: state-independent
            out["policy_std_mean"] = float(ls.detach().exp().mean())

    mem = getattr(agent, "memory", None)
    n = min(getattr(mem, "mem_cntr", 0), getattr(mem, "mem_size", 0) or 0)
    if mem is None or n < batch:
        return out
    out["buffer_fill"] = float(n)

    try:
        s, a, r, s2, d, _, _ = mem.sample_buffer_per(batch)
    except Exception:
        return out

    dev = agent.actor.device
    to = lambda x: T.tensor(x, dtype=T.float, device=dev)
    s_t, a_t, r_t = to(s), to(a), to(r)

    with T.no_grad():
        # Actor behaviour. Saturation is the signal that matters for INT8
        # deployment (§8.3 — error hides in saturated dims, bites in mid-range).
        #
        # Deterministic AND sampled magnitudes are both recorded: their RATIO
        # is the exploration magnitude relative to the policy signal, and it is
        # the diagnostic that caught a SAC log_std init emitting sigma ~= 1.0,
        # where noise ran ~6x the signal and silently destroyed the warm start.
        # `action_abs_mean` alone cannot show this — it is the deterministic
        # path and looks healthy either way.
        act = agent.actor(s_t)
        out["action_abs_mean"] = float(act.abs().mean())
        out["action_saturated_frac"] = float((act.abs() > 0.99).float().mean())
        out["action_gripper_mean"] = float(act[:, -1].mean())

        explore = None
        if hasattr(agent.actor, "sample"):                 # SAC
            explore, _ = agent.actor.sample(s_t)
        elif hasattr(agent.actor, "dist"):                 # PPO
            explore = agent.actor.dist(s_t).sample()
        if explore is not None:
            out["action_explore_abs_mean"] = float(explore.abs().mean())
            out["explore_ratio"] = float(explore.abs().mean()
                                         / max(1e-6, act.abs().mean()))

        c1 = getattr(agent, "critic_1", None)
        if c1 is not None:
            q_data = c1(s_t, a_t)
            q_pi = c1(s_t, act)
            out["q_data_mean"] = float(q_data.mean())
            out["q_data_std"] = float(q_data.std())
            out["q_policy_mean"] = float(q_pi.mean())
            # Overestimation proxy: how far the policy's own Q sits above the
            # Q of actions actually taken.
            out["q_gap"] = float((q_pi - q_data).mean())
            out["reward_mean"] = float(r_t.mean())
        elif hasattr(agent, "critic"):                 # PPO value net
            out["v_mean"] = float(agent.critic(s_t).mean())

    return out
