#!/usr/bin/env python3
"""Figure 6: does resetting the critic head cure the universal decay?

Phase A showed every algorithm decaying after its peak, PPO included -- and PPO
has no replay buffer, which falsifies stale replay composition as the cause.
The surviving explanation is plasticity loss: as the policy improves, its own
data narrows onto near-success trajectories and the critic over-specialises.

This plots the reset arms against their seed-matched no-reset controls. For
td3_ln the two runs are bit-identical until the first reset (numpy-seeded
exploration noise), so the gap after it is caused by the reset alone; SAC
samples on the GPU and is not bit-reproducible, so its gap is suggestive only.

Read it as: dip at each reset then recovery to a HIGHER plateau => plasticity
loss. No systematic gap => the decay is intrinsic to the task.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_thesis import (style, recessive, save, SERIES, INK, INK_2,
                         INK_MUTED, GRID)

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
ARMS = [("td3_ln", "TD3 + LayerNorm", True), ("sac", "SAC", False)]


# The 100-episode rolling window is not full until episode 100; before that
# success_100 averages a handful of episodes and spikes to 100%. Every read of
# the curve -- including the control's peak -- must start after it fills.
WARMUP = 100


def load(run):
    p = os.path.join(LOGS, f"grasp_rand_{run}/episodes.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df[df["episode"] >= WARMUP].reset_index(drop=True)


def reset_points(df, every=25000):
    """Episodes at which a reset fired, from the logged gradient-step count."""
    out, nxt = [], every
    for ep, gs in zip(df["episode"], df["grad_steps"]):
        while gs >= nxt:
            out.append(ep); nxt += every
    return out


def main():
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), sharey=True)
    for ax, (algo, nice, decisive) in zip(axes, ARMS):
        ctrl = load(f"{algo}_phaseA_s0")
        rst = load(f"{algo}_resetA_s0")
        if ctrl is None or rst is None:
            ax.set_visible(False); continue

        for ep in reset_points(rst):
            ax.axvline(ep, color=GRID, lw=1.0, zorder=0)

        ax.plot(ctrl["episode"], ctrl["success_100"], color=INK_MUTED,
                lw=1.9, label="no reset (control)", zorder=2)
        ax.plot(rst["episode"], rst["success_100"], color=SERIES[0],
                lw=2.1, label="critic reset / 25k updates", zorder=3)

        # Direct labels: identity must not rest on colour alone.
        for df, col, txt in ((ctrl, INK_MUTED, "control"),
                             (rst, SERIES[0], "reset")):
            ax.annotate(txt, (df["episode"].iloc[-1], df["success_100"].iloc[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=col, fontsize=9, va="center", fontweight="bold")

        cb = ctrl["best_metric"].iloc[-1]
        peak = ctrl["success_100"].max()
        ax.axhline(peak, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.annotate(f"control peak {peak:.0f}%", (ctrl["episode"].max(), peak),
                    xytext=(-4, 4), textcoords="offset points", ha="right",
                    color=INK_MUTED, fontsize=8.5)
        title = f"{nice}" + ("  (bit-identical until first reset)" if decisive
                             else "  (GPU sampling: not bit-reproducible)")
        ax.set_title(title, loc="left")
        ax.set_xlabel("episode")
        ax.set_ylim(0, 100)
        ax.text(0.02, 0.04, f"control best metric {cb:.3f}   "
                            f"reset {rst['best_metric'].iloc[-1]:.3f}",
                transform=ax.transAxes, color=INK_2, fontsize=8.5)
        recessive(ax)
    axes[0].set_ylabel("grasp success, 100-episode rolling  (%)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Critic-head resets recover the decayed policy - the decay is "
                 "plasticity loss, not a task contradiction",
                 x=0.005, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.005, -0.04, "Vertical rules mark resets. Dashed line is the "
             "control's own peak. Same seed, same warm start, same schedule; "
             "only --critic-reset-every differs.",
             ha="left", fontsize=8.5, color=INK_MUTED)
    save(fig, os.path.join("Results", "figures"), "fig6_critic_reset_ablation")


if __name__ == "__main__":
    main()
