#!/usr/bin/env python3
"""Thesis figures from the training / ablation logs.

Reads the CSVs written by `train_rand.py` (episodes.csv, probe.csv,
manifest.json) and `sweep_perception.py` (perception_sweep.csv) and emits
publication figures as both PNG (300 dpi) and PDF (vector, for LaTeX).

Figures
  fig1_learning_curves   rolling success + score per algorithm
  fig2_done_reasons      done-reason composition over training (per run)
  fig3_spawn_heatmap     success rate binned over the spawn box (per run)
  fig4_probe             critic Q, action saturation, SAC temperature
  fig5_perception        periodic-perception ablation (recompute vs frozen)

Usage
  python3 plot_thesis.py                          # everything it can find
  python3 plot_thesis.py --runs logs/grasp_rand_*_phaseA_s0
  python3 plot_thesis.py --sweep Results/perception_sweep.csv
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# --- validated palette (see dataviz references/palette.md) ------------------
# Categorical slots 1-4, light mode. Validated with scripts/validate_palette.js:
# all checks PASS on the adjacent pairlist (worst CVD dE 9.1, normal 22.9).
# The contrast WARN on slots 3-4 obliges relief -> every series is DIRECT
# LABELLED as well as legended, so identity is never carried by colour alone.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_EXTRA = ["#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"

CMAP_BLUE = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# Done-reason order is fixed so colour follows the entity, never its rank.
REASON_ORDER = ["success", "timeout_flicker", "timeout_touched",
                "timeout_no_reach"]
REASON_LABEL = {"success": "Success",
                "timeout_flicker": "Flicker (marginal grip)",
                "timeout_touched": "Touched, no grasp",
                "timeout_no_reach": "Never reached"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "lines.linewidth": 2.0,
        "figure.dpi": 110,
    })


def recessive(ax):
    """Grid and axes must sit behind the data, not compete with it."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_axisbelow(True)


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def load_runs(patterns):
    runs = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            ep = os.path.join(d, "episodes.csv")
            if not os.path.exists(ep):
                continue
            try:
                df = pd.read_csv(ep)
            except Exception:
                continue
            if df.empty:
                continue
            man = {}
            mp = os.path.join(d, "manifest.json")
            if os.path.exists(mp):
                man = json.load(open(mp))
            label = man.get("algo") or os.path.basename(d)
            runs.append(dict(dir=d, df=df, manifest=man, label=label))
    return runs


def roll(s, w=100):
    return pd.Series(s).rolling(w, min_periods=max(5, w // 10)).mean()


# --- fig 1: learning curves -------------------------------------------------

def fig_learning(runs, outdir, window=100):
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax in axes:
        recessive(ax)

    for i, r in enumerate(runs):
        c = (SERIES + SERIES_EXTRA)[i % 8]
        df, x = r["df"], r["df"]["episode"]
        y1 = roll(df["grasp_success"], window) * 100
        y2 = roll(df["score"], window)
        axes[0].plot(x, y1, color=c, label=r["label"])
        axes[1].plot(x, y2, color=c, label=r["label"])
        # Direct label at the line end — the relief the contrast WARN requires.
        for ax, y in ((axes[0], y1), (axes[1], y2)):
            if y.notna().any():
                xi = x.iloc[y.last_valid_index()]
                ax.annotate(r["label"], (xi, y.iloc[y.last_valid_index()]),
                            xytext=(5, 0), textcoords="offset points",
                            color=c, fontsize=8.5, va="center",
                            fontweight="bold")

    axes[0].set_title(f"Grasp success (rolling {window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_title(f"Episode score (rolling {window})")
    # Right-hand headroom so end-of-line direct labels are never clipped.
    for ax in axes:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.18)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    if len(runs) >= 2:
        axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, outdir, "fig1_learning_curves")


# --- fig 2: done-reason composition ----------------------------------------

def fig_done_reasons(runs, outdir, window=50):
    runs = [r for r in runs if "done_reason" in r["df"].columns]
    if not runs:
        return
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), squeeze=False)
    for ax, r in zip(axes[0], runs):
        recessive(ax)
        df = r["df"]
        idx, series = [], {k: [] for k in REASON_ORDER}
        for start in range(0, len(df) - window + 1, window):
            win = df["done_reason"].iloc[start:start + window]
            idx.append(df["episode"].iloc[start + window - 1])
            for k in REASON_ORDER:
                series[k].append((win == k).sum() / len(win) * 100)
        if not idx:
            ax.text(0.5, 0.5, "not enough episodes", ha="center",
                    transform=ax.transAxes, color=INK_MUTED)
            continue
        ax.stackplot(idx, [series[k] for k in REASON_ORDER],
                     colors=SERIES[:len(REASON_ORDER)],
                     labels=[REASON_LABEL[k] for k in REASON_ORDER],
                     edgecolor=SURFACE, linewidth=2.0)  # 2px surface gap
        ax.set_title(f"{r['label']} — outcome composition")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Share of episodes (%)")
        ax.set_ylim(0, 100)
        ax.margins(x=0)
    axes[0][0].legend(loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, outdir, "fig2_done_reasons")


# --- fig 3: success vs spawn position --------------------------------------

def fig_spawn_heatmap(runs, outdir, bins=7):
    runs = [r for r in runs
            if {"spawn_x", "spawn_y"} <= set(r["df"].columns)
            and r["df"]["spawn_x"].notna().any()]
    if not runs:
        return
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.9), squeeze=False)
    for ax, r in zip(axes[0], runs):
        recessive(ax)
        df = r["df"].dropna(subset=["spawn_x", "spawn_y"])
        if df.empty:
            continue
        stat, xe, ye = np.histogram2d(
            df["spawn_x"], df["spawn_y"], bins=bins,
            weights=df["grasp_success"].astype(float))
        cnt, _, _ = np.histogram2d(df["spawn_x"], df["spawn_y"], bins=[xe, ye])
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(cnt > 0, stat / cnt * 100, np.nan)
        im = ax.imshow(rate.T, origin="lower", cmap=CMAP_BLUE,
                       vmin=0, vmax=100, aspect="auto",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]])
        # Direct value labels — relief, and a thesis reader wants the number.
        for ix in range(rate.shape[0]):
            for iy in range(rate.shape[1]):
                if np.isnan(rate[ix, iy]):
                    continue
                cx = (xe[ix] + xe[ix + 1]) / 2
                cy = (ye[iy] + ye[iy + 1]) / 2
                ax.text(cx, cy, f"{rate[ix, iy]:.0f}", ha="center",
                        va="center", fontsize=7,
                        color="#ffffff" if rate[ix, iy] > 55 else INK)
        ax.set_title(f"{r['label']} — success by spawn position")
        ax.set_xlabel("spawn x (m)")
        ax.set_ylabel("spawn y (m)")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Success rate (%)", color=INK_2, fontsize=8.5)
        cb.outline.set_visible(False)
    fig.tight_layout()
    save(fig, outdir, "fig3_spawn_heatmap")


# --- fig 4: agent internals -------------------------------------------------

def fig_probe(runs, outdir):
    probes = []
    for r in runs:
        p = os.path.join(r["dir"], "probe.csv")
        if os.path.exists(p):
            try:
                d = pd.read_csv(p)
            except Exception:
                continue
            if not d.empty:
                probes.append((r["label"], d))
    if not probes:
        return
    panels = [("q_data_mean", "Critic Q (actions taken)"),
              ("q_gap", "Q overestimation gap"),
              ("action_saturated_frac", "Actions saturated (|a|>0.99)"),
              ("sac_alpha", "SAC temperature")]
    avail = [(k, t) for k, t in panels if any(k in d.columns for _, d in probes)]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(3.7 * len(avail), 3.4),
                             squeeze=False)
    for ax, (key, title) in zip(axes[0], avail):
        recessive(ax)
        for i, (label, d) in enumerate(probes):
            if key not in d.columns or d[key].notna().sum() == 0:
                continue
            c = (SERIES + SERIES_EXTRA)[i % 8]
            ax.plot(d["episode"], d[key], color=c, label=label)
        ax.set_title(title)
        ax.set_xlabel("Episode")
    if len(probes) >= 2:
        axes[0][0].legend(loc="best")
    fig.tight_layout()
    save(fig, outdir, "fig4_probe")


# --- fig 5: perception ablation --------------------------------------------

def fig_perception(sweep_csv, outdir):
    if not sweep_csv or not os.path.exists(sweep_csv):
        return
    df = pd.read_csv(sweep_csv)
    if df.empty:
        return
    noises = sorted(df["noise_pos_m"].unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(4.3 * len(noises), 3.7),
                             squeeze=False, sharey=True)
    modes = ["recompute", "frozen"]
    for ax, noi in zip(axes[0], noises):
        recessive(ax)
        sub = df[df["noise_pos_m"] == noi]
        for i, mode in enumerate(modes):
            m = sub[sub["mode"] == mode].sort_values("period")
            if m.empty:
                continue
            ax.plot(m["period"], m["success_pct"], color=SERIES[i],
                    marker="o", markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2, label=mode)   # 2px surface ring
            last = m.iloc[-1]
            ax.annotate(mode, (last["period"], last["success_pct"]),
                        xytext=(6, 0), textcoords="offset points",
                        color=SERIES[i], fontsize=8.5, va="center",
                        fontweight="bold")
        ax.set_title(f"Position noise {noi * 1000:.0f} mm")
        ax.set_xlabel("Perception period (control steps)")
        ax.set_ylim(0, 105)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.30)   # room for direct labels
    axes[0][0].set_ylabel("Grasp success (%)")
    axes[0][0].legend(loc="lower left")
    fig.suptitle("Periodic perception: proprioceptive recompute vs frozen block",
                 fontsize=11.5, fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    save(fig, outdir, "fig5_perception")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["logs/grasp_rand_*"])
    p.add_argument("--sweep", default="Results/perception_sweep.csv")
    p.add_argument("--outdir", default="Results/figures")
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args()

    style()
    runs = load_runs(args.runs)
    print(f"found {len(runs)} run(s): {[r['label'] for r in runs]}")
    fig_learning(runs, args.outdir, args.window)
    fig_done_reasons(runs, args.outdir)
    fig_spawn_heatmap(runs, args.outdir)
    fig_probe(runs, args.outdir)
    fig_perception(args.sweep, args.outdir)
    print(f"\nfigures in {args.outdir}/")


if __name__ == "__main__":
    main()
