#!/usr/bin/env python3
"""Early stopping for place training on periodic END-TO-END evaluation.

    python es_watch.py --runs checkpoints/td3_place_A checkpoints/td3_place_B \
                       --out Results/place_es

For every run directory, each complete snapshot (snapshots/ep_XXXXX, written by
train_place.py --snapshot-every) is scored through fsm_sim.py on the selection
seeds -- by default cereal, the cereal_alignwarm_s0 grasp, --regrasp --rc-steps 60,
--rot-anchor-eps 1e-6, robosuite criterion, --settle-steps 60, seeds 7 and 31 x 50.

STOPPING RULE.  Among the leading run of fully scored snapshots (in episode order),
the best is the highest success rate (ties -> earliest).  Once `patience` later
snapshots have been scored without beating it, and the latest is at or beyond
--min-episode, the watcher writes <run>/STOP.  The trainer (started with
--stop-file <run>/STOP) exits at its next step.  The run's selected policy is the
best snapshot, not the weights it stopped with.

Snapshots are written to ep_XXXXX.tmp and renamed, so scoring them while the
trainer runs is safe.  Trainers are detected by /proc comm (python*) plus the
cmdline, never by pgrep -f.  Resumable: evaluations already in <out>/eval.tsv are
not repeated.

Outputs: <out>/eval.tsv (run, episode, seed, ok, n, rested),
         <out>/decisions.tsv (run, event, best_episode, best_pct, latest_episode, time).
"""
import argparse, csv, os, re, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = "/home/fahim/Thesis_fahim/venv/bin/python"


def trainer_alive(run):
    base = os.path.basename(os.path.normpath(run))
    pat = re.compile(re.escape(base) + r"(/|\s|$)")
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if not open(f"/proc/{pid}/comm").read().startswith("python"):
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "train_place.py" in cmd and pat.search(cmd):
            return True
    return False


def snapshots(run):
    d = os.path.join(run, "snapshots")
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = re.fullmatch(r"ep_(\d{5})", name)
        if m and os.path.exists(os.path.join(d, name, "actor_td3")):
            out.append(int(m.group(1)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="place checkpoint dirs, relative to the repo root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 31])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-episode", type=int, default=1000)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--object-type", default="cereal")
    ap.add_argument("--grasp-ckpt", default="checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best")
    ap.add_argument("--rot-anchor-eps", default="1e-6")
    ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs(os.path.join(a.out, "csv"), exist_ok=True)
    ev_path, dec_path = os.path.join(a.out, "eval.tsv"), os.path.join(a.out, "decisions.tsv")

    score = {}
    if os.path.exists(ev_path):
        for line in open(ev_path):
            f = line.rstrip("\n").split("\t")
            if len(f) == 6 and f[4] not in ("", "0"):
                score[(f[0], int(f[1]), int(f[2]))] = (int(f[3]), int(f[4]), int(f[5]))
    stopped = {r: os.path.exists(os.path.join(r, "STOP")) for r in a.runs}
    finished = set()
    running = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def log_decision(run, event, best_ep, best_pct, latest):
        with open(dec_path, "a") as fh:
            fh.write(f"{run}\t{event}\t{best_ep}\t{best_pct:.1f}\t{latest}\t{time.strftime('%F %T')}\n")

    while True:
        for p in list(running):
            if p.poll() is None:
                continue
            run, ep, s, path = running.pop(p)
            ok = n = rest = 0
            if os.path.exists(path):
                rows = list(csv.DictReader(open(path)))
                n = len(rows)
                ok = sum(int(r["success"]) for r in rows)
                rest = sum(int(r["rested"] or 0) for r in rows)
            score[(run, ep, s)] = (ok, n, rest)
            with open(ev_path, "a") as fh:
                fh.write(f"{run}\t{ep}\t{s}\t{ok}\t{n}\t{rest}\n")

        queued = {v[:3] for v in running.values()}
        todo = sorted((ep, run, s) for run in a.runs for ep in snapshots(run) for s in a.seeds
                      if (run, ep, s) not in score and (run, ep, s) not in queued)
        launched = 0
        for ep, run, s in todo:
            if len(running) >= a.parallel:
                break
            base = os.path.basename(os.path.normpath(run))
            path = os.path.join(a.out, "csv", f"{base}_ep{ep:05d}_s{s}.csv")
            cmd = [PY, "fsm_sim.py", "--grasp-ckpt", a.grasp_ckpt,
                   "--place-ckpt", os.path.join(run, "snapshots", f"ep_{ep:05d}"),
                   "--object-type", a.object_type, "--regrasp", "--rc-steps", "60",
                   "--rot-anchor-eps", a.rot_anchor_eps, "--success-criterion", "robosuite",
                   "--settle-steps", "60", "--episodes", str(a.episodes), "--seed", str(s),
                   "--out", path]
            running[subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)] = (run, ep, s, path)
            launched += 1

        alive = {r: trainer_alive(r) for r in a.runs}
        for run in a.runs:
            prefix = []
            for ep in snapshots(run):
                if all((run, ep, s) in score for s in a.seeds):
                    prefix.append(ep)
                else:
                    break
            if not prefix:
                continue
            rates = [sum(score[(run, ep, s)][0] for s in a.seeds) / max(1, sum(score[(run, ep, s)][1] for s in a.seeds))
                     for ep in prefix]
            bi = max(range(len(rates)), key=lambda i: (rates[i], -i))
            if not stopped[run] and len(rates) - 1 - bi >= a.patience and prefix[-1] >= a.min_episode:
                with open(os.path.join(run, "STOP"), "w") as fh:
                    fh.write(f"best snapshot ep_{prefix[bi]:05d} at {100*rates[bi]:.1f}%; "
                             f"{len(rates)-1-bi} later snapshots up to ep {prefix[-1]} did not beat it\n")
                stopped[run] = True
                log_decision(run, "STOP", prefix[bi], 100 * rates[bi], prefix[-1])
            pending = [ep for ep in snapshots(run) if ep not in prefix]
            if run not in finished and not alive[run] and not pending \
                    and not any(v[0] == run for v in running.values()):
                finished.add(run)
                log_decision(run, "FINAL" if stopped[run] else "FINAL_NO_STOP", prefix[bi], 100 * rates[bi], prefix[-1])

        if not running and len(todo) == launched and len(finished) == len(a.runs):
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
