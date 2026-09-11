#!/usr/bin/env python3
"""Passive joint-limit probe around fsm_sim.main(): sizes the BLOCKED class.

Runs fsm_sim.py's own main() with the exact CLI given, so every success/failure
is produced by the real harness (respawn-and-retry loop included), and records
for each scored episode what the carry looked like at the end of TRANSPORT.
The hooks only READ sim.data after fsm_sim's own calls; nothing is written.

    python anchor_probe.py --probe-out X.probe.csv <fsm_sim.py arguments>

BLOCKED (Results/transport_stall_diagnosis.txt, cause 2): the episode fails out
of TRANSPORT while still holding the object, commanding motion (act_tail > 0.5)
that produces almost none (object path over the last 50 TRANSPORT steps
< 0.02 m).  PINNED: blocked, and some arm joint within 0.02 rad of its limit at
the last TRANSPORT step (joint index 5 is the one-sided wrist joint of §9.17).

Positive controls, both required before trusting a count on c2m512_s1:
  * per-episode success here == plain fsm_sim.py on the same seed (passivity)
  * bi_s0 at --rot-anchor-eps 0 flags ~3% blocked, mostly joint 5: the class
    ROT_ANCHOR_EPS recovered (+2.92 of 3.08%, Results/orientation_anchor.txt)
"""
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
argv = sys.argv[1:]
if "--probe-out" not in argv:
    sys.exit("usage: anchor_probe.py --probe-out FILE <fsm_sim.py arguments>")
i = argv.index("--probe-out"); PROBE_OUT = argv[i + 1]; del argv[i:i + 2]
sys.argv = ["fsm_sim.py"] + argv

import fsm_sim as F
from robosuite.wrappers import GymWrapper

TAIL = 50
S = dict(pre=None, post=None, grasped=False, cur=None)
records = []


def _finalize():
    c = S["cur"]; S["cur"] = None
    if not c or not c["tr"]:
        return                      # attempt never reached TRANSPORT: not scored
    tr = c["tr"]; tail = tr[-TAIL:]
    P = np.array([t[0] for t in tail])
    path = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0
    act = float(np.mean([t[1] for t in tail]))
    pos, _, grasped, marg, q5 = tr[-1]
    j = int(np.argmin(marg))
    final, fail_from = c["final"], c["fail_from"]
    blocked = bool(final == F.FAIL and fail_from == F.TRANSPORT and grasped
                   and path < 0.02 and act > 0.5)
    records.append(dict(
        idx=len(records),
        final=F.NAMES[final] if final is not None else "",
        fail_from=F.NAMES[fail_from] if fail_from is not None else "",
        tr_steps=len(tr), grasped_end=int(grasped),
        path_tail=round(path, 4), act_tail=round(act, 3),
        d_end=round(float(np.hypot(pos[0] - F.BIN_X, pos[1] - F.BIN_Y)), 4),
        margin_end=round(float(marg[j]), 4), joint_end=j, q5_end=round(q5, 4),
        margin_min_tr=round(float(min(t[3].min() for t in tr)), 4),
        blocked=int(blocked), pinned=int(blocked and marg[j] < 0.02)))


_orig_fsm_step = F.FSM.step


def _fsm_step(self, s, grasped, placed, grasp_actor, place_actor):
    pre = self.phase
    act = _orig_fsm_step(self, s, grasped, placed, grasp_actor, place_actor)
    S["pre"], S["post"], S["grasped"] = pre, self.phase, grasped
    return act


_orig_reset = GymWrapper.reset


def _reset(self, *a, **k):
    _finalize()
    S["cur"] = dict(tr=[], final=None, fail_from=None)
    S["pre"] = S["post"] = None
    return _orig_reset(self, *a, **k)


_orig_step = GymWrapper.step


def _step(self, action, *a, **k):
    out = _orig_step(self, action, *a, **k)
    c = S["cur"]
    if c is not None and S["pre"] is not None:
        raw = self.env
        if S["pre"] == F.TRANSPORT:
            m, d = raw.sim.model, raw.sim.data
            ids = [m.joint_name2id(n) for n in raw.robots[0].robot_model.joints]
            q = np.array([d.qpos[m.jnt_qposadr[i]] for i in ids])
            rng = m.jnt_range[ids]
            marg = np.minimum(q - rng[:, 0], rng[:, 1] - q)
            bid = raw.obj_body_id[raw.objects[raw.object_id].name]
            c["tr"].append((np.array(d.body_xpos[bid][:3]),
                            float(np.linalg.norm(np.asarray(action)[0:3])),
                            bool(S["grasped"]), marg, float(q[5])))
        if S["post"] in (F.OK, F.FAIL) and c["final"] is None:
            c["final"] = S["post"]
            c["fail_from"] = S["pre"] if S["post"] == F.FAIL else None
        S["pre"] = None
    return out


F.FSM.step = _fsm_step
GymWrapper.reset = _reset
GymWrapper.step = _step

F.main()
_finalize()
keys = list(records[0].keys()) if records else ["idx"]
with open(PROBE_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(records)
nb = sum(r["blocked"] for r in records); npin = sum(r["pinned"] for r in records)
joints = sorted({r["joint_end"] for r in records if r["pinned"]})
print(f"probe: {len(records)} scored, blocked {nb}, pinned {npin} (joints {joints}) -> {PROBE_OUT}")
