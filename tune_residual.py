#!/usr/bin/env python3
"""Tune the AprilTag residual corrector.

PROTOCOL. 2181 samples over 6 seeds is small enough that tuning against the
same folds you report on would overfit the search itself. So seeds are split:
  TUNE   7, 31, 47, 89   -- leave-one-seed-out CV inside this set only
  HOLD  101, 123         -- never seen by the search, used once at the end
The reported number is the held-out one; the CV number is only how the winner
was chosen.
"""
import csv, glob, itertools, json, os, sys
import numpy as np, torch as T, torch.nn as nn

F = ["reproj","area_px","obliq_deg","cam_range","gripper_perp","gripper_dz",
     "cx","cy","obj_yaw","det_x","det_y","det_z"]
TUNE = {"ds_s7","ds_s31","ds_s47","ds_s89"}
HOLD = {"ds_s101","ds_s123"}
dev = T.device("cuda" if T.cuda.is_available() else "cpu")


def load():
    X, Y, G = [], [], []
    for f in sorted(glob.glob("Results/tag_ds/ds_s*.csv")):
        tag = os.path.basename(f).replace(".csv", "")
        for r in csv.DictReader(open(f)):
            X.append([float(r[k]) for k in F])
            Y.append([float(r[k]) for k in ("res_x","res_y","res_z")])
            G.append(tag)
    # NOTE: targets are returned in MILLIMETRES so the sweep prints readable
    # numbers. Anything SAVED for tag_e2e must divide the final layer by 1000 --
    # tag_e2e adds the correction to a position in metres, and skipping this
    # displaced every detection by ~11.5 m and scored 0.0% on all 12 seeds.
    return (np.array(X, np.float32), np.array(Y, np.float32) * 1000,
            np.array(G))


def fit(Xtr, Ytr, Xte, h, depth, bs, ep, lr, wd, seed=0):
    T.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    xt = T.tensor((Xtr - mu) / sd, device=dev); yt = T.tensor(Ytr, device=dev)
    xe = T.tensor((Xte - mu) / sd, device=dev)
    layers, d = [], len(F)
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers += [nn.Linear(d, 3)]
    m = nn.Sequential(*layers).to(dev)
    o = T.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sch = T.optim.lr_scheduler.CosineAnnealingLR(o, T_max=ep)
    n = len(xt)
    for _ in range(ep):
        if bs and bs < n:
            idx = T.randperm(n, device=dev)
            for i in range(0, n, bs):
                b = idx[i:i+bs]
                o.zero_grad(); ((m(xt[b]) - yt[b])**2).mean().backward(); o.step()
        else:
            o.zero_grad(); ((m(xt) - yt)**2).mean().backward(); o.step()
        sch.step()
    m.eval()
    with T.no_grad():
        return m(xe).cpu().numpy(), (m, mu, sd)


def score(Y, P):
    e = np.linalg.norm(Y - P, axis=1)
    r2 = [1 - ((Y[:,j]-P[:,j])**2).sum() / ((Y[:,j]-Y[:,j].mean())**2).sum()
          for j in range(3)]
    return float(np.median(e)), float(e.mean()), r2


def main():
    X, Y, G = load()
    tune = np.isin(G, list(TUNE)); hold = np.isin(G, list(HOLD))
    print(f"tune {tune.sum()} samples / {len(TUNE)} seeds | "
          f"hold {hold.sum()} / {len(HOLD)} seeds | device {dev}")
    base = np.median(np.linalg.norm(Y[hold], axis=1))
    grid = list(itertools.product([64,128,256],[2,3],[64,256,0],
                                  [1500],[1e-3],[1e-4,1e-3]))
    results = []
    for h,depth,bs,ep,lr,wd in grid:
        preds = np.zeros_like(Y[tune]); gt = G[tune]
        for s in np.unique(gt):
            te = gt == s
            P,_ = fit(X[tune][~te], Y[tune][~te], X[tune][te],
                      h,depth,bs,ep,lr,wd)
            preds[te] = P
        med,_,r2 = score(Y[tune], preds)
        results.append((med,(h,depth,bs,ep,lr,wd),r2))
        print(f"  h{h:<4} d{depth} bs{bs if bs else 'full':<5} wd{wd:<6} "
              f"CV median {med:6.2f} mm  R2 {r2[0]:+.2f}/{r2[1]:+.2f}/{r2[2]:+.2f}",
              flush=True)
    results.sort(key=lambda t: t[0])
    med, best, _ = results[0]
    h,depth,bs,ep,lr,wd = best
    print(f"\nbest by CV: h{h} d{depth} bs{bs} wd{wd}  (CV {med:.2f} mm)")
    P, (m, mu, sd) = fit(X[tune], Y[tune], X[hold], *best)
    hm, hmean, hr2 = score(Y[hold], P)
    print(f"HELD-OUT seeds (never tuned on): baseline {base:.2f} mm -> "
          f"{hm:.2f} mm ({(1-hm/base)*100:+.0f}%)  R2 {hr2[0]:+.2f}/{hr2[1]:+.2f}/{hr2[2]:+.2f}")
    # refit on ALL data with the chosen config for deployment
    _, (mf, muf, sdf) = fit(X, Y, X[:1], *best)
    sdd = mf.state_dict(); li = [i for i,l in enumerate(mf) if isinstance(l, nn.Linear)]
    json.dump(dict(features=F, mu=muf.tolist(), sd=sdf.tolist(),
                   W=[sdd[f"{i}.weight"].cpu().tolist() for i in li],
                   b=[sdd[f"{i}.bias"].cpu().tolist() for i in li],
                   config=dict(h=h,depth=depth,bs=bs,ep=ep,lr=lr,wd=wd)),
              open("assets/tag_residual_mlp_tuned.json","w"))
    n=sum(p.numel() for p in mf.parameters())
    print(f"saved assets/tag_residual_mlp_tuned.json  ({n:,} params)")


if __name__ == "__main__":
    main()
