#!/usr/bin/env python3
"""Net2WiderNet (Chen et al. 2016) for the decomposed grasp actor.

Why this exists. Every random-spawn grasp policy in this project sits at the
end of a 5-stage, 26k-episode warm-start curriculum (backup -> phaseA ->
resetA -> phaseBr -> liftcert_phaseB -> bi/wclip/qat). A wider actor cannot
inherit any of it -- shape mismatch at every layer -- and an aborted attempt to
train wide-from-scratch confirmed the obvious: cold starts fail at BOTH widths,
so they say nothing about capacity. Net2WiderNet transfers the trained policy
into a wider net EXACTLY, so training resumes rather than restarts.

The transform. To widen a layer's output from n to N, pick a mapping
    g(j) = j                      for j < n        (keep the originals)
    g(j) = uniform from [0, n)    for j >= n       (duplicate a random unit)
then
    W1'[j, :] = W1[g(j), :]       b1'[j] = b1[g(j)]
    W2'[:, j] = W2[:, g(j)] / c[g(j)]      c[k] = |{j : g(j) = k}|
Dividing the OUTGOING weights by each unit's replication count keeps every
downstream pre-activation identical, so the network computes exactly the same
function. Valid here because the actor is plain Linear/ReLU/tanh with no
normalisation: a LayerNorm between the layers would break it, since LN
renormalises over a feature count that duplication changes.

Duplicated units are perfectly symmetric: identical activations and identical
outgoing weights mean identical gradients forever, so the added capacity would
never be used. Instead of perturbing weights (which breaks the function --
measured max|d| 0.46 at 0.5% noise), the outgoing weights are split UNEQUALLY
between duplicates. The shares still sum to 1 per source unit, so the function
is preserved exactly, while the differing shares give the duplicates different
gradients from the first step.
"""
import argparse, glob, os, shutil
import numpy as np, torch as T


def widen(sd, n1_new, n2_new, rng, split_noise=0.5):
    # Arithmetic in float64: the /c[g(j)] division is the only lossy step, and
    # in float32 it leaves a ~4e-5 residual on the output. Verified exact in
    # double (max|d| 1e-13), so we divide in double and cast back once.
    dt = sd["fc1.weight"].dtype
    W1, b1 = sd["fc1.weight"].double(), sd["fc1.bias"].double()
    W2, b2 = sd["fc2.weight"].double(), sd["fc2.bias"].double()
    W3, b3 = sd["output.weight"].double(), sd["output.bias"].double()
    n1, n2 = W1.shape[0], W2.shape[0]
    if n1_new < n1 or n2_new < n2:
        raise SystemExit(f"Net2Wider only widens: have {n1}/{n2}, asked {n1_new}/{n2_new}")

    def mapping(n, N):
        """Unit mapping plus each new unit's SHARE of its source's outgoing
        weights. Shares within a group sum to 1, so the function is preserved
        exactly; making them UNEQUAL is what breaks the symmetry.

        Equal shares (the textbook 1/c) leave duplicates with identical
        activations AND identical outgoing weights, so they receive identical
        gradients forever and the added capacity is never used. Perturbing the
        duplicates' incoming weights instead does break the tie, but destroys
        the function: measured max|d| 0.46 at only 0.5% noise, against a 0.1
        exploration scale. Unequal shares cost nothing and work.
        """
        g = np.arange(N)
        if N > n:
            g[n:] = rng.integers(0, n, size=N - n)
        share = np.ones(N, dtype=np.float64)
        if split_noise > 0:
            share += split_noise * rng.uniform(-1.0, 1.0, size=N)
        tot = np.zeros(n, dtype=np.float64)
        np.add.at(tot, g, share)
        return g, (share / tot[g])

    # ---- layer 1: widen fc1's output, compensate on fc2's input -----------
    g1, sh1 = mapping(n1, n1_new)
    W1n, b1n = W1[g1, :], b1[g1]
    W2n = W2[:, g1] * T.tensor(sh1, dtype=W2.dtype).unsqueeze(0)

    # ---- layer 2: widen fc2's output, compensate on output's input --------
    g2, sh2 = mapping(n2, n2_new)
    W2n, b2n = W2n[g2, :], b2[g2]
    W3n = W3[:, g2] * T.tensor(sh2, dtype=W3.dtype).unsqueeze(0)



    out = dict(sd)
    out.update({"fc1.weight": W1n.to(dt), "fc1.bias": b1n.to(dt),
                "fc2.weight": W2n.to(dt), "fc2.bias": b2n.to(dt),
                "output.weight": W3n.to(dt), "output.bias": b3.to(dt)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="source checkpoint dir (has actor_td3)")
    p.add_argument("--dst", required=True, help="dir to write the widened checkpoint set")
    p.add_argument("--fc1", type=int, required=True)
    p.add_argument("--fc2", type=int, required=True)
    p.add_argument("--split-noise", type=float, default=0.5,
                   help="spread of the unequal outgoing-weight split that breaks "
                        "duplicate symmetry, in [0,1). The function is preserved "
                        "EXACTLY at any value; 0 means equal shares, which leaves "
                        "duplicates tied and the extra capacity unusable.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.dst, exist_ok=True)
    sys_path = os.path.dirname(os.path.abspath(__file__))
    import sys; sys.path.insert(0, sys_path)
    from networks import ActorNetwork

    for name in ("actor", "target_actor"):
        src = os.path.join(a.src, f"{name}_td3")
        if not os.path.exists(src):
            src = os.path.join(a.src, "actor_td3")
        sd = T.load(src, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("log_std")}
        n1, n2 = sd["fc1.weight"].shape[0], sd["fc2.weight"].shape[0]
        wide = widen(sd, a.fc1, a.fc2, np.random.default_rng(a.seed), a.split_noise)
        T.save(wide, os.path.join(a.dst, f"{name}_td3"))

        if name == "actor":                     # verify function preservation
            old = ActorNetwork(sd["fc1.weight"].shape[1], n1, n2,
                               sd["output.weight"].shape[0], chkpt_dir="/tmp")
            old.load_state_dict(sd); old.to("cpu"); old.eval()
            new = ActorNetwork(sd["fc1.weight"].shape[1], a.fc1, a.fc2,
                               sd["output.weight"].shape[0], chkpt_dir="/tmp")
            new.load_state_dict(wide); new.to("cpu"); new.eval()
            # Seeded: an unseeded probe makes the check stochastic, and the
            # same widening then reported 1.04e-4 on one draw and 3.8e-5 on the
            # next -- which is how seed 3 tripped a threshold it should not have.
            T.manual_seed(12345)
            x = T.randn(8192, sd["fc1.weight"].shape[1])
            with T.no_grad():
                d = (old(x) - new(x)).abs()
            print(f"  {n1}/{n2} -> {a.fc1}/{a.fc2}  params "
                  f"{sum(v.numel() for v in sd.values()):,} -> "
                  f"{sum(v.numel() for v in wide.values()):,}")
            print(f"  function preservation over 8192 seeded states: "
                  f"max|d|={float(d.max()):.3e}  mean|d|={float(d.mean()):.3e}")
            # float32 storage bounds how exact this can be; the transform
            # itself verifies to 1e-13 in double. Observed max|d| across six
            # seeds is 3.3e-5 to 1.04e-4 -- a 1e-4 threshold tripped seed 3 on
            # rounding alone. 5e-4 is still 200x below the 0.1 exploration
            # noise, so it catches a real algorithmic break without flagging
            # float32 dust. mean|d| (~1e-7) is the stable check.
            if float(d.max()) > 5e-4:
                raise SystemExit("FAILED: exact widening did not preserve the function")

    # critics keep their width and their training state; copy them unchanged
    n = 0
    for f in glob.glob(os.path.join(a.src, "*_td3")):
        if "actor" in os.path.basename(f):
            continue
        shutil.copy2(f, os.path.join(a.dst, os.path.basename(f))); n += 1
    print(f"  copied {n} critic checkpoints unchanged (critics are not widened)")
    print(f"  wrote {a.dst}")


if __name__ == "__main__":
    main()
