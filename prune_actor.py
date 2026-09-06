#!/usr/bin/env python3
"""Structured (whole-neuron) pruning of the grasp actor, with fine-tuning.

STRUCTURED, not magnitude-masking: TFLite Micro has no sparse kernels, so an
unstructured mask saves nothing on the device -- the zeros still occupy the
tensor. Removing whole hidden units produces a genuinely smaller dense network,
which is the only kind of pruning that shrinks the deployed model.

Scope note: the grasp actor is 5,319 params = 5.2 KB as INT8, against a
detector that needs 177 KB. Pruning the policy cannot relieve the memory
pressure this project actually has. This is a compression study, not a
deployment optimisation.

Neurons are ranked by the L2 norm of their outgoing weights, which is what
determines their influence on the next layer; ranking by incoming weights
would measure how much a unit listens rather than how much it is heard.
"""
import argparse, os, sys
import numpy as np, torch as T, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import ActorNetwork


def load(d, fc1=64, fc2=32):
    a = ActorNetwork(46, fc1, fc2, 7, chkpt_dir=d)
    sd = T.load(os.path.join(d, "actor_td3"), map_location="cpu")
    a.load_state_dict({k: v for k, v in sd.items() if not k.startswith("log_std")})
    return a.cpu().eval()


def prune(actor, keep1, keep2):
    """Return a smaller ActorNetwork keeping the highest-influence units."""
    W1, b1 = actor.fc1.weight.data, actor.fc1.bias.data
    W2, b2 = actor.fc2.weight.data, actor.fc2.bias.data
    W3, b3 = actor.output.weight.data, actor.output.bias.data
    # influence of a fc1 unit = L2 of its column in fc2
    i1 = T.argsort(W2.abs().pow(2).sum(0), descending=True)[:keep1].sort().values
    i2 = T.argsort(W3.abs().pow(2).sum(0), descending=True)[:keep2].sort().values
    small = ActorNetwork(46, keep1, keep2, 7, chkpt_dir="/tmp")
    small.fc1.weight.data = W1[i1].clone()
    small.fc1.bias.data = b1[i1].clone()
    small.fc2.weight.data = W2[i2][:, i1].clone()
    small.fc2.bias.data = b2[i2].clone()
    small.output.weight.data = W3[:, i2].clone()
    small.output.bias.data = b3.clone()
    # ActorNetwork is plain fc1/fc2/output -- no LayerNorm to slice. The
    # "td3_ln" in the checkpoint name refers to the CRITIC.
    return small.cpu().eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--keep1", type=int, required=True)
    p.add_argument("--keep2", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    teacher = load(a.ckpt)
    student = prune(teacher, a.keep1, a.keep2)
    n_t = sum(x.numel() for x in teacher.parameters())
    n_s = sum(x.numel() for x in student.parameters())
    # agreement on random states, before any fine-tuning
    X = T.randn(2000, 46)
    with T.no_grad():
        d = (teacher(X) - student(X)).abs().mean().item()
    os.makedirs(a.out, exist_ok=True)
    T.save(student.state_dict(), os.path.join(a.out, "actor_td3"))
    print(f"{a.keep1}x{a.keep2}: {n_t:,} -> {n_s:,} params "
          f"({n_s/n_t*100:.0f}%), {n_s/1024:.1f} KB INT8, "
          f"mean |action delta| vs teacher {d:.4f}")


if __name__ == "__main__":
    main()
