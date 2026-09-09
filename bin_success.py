#!/usr/bin/env python3
"""Object-aware placement success for robosuite's PickPlace.

WHY THIS EXISTS
robosuite's `PickPlace.not_in_bin` accepts an object only when

    bin2_pos[2] < obj_z < bin2_pos[2] + 0.1          # 0.80 < z < 0.90

That window is a constant, but the height of a box resting on the bin floor is
not: its centre sits at `bin_z + half_height`.

    object   half-height   resting centre   inside 0.80..0.90 ?
    Bread      0.045          0.845          yes, comfortably
    Can        0.060          0.860          yes
    Milk       0.085          0.885          barely
    Cereal     0.100          0.900          NO -- exactly the exclusive bound

A cereal box lying flat on the bin floor is therefore scored as NOT placed. It
counts only if it settles a few millimetres low or the check happens to be
sampled mid-bounce.  Measured (Results/orientation_anchor.txt): of 62 scored
failures, 60 came to rest inside the correct compartment; release accuracy and
release height were identical to the successes to four decimals; the entire
success/failure split was ~1.4 cm of resting height.  Physical placement was
598/600 = 99.7% against a scored 89.7%.

Because bread rests at 0.845 and cereal at 0.900, the same criterion is lenient
for one object and impossible for the other -- so every bread-vs-cereal
comparison in this project is confounded by it, and any reward built on
`_check_success` teaches a cereal policy to exploit the miscalibration rather
than to place the box.

THE FIX, AND WHY IT CANNOT INVALIDATE EXISTING RESULTS
Only the z upper bound moves, and only upward, and only when the object needs
it:

    upper = bin_z + max(0.1, half_height + Z_MARGIN)

    Bread  max(0.1, 0.075) = 0.1   -> unchanged
    Can    max(0.1, 0.090) = 0.1   -> unchanged
    Milk   max(0.1, 0.115) = 0.115 -> 0.915
    Cereal max(0.1, 0.130) = 0.130 -> 0.930

Bread and can are bit-identical to robosuite by construction, so every recorded
number for them stands.  The xy bounds and the gripper-clearance guard
(`r_reach < 0.6`, i.e. the gripper at least ~4.2 cm from the object, which is
what stops a still-held object counting as placed) are reproduced exactly from
`pick_place.py`.

ONE DEFINITION, TWO CONSUMERS
`place_env_wrapper.py` (the training reward) and `fsm_sim.py` (evaluation) both
import from here.  Training against one criterion and scoring against another
is the failure this module exists to prevent.
"""
import numpy as np

Z_MARGIN = 0.03          # clearance above a resting box centre, metres
R_REACH_MAX = 0.6        # robosuite's gripper-clearance guard: r_reach < 0.6


def object_half_height(raw_env, obj_index):
    """Half-height of an object, from its own model (bottom_offset is -half)."""
    return abs(float(raw_env.objects[obj_index].bottom_offset[2]))


def bin_z_window(raw_env, obj_index):
    """(low, high) z bounds for 'resting in the bin', object-aware."""
    bin_z = float(raw_env.bin2_pos[2])
    half = object_half_height(raw_env, obj_index)
    return bin_z, bin_z + max(0.1, half + Z_MARGIN)


def in_bin(raw_env, obj_index, require_gripper_clear=True):
    """True if object `obj_index` is in its target compartment.

    Mirrors pick_place.not_in_bin's xy bounds and _check_success's gripper
    guard exactly; only the z upper bound is object-aware.
    """
    pos = raw_env.sim.data.body_xpos[
        raw_env.obj_body_id[raw_env.objects[obj_index].name]]

    x_lo = raw_env.bin2_pos[0]
    y_lo = raw_env.bin2_pos[1]
    if obj_index in (0, 2):
        x_lo -= raw_env.bin_size[0] / 2
    if obj_index < 2:
        y_lo -= raw_env.bin_size[1] / 2
    if not (x_lo < pos[0] < x_lo + raw_env.bin_size[0] / 2):
        return False
    if not (y_lo < pos[1] < y_lo + raw_env.bin_size[1] / 2):
        return False

    z_lo, z_hi = bin_z_window(raw_env, obj_index)
    if not (z_lo < pos[2] < z_hi):
        return False

    if require_gripper_clear:
        eef = raw_env.sim.data.site_xpos[raw_env.robots[0].eef_site_id]
        r_reach = 1 - np.tanh(10.0 * np.linalg.norm(eef - pos))
        if not (r_reach < R_REACH_MAX):
            return False
    return True


def check_success(raw_env):
    """Drop-in replacement for PickPlace._check_success().

    Honours single_object_mode the same way: modes 1 and 2 succeed when any
    one object is placed, mode 0 when all are.
    """
    flags = [in_bin(raw_env, i) for i in range(len(raw_env.objects))]
    if getattr(raw_env, "single_object_mode", 0) in (1, 2):
        return bool(sum(flags) > 0)
    return bool(sum(flags) == len(flags))
