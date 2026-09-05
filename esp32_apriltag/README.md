# AprilTag detector for a plain ESP32 (no PSRAM)

Vendored from **stnk20/apriltag**, branch `esp-idf` (itself a fork of
AprilRobotics/apriltag), BSD 2-Clause — see `LICENSE.md`. Only the `tag16h5`
family is included; that is why the deployment path uses 16h5 (see
`Results/tag16h5_deployment.txt`).

## Why it needed modifying

The upstream port targets an **ESP32-CAM with 4 MB PSRAM**. This project's
board has none — the boot log reports `PSRAM: absent`, with ~320 KB of
internal SRAM free after the FSM, the two TFLite arenas and the corrector
(`Results/wrist_camera_route.txt`).

Peak heap was measured by compiling the detector on the host with a
`--wrap=malloc/calloc/realloc/free` interposer and running it over 60 real
320x240 wrist frames rendered from the sim (`esp32_apriltag/` + the probe in
this repo's history). Excluding the frame buffer itself:

| configuration | peak heap | tag detected |
|---|---|---|
| stock, 320x240, decimate 1 | **1675 KB** | 43/60 |
| stock, 320x240, decimate 2 |  516 KB | 43/60 |
| stock, 320x240, decimate 3 |  267 KB | 26/60 |

So the stock port overruns the budget by 5x at full resolution. Decimating to
3 fits but detection collapses, because the wrist tag is only ~20 px across
and a 1/3 subsample leaves ~7 px — below what 16h5's 6x6 grid can decode.

## What was changed

**1. `common/unionfind.h` — 8 B/px to 4 B/px.**
Upstream stores `{uint32 parent, uint32 size}` per pixel. At 320x240 that
single allocation is 600 KB. This version keeps parent and size in separate
`uint16` arrays when `maxid <= 65534`, and falls back to the upstream 32-bit
layout otherwise (so full-resolution 320x240 still works, just without the
saving). Separate arrays rather than a 6-byte struct, which would pad back to
8. `size` saturates at 65535, which is safe: it is only read to pick the
larger tree when joining and to filter clusters by size, and every cluster of
interest is orders of magnitude smaller.

*Verified function-preserving*: detection is 43/60 before and after, at both
decimate 1 and 2.

**2. `apriltag_quad_thresh.c` — `nclustermap` and `mem_chunk_size` are now
compile-time knobs** (`AT_CLUSTERMAP_FRAC`, default 0.2; `AT_MEM_CHUNK`,
default 2048). The deployment build uses 0.1 / 1024. Beyond that the curve is
flat — the remainder is cluster point storage, which is inherent.

**3. Host-build guards only.** `random()`/`srandom()` are wrapped in
`#ifndef HOST_BUILD` because the fork defines them unconditionally (newlib
lacks them, glibc has them). No effect on the ESP32 build.

## The ROI crop

Rather than decimate the whole frame, the deployment path crops first. At t=0
the arm is at its home pose, so the tag lands in a predictable region. Over
**n=2186** wrist detections (`Results/wrist16_ds`): cx 126.2..305.2,
cy 13.0..164.2, max tag side 25.5 px. Padded by 12 px that is
**x[98,320) y[0,193) = 222x193**, 56% of the frame.

(An n=39 sample put the region at 37-43%. It was too small — the same
small-sample optimism recorded in `Results/tag16h5_deployment.txt`. The
n=2186 bound is the one used.)

| configuration | peak heap | + frame | total | id0 detected |
|---|---|---|---|---|
| full frame, decimate 2 | 441 KB | 75 KB | 516 KB | 43/60 |
| ROI 222x193, decimate 2 | 298 KB | 42 KB | **340 KB** | 43/60 |

Both after the unionfind change. Detection is identical; the crop buys 176 KB.

## Status of the memory budget — READ THIS

**340 KB is a host (x86-64) measurement and is an UPPER BOUND, not the ESP32
figure.** Several of the remaining blocks are pointer-bearing — the
`uint64_zarray_entry` pools (24 B/entry here vs 16 on a 32-bit target), the
`clustermap` pointer table, and every `zarray_t` header (24 vs 16) — so the
ESP32 peak is lower. It is *not* estimated here, because there is no xtensa
toolchain in this workspace and an extrapolation would be a guess.

The unionfind arrays (75 KB at decimate 2, the largest remaining item) are
`uint16` and identical on both.

**The device settles this.** The sketch prints free heap before and after the
first detection; flash it and read the boot log. If it does not fit, the next
lever is a tighter ROI (the bound above is padded by 12 px) or decimate 2 on
an already-cropped 160x120 capture.

## Equivalence to the OpenCV detector used for all prior results

`at32.py` drives this exact C code from Python via `libat32.so`, so the
desktop experiments and the firmware run the same detector over the same crop
at the same decimation.

- 60 recorded frames: OpenCV full-res 39/60, this port at ROI+decimate2
  **43/60** — a superset, all 39 plus 4 more.
- Corner agreement on the 39 shared: **median 0.95 px**, max 1.31 px.
- Corner order differs from OpenCV aruco by the involution `[1,0,3,2]`,
  consistent on 39/39; `at32.py` reorders so `solvePnPGeneric`, the upright
  prior and the calibration are untouched.
- Live sim, 150 resets: OpenCV 98/150 at 20.53 mm raw error, this port
  102/150 at **20.46 mm**.

Because corners differ by ~1 px, the pose differs, and the residual corrector
is a per-setup calibration **fitted to whichever detector produced it**. The
corrector shipped for this path is refit on this detector's output
(`Results/wrist32_ds`) — a corrector fitted on OpenCV corners is not valid
here.

## Building

Host (for the Python bridge):
```
gcc -O2 -fPIC -shared -I esp32_apriltag -DHOST_BUILD \
    -DAT_CLUSTERMAP_FRAC=0.1 -DAT_MEM_CHUNK=1024 \
    -o esp32_apriltag/libat32.so esp32_apriltag/at32_shim.c \
    esp32_apriltag/*.c esp32_apriltag/common/*.c -lm -lpthread
```

Arduino: copy this directory to `src/esp32_apriltag/` inside the sketch
folder — the IDE compiles `src/` recursively, but not other subdirectories.
Build with `-DAT_CLUSTERMAP_FRAC=0.1 -DAT_MEM_CHUNK=1024` and WITHOUT
`HOST_BUILD`.
