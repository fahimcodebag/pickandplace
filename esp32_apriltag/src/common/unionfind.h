/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
   BSD 2-Clause. See LICENSE.md in this directory.

   MODIFIED for memory-constrained targets (plain ESP32, no PSRAM).

   Upstream stores {uint32 parent, uint32 size} per pixel = 8 bytes. At
   320x240 that is 600 KB, which alone exceeds the ~320 KB of internal SRAM
   free on a plain ESP32 after the FSM, the two TFLite arenas and the
   corrector. This version stores parent and size in SEPARATE uint16 arrays
   whenever maxid fits in 16 bits (<= 65534), giving 4 bytes per pixel, and
   falls back to the upstream 32-bit layout otherwise. Separate arrays rather
   than a 6-byte struct because the struct would pad back up to 8.

   `size` saturates at 65535 in the 16-bit path. That is safe here: size is
   only read to (a) pick the larger tree when joining and (b) filter clusters
   by size, and every cluster of interest is orders of magnitude smaller than
   65535 pixels. Saturation only makes a huge blob look slightly smaller than
   another huge blob, and both are rejected. Sizes start at 1, as upstream.
*/
#pragma once

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

typedef struct unionfind unionfind_t;

struct unionfind
{
    uint32_t maxid;
    int      narrow;      // 1 => 16-bit arrays (4 B/px), 0 => 32-bit (8 B/px)
    uint16_t *p16, *s16;
    uint32_t *p32, *s32;
};

static inline unionfind_t *unionfind_create(uint32_t maxid)
{
    unionfind_t *uf = (unionfind_t*) calloc(1, sizeof(unionfind_t));
    uf->maxid = maxid;
    uf->narrow = (maxid <= 65534u);
    // Every node starts as its own root with size 1 -- matching upstream.
    // (An earlier draft imported a 0xffffffff lazy-init sentinel from a
    // different apriltag generation and left size at 0; that silently made
    // every cluster measure size 0, so all were filtered and NOTHING was
    // detected. Keep parent=self, size=1.)
    if (uf->narrow) {
        uf->p16 = (uint16_t*) malloc((maxid+1) * sizeof(uint16_t));
        uf->s16 = (uint16_t*) malloc((maxid+1) * sizeof(uint16_t));
        for (uint32_t i = 0; i <= maxid; i++) { uf->p16[i] = (uint16_t) i; uf->s16[i] = 1; }
    } else {
        uf->p32 = (uint32_t*) malloc((maxid+1) * sizeof(uint32_t));
        uf->s32 = (uint32_t*) malloc((maxid+1) * sizeof(uint32_t));
        for (uint32_t i = 0; i <= maxid; i++) { uf->p32[i] = i; uf->s32[i] = 1; }
    }
    return uf;
}

static inline void unionfind_destroy(unionfind_t *uf)
{
    free(uf->p16); free(uf->s16); free(uf->p32); free(uf->s32);
    free(uf);
}

static inline uint32_t unionfind_get_representative(unionfind_t *uf, uint32_t id)
{
    if (uf->narrow) {
        uint32_t root = id;
        while (uf->p16[root] != root) root = uf->p16[root];
        // path compression
        while (uf->p16[id] != root) { uint32_t n = uf->p16[id];
                                      uf->p16[id] = (uint16_t) root; id = n; }
        return root;
    } else {
        uint32_t root = id;
        while (uf->p32[root] != root) root = uf->p32[root];
        while (uf->p32[id] != root) { uint32_t n = uf->p32[id];
                                      uf->p32[id] = root; id = n; }
        return root;
    }
}

static inline uint32_t unionfind_get_set_size(unionfind_t *uf, uint32_t id)
{
    uint32_t r = unionfind_get_representative(uf, id);
    return uf->narrow ? (uint32_t) uf->s16[r] : uf->s32[r];
}

static inline uint32_t unionfind_connect(unionfind_t *uf, uint32_t aid, uint32_t bid)
{
    uint32_t aroot = unionfind_get_representative(uf, aid);
    uint32_t broot = unionfind_get_representative(uf, bid);
    if (aroot == broot) return aroot;

    if (uf->narrow) {
        uint32_t asize = uf->s16[aroot], bsize = uf->s16[broot];
        uint32_t sum;
        if (asize > bsize) {
            uf->p16[broot] = (uint16_t) aroot;
            sum = asize + bsize; uf->s16[aroot] = (uint16_t)(sum > 65535u ? 65535u : sum);
            return aroot;
        } else {
            uf->p16[aroot] = (uint16_t) broot;
            sum = asize + bsize; uf->s16[broot] = (uint16_t)(sum > 65535u ? 65535u : sum);
            return broot;
        }
    } else {
        uint32_t asize = uf->s32[aroot], bsize = uf->s32[broot];
        if (asize > bsize) { uf->p32[broot] = aroot; uf->s32[aroot] += bsize; return aroot; }
        else               { uf->p32[aroot] = broot; uf->s32[broot] += asize; return broot; }
    }
}
