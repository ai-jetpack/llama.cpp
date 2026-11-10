#pragma once
// In-process profiler (see LlamaScope docs/PROFILER.md). Off unless GGML_PROFILE=op|block.
// Hot path: plain stores into a per-thread, cache-line-aligned slot; no locks, no kernel calls, no allocation.
#include "ggml.h"
#include <stdint.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_prof_kind {
    GGML_PROF_OP       = 1,   // one node: t0 start, t1 compute end, t2 barrier end
    GGML_PROF_BLOCK    = 2,   // folded nodes: t0 first start, t1 = sum of compute ticks, t2 = sum of wait ticks
    GGML_PROF_SCHED    = 3,   // scheduler event: t0 start, t1 end, name = event id, count = split index
    GGML_PROF_COPY     = 4,   // one split input copy: t0 start, t1 sync done, t2 end, name = tensor, count = split, thread = path
};

// copy paths (GGML_PROF_COPY entries, stored in the thread field)
enum ggml_prof_copy_path {
    GGML_PROF_COPY_ASYNC    = 0,   // backend async copy, no host wait
    GGML_PROF_COPY_SYNC     = 1,   // fallback: synchronize both backends, then blocking copy
    GGML_PROF_COPY_USER     = 2,   // user input: synchronize split backend, then blocking copy
    GGML_PROF_COPY_CALLBACK = 3,   // host weights copied by the user callback
};

enum ggml_prof_sched_event {
    GGML_PROF_EV_COPY_IN  = 1,   // input copies of a split (the CPU<->GPU transfers)
    GGML_PROF_EV_SUBMIT   = 2,   // graph_compute_async of a split
    GGML_PROF_EV_SYNC     = 3,   // ggml_backend_synchronize (CPU waiting for a device)
    GGML_PROF_EV_GRAPH    = 4,   // start of a graph compute, t2 = number of tokens
};

// name of the GPU marker written at the start of each GPU split
#define GGML_PROF_GPU_BEGIN 1u

// GPU marker written by a device kernel: GPU clock (ns) at a block end.
typedef struct ggml_prof_gpu_marker {
    uint64_t ns;
    uint32_t name;
    uint32_t seq;
} ggml_prof_gpu_marker;

// expert trace entry: experts picked by one generated token in one layer (GGML_PROFILE_TRACE=1)
typedef struct ggml_prof_trace_entry {
    uint16_t layer;
    uint16_t n;
    uint16_t ids[16];
} ggml_prof_trace_entry;

#define GGML_PROF_MAX_LAYERS  256
#define GGML_PROF_MAX_EXPERTS 1024

typedef struct ggml_prof_entry {
    uint64_t t0, t1, t2;
    uint32_t name;      // hash of the tensor name, or sched event id
    uint8_t  thread;    // ggml thread index (ith)
    uint8_t  kind;      // enum ggml_prof_kind
    uint16_t count;     // nodes folded (block) or split index (sched)
} ggml_prof_entry;

typedef struct ggml_prof_slot {
    ggml_prof_entry * buf;
    uint32_t n, cap;
    uint64_t dropped;
    char pad[64 - sizeof(void *) - 2 * sizeof(uint32_t) - sizeof(uint64_t)];
} ggml_prof_slot;

// Level: 0 off, 1 block, 2 op. Returns the slot array (one per thread index) or NULL when off.
GGML_API ggml_prof_slot * ggml_prof_slots(int * level, int * n_slots);
// Name dictionary: called by thread 0 only; cheap after the first sight of a name.
GGML_API void ggml_prof_name(uint32_t hash, const char * name);
// Same, and remember the tensor size (used for copied inputs).
GGML_API void ggml_prof_name_bytes(uint32_t hash, const char * name, uint64_t bytes);
// Is this (prefix) hash a block-end marker? (attn_residual, ffn_moe_topk, ffn_moe_out, ffn_shexp, l_out)
GGML_API int  ggml_prof_is_block_end(uint32_t prefix_hash);
GGML_API int  ggml_prof_is_topk(uint32_t prefix_hash);
// GPU markers; (ref_tsc, ref_ns) is a pair of simultaneous CPU and GPU clock readings for conversion.
GGML_API void ggml_prof_add_gpu_markers(const ggml_prof_gpu_marker * m, uint32_t n, uint64_t ref_tsc, uint64_t ref_ns);
// Add expert selection counts: arrays of [GGML_PROF_MAX_LAYERS][GGML_PROF_MAX_EXPERTS].
GGML_API void ggml_prof_add_expert_counts(const uint32_t * gen, const uint32_t * prompt);
// Count expert ids of a topk tensor in host memory (int32, ne0 = experts used, ne1 = tokens).
GGML_API void ggml_prof_count_experts(const struct ggml_tensor * topk);
GGML_API int  ggml_prof_layer(const char * name);
GGML_API void ggml_prof_add_trace(const ggml_prof_trace_entry * t, uint32_t n);

static inline uint64_t ggml_prof_tick(void) {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#else
    return (uint64_t) ggml_time_us() * 1000;
#endif
}

// FNV-1a of the whole name and of its prefix before the first '-' (the layer suffix).
static inline uint32_t ggml_prof_hash(const char * s, uint32_t * prefix_hash) {
    uint32_t h = 2166136261u, p = 0;
    int in_prefix = 1;
    for (; *s; ++s) {
        if (in_prefix && *s == '-') { p = h; in_prefix = 0; }
        h = (h ^ (uint8_t) *s) * 16777619u;
    }
    *prefix_hash = in_prefix ? h : p;
    return h;
}

static inline void ggml_prof_push(ggml_prof_slot * s, uint64_t t0, uint64_t t1, uint64_t t2, uint32_t name,
                                  int thread, int kind, int count) {
    if (s->n < s->cap) {
        ggml_prof_entry * e = &s->buf[s->n++];
        e->t0 = t0; e->t1 = t1; e->t2 = t2; e->name = name;
        e->thread = (uint8_t) thread; e->kind = (uint8_t) kind; e->count = (uint16_t) count;
    } else {
        s->dropped++;
    }
}

#ifdef __cplusplus
}
#endif
