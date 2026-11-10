#include "profiler.cuh"
#include "../ggml-profiler.h"

#include <vector>

namespace {

constexpr uint32_t kMarkerCap = 1u << 20;
constexpr size_t   kCounts    = (size_t) GGML_PROF_MAX_LAYERS * GGML_PROF_MAX_EXPERTS;

struct prof_state {
    ggml_prof_gpu_marker * markers = nullptr;
    unsigned int *         counter = nullptr;
    uint32_t *             counts  = nullptr;   // [2][layers][experts]: generation, then prompt
    unsigned long long *   now     = nullptr;
    ggml_prof_trace_entry * trace  = nullptr;   // GGML_PROFILE_TRACE=1: experts per generated token and layer
    unsigned int *         trace_n = nullptr;
    unsigned int           trace_cap = 0;
};

__device__ __forceinline__ unsigned long long gpu_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void k_marker(ggml_prof_gpu_marker * m, unsigned int * counter, uint32_t name) {
    const unsigned int i = atomicAdd(counter, 1u);
    if (i < kMarkerCap) {
        m[i].ns   = gpu_ns();
        m[i].name = name;
        m[i].seq  = i;
    }
}

__global__ void k_now(unsigned long long * out) {
    *out = gpu_ns();
}

__global__ void k_count(const char * ids, int64_t nb0, int64_t nb1, int n_used, int n_tokens, uint32_t * row,
                        ggml_prof_trace_entry * trace, unsigned int * trace_n, unsigned int trace_cap, int layer) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_used * n_tokens) {
        return;
    }
    if (trace && n_tokens == 1 && i == 0) {
        const unsigned int k = atomicAdd(trace_n, 1u);
        if (k < trace_cap) {
            trace[k].layer = (uint16_t) layer;
            trace[k].n     = (uint16_t) min(n_used, 16);
            for (int j = 0; j < n_used && j < 16; ++j) {
                trace[k].ids[j] = (uint16_t) *(const int32_t *) (ids + j * nb0);
            }
        }
    }
    const int32_t e = *(const int32_t *) (ids + (i % n_used) * nb0 + (i / n_used) * nb1);
    if (e >= 0 && e < GGML_PROF_MAX_EXPERTS) {
        atomicAdd(&row[e], 1u);
    }
}

prof_state * state(ggml_backend_cuda_context & ctx) {
    return (prof_state *) ctx.prof;
}

} // namespace

void ggml_cuda_prof_prepare(ggml_backend_cuda_context & ctx) {
    int level = 0, n = 0;
    if (ctx.prof || !ggml_prof_slots(&level, &n)) {
        return;
    }
    const char * gpu = getenv("GGML_PROFILE_GPU");
    if (gpu && strcmp(gpu, "0") == 0) {
        return;
    }
    prof_state * s = new prof_state;
    CUDA_CHECK(cudaMalloc(&s->markers, kMarkerCap * sizeof(ggml_prof_gpu_marker)));
    CUDA_CHECK(cudaMalloc(&s->counter, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&s->counts, 2 * kCounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s->now, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(s->counter, 0, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(s->counts, 0, 2 * kCounts * sizeof(uint32_t)));
    const char * tr = getenv("GGML_PROFILE_TRACE");
    if (tr && atoi(tr) > 0) {
        const char * mb = getenv("GGML_PROFILE_TRACE_MB");
        s->trace_cap = (unsigned int) ((size_t) (mb ? atoi(mb) : 256) * 1024 * 1024 / sizeof(ggml_prof_trace_entry));
        CUDA_CHECK(cudaMalloc(&s->trace, (size_t) s->trace_cap * sizeof(ggml_prof_trace_entry)));
        CUDA_CHECK(cudaMalloc(&s->trace_n, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(s->trace_n, 0, sizeof(unsigned int)));
    }
    ctx.prof = s;
}

void ggml_cuda_prof_begin(ggml_backend_cuda_context & ctx) {
    prof_state * s = state(ctx);
    if (s) {
        k_marker<<<1, 1, 0, ctx.stream()>>>(s->markers, s->counter, GGML_PROF_GPU_BEGIN);
    }
}

void ggml_cuda_prof_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    prof_state * s = state(ctx);
    if (!s) {
        return;
    }
    uint32_t prefix;
    const uint32_t name = ggml_prof_hash(node->name, &prefix);
    if (!ggml_prof_is_block_end(prefix)) {
        return;
    }
    ggml_prof_name(name, node->name);
    k_marker<<<1, 1, 0, ctx.stream()>>>(s->markers, s->counter, name);

    const int layer = ggml_prof_layer(node->name);
    if (ggml_prof_is_topk(prefix) && layer >= 0 && node->type == GGML_TYPE_I32) {
        const int n_used = (int) node->ne[0], n_tokens = (int) node->ne[1];
        uint32_t * row = s->counts + (n_tokens == 1 ? 0 : kCounts) + (size_t) layer * GGML_PROF_MAX_EXPERTS;
        const int n = n_used * n_tokens;
        k_count<<<(n + 255) / 256, 256, 0, ctx.stream()>>>((const char *) node->data, node->nb[0], node->nb[1], n_used, n_tokens, row,
                                                            s->trace, s->trace_n, s->trace_cap, layer);
    }
}

void ggml_cuda_prof_flush(ggml_backend_cuda_context & ctx) {
    prof_state * s = state(ctx);
    if (!s) {
        return;
    }
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaDeviceSynchronize());

    // clock pair: GPU time read by a kernel, CPU time at the middle of the shortest round trip
    uint64_t best_rtt = UINT64_MAX, ref_tsc = 0;
    unsigned long long ref_ns = 0;
    for (int i = 0; i < 16; ++i) {
        const uint64_t a = ggml_prof_tick();
        k_now<<<1, 1, 0, ctx.stream()>>>(s->now);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        const uint64_t b = ggml_prof_tick();
        if (b - a < best_rtt) {
            best_rtt = b - a;
            ref_tsc  = a + (b - a) / 2;
            CUDA_CHECK(cudaMemcpy(&ref_ns, s->now, sizeof(ref_ns), cudaMemcpyDeviceToHost));
        }
    }

    unsigned int n = 0;
    CUDA_CHECK(cudaMemcpy(&n, s->counter, sizeof(n), cudaMemcpyDeviceToHost));
    n = std::min(n, kMarkerCap);
    std::vector<ggml_prof_gpu_marker> m(n);
    if (n) {
        CUDA_CHECK(cudaMemcpy(m.data(), s->markers, n * sizeof(ggml_prof_gpu_marker), cudaMemcpyDeviceToHost));
    }
    std::vector<uint32_t> counts(2 * kCounts);
    CUDA_CHECK(cudaMemcpy(counts.data(), s->counts, counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    ggml_prof_add_gpu_markers(m.data(), n, ref_tsc, ref_ns);
    ggml_prof_add_expert_counts(counts.data(), counts.data() + kCounts);
    if (s->trace) {
        unsigned int nt = 0;
        CUDA_CHECK(cudaMemcpy(&nt, s->trace_n, sizeof(nt), cudaMemcpyDeviceToHost));
        nt = std::min(nt, s->trace_cap);
        std::vector<ggml_prof_trace_entry> t(nt);
        if (nt) {
            CUDA_CHECK(cudaMemcpy(t.data(), s->trace, (size_t) nt * sizeof(ggml_prof_trace_entry), cudaMemcpyDeviceToHost));
        }
        ggml_prof_add_trace(t.data(), nt);
        CUDA_CHECK(cudaFree(s->trace));
        CUDA_CHECK(cudaFree(s->trace_n));
    }

    CUDA_CHECK(cudaFree(s->markers));
    CUDA_CHECK(cudaFree(s->counter));
    CUDA_CHECK(cudaFree(s->counts));
    CUDA_CHECK(cudaFree(s->now));
    delete s;
    ctx.prof = nullptr;
}
