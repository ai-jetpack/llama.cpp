#include "ggml-profiler.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int    kMaxSlots = 64;
constexpr size_t kDictSize = 1 << 14;   // power of two

struct dict_entry {
    uint32_t hash;
    uint64_t bytes;
    char     name[GGML_MAX_NAME];
};

struct profiler {
    int                       level = 0;
    ggml_prof_slot            slots[kMaxSlots] = {};
    std::vector<dict_entry>   dict;
    uint32_t                  block_end[5] = {};
    std::vector<ggml_prof_gpu_marker> gpu;
    std::vector<std::pair<size_t, std::pair<uint64_t, uint64_t>>> gpu_refs;   // (first index, (ref_tsc, ref_ns))
    std::vector<uint32_t>     gen, prompt;
    std::vector<ggml_prof_trace_entry> trace;
    uint64_t                  tsc0 = 0;
    std::chrono::steady_clock::time_point wall0;
    std::string               path;

    profiler() {
        const char * mode = getenv("GGML_PROFILE");
        if (!mode) {
            return;
        }
        level = strcmp(mode, "op") == 0 ? 2 : strcmp(mode, "block") == 0 ? 1 : 0;
        if (!level) {
            return;
        }
        const char * file = getenv("GGML_PROFILE_FILE");
        path = file ? file : "ggml-profile.bin";
        const char * mb = getenv("GGML_PROFILE_MB");
        const size_t cap = (size_t) (mb ? atoi(mb) : 32) * 1024 * 1024 / sizeof(ggml_prof_entry);
        // one slot per possible thread index; ggml uses at most one thread per logical CPU by default
        const char * nt = getenv("GGML_PROFILE_THREADS");
        const int n_alloc = std::min(kMaxSlots, nt ? atoi(nt) : (int) std::thread::hardware_concurrency() + 1);
        for (int i = 0; i < n_alloc; ++i) {
            ggml_prof_slot & s = slots[i];
            s.buf = (ggml_prof_entry *) malloc(cap * sizeof(ggml_prof_entry));
            if (s.buf) {
                // touch all pages now so the hot path never page-faults
                memset(s.buf, 0, cap * sizeof(ggml_prof_entry));
            }
            s.cap = s.buf ? (uint32_t) cap : 0;
        }
        dict.resize(kDictSize);
        gen.assign((size_t) GGML_PROF_MAX_LAYERS * GGML_PROF_MAX_EXPERTS, 0);
        prompt.assign(gen.size(), 0);
        const char * ends[5] = { "attn_residual", "ffn_moe_topk", "ffn_moe_out", "ffn_shexp", "l_out" };
        for (int i = 0; i < 5; ++i) {
            ggml_prof_hash(ends[i], &block_end[i]);
        }
        wall0 = std::chrono::steady_clock::now();
        tsc0  = ggml_prof_tick();
    }

    ~profiler() {
        if (!level) {
            return;
        }
        const uint64_t tsc1 = ggml_prof_tick();
        const double   secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        const uint64_t hz   = secs > 0 ? (uint64_t) ((tsc1 - tsc0) / secs) : 0;
        write(hz);
        for (auto & s : slots) {
            free(s.buf);
        }
    }

    void write(uint64_t hz) {
        FILE * f = fopen(path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "ggml-profile: cannot write %s\n", path.c_str());
            return;
        }
        uint32_t n_slots = 0, n_names = 0;
        uint64_t dropped = 0, total = 0;
        for (int i = 0; i < kMaxSlots; ++i) {
            if (slots[i].n) {
                n_slots = i + 1;
            }
            dropped += slots[i].dropped;
            total += slots[i].n;
        }
        for (const auto & d : dict) {
            n_names += d.hash != 0;
        }
        const uint32_t version = 3, lvl = (uint32_t) level;
        fwrite("GGMLPRF3", 1, 8, f);
        fwrite(&version, 4, 1, f);
        fwrite(&lvl, 4, 1, f);
        fwrite(&hz, 8, 1, f);
        fwrite(&tsc0, 8, 1, f);
        fwrite(&dropped, 8, 1, f);
        fwrite(&n_slots, 4, 1, f);
        for (uint32_t i = 0; i < n_slots; ++i) {
            fwrite(&slots[i].n, 4, 1, f);
            fwrite(slots[i].buf, sizeof(ggml_prof_entry), slots[i].n, f);
        }
        fwrite(&n_names, 4, 1, f);
        for (const auto & d : dict) {
            if (!d.hash) {
                continue;
            }
            const uint16_t len = (uint16_t) strnlen(d.name, GGML_MAX_NAME);
            fwrite(&d.hash, 4, 1, f);
            fwrite(&len, 2, 1, f);
            fwrite(d.name, 1, len, f);
            fwrite(&d.bytes, 8, 1, f);
        }

        // GPU markers, converted to TSC ticks with the reference pair of their batch
        const uint32_t n_gpu = (uint32_t) gpu.size();
        fwrite(&n_gpu, 4, 1, f);
        size_t r = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            while (r + 1 < gpu_refs.size() && gpu_refs[r + 1].first <= i) {
                r++;
            }
            const uint64_t ref_tsc = gpu_refs[r].second.first, ref_ns = gpu_refs[r].second.second;
            const double   dt_ns   = (double) (int64_t) (gpu[i].ns - ref_ns);
            const uint64_t tsc     = ref_tsc + (uint64_t) (int64_t) (dt_ns * (double) hz / 1e9);
            fwrite(&tsc, 8, 1, f);
            fwrite(&gpu[i].name, 4, 1, f);
            fwrite(&gpu[i].seq, 4, 1, f);
        }

        // expert counts, trimmed to the used layers and experts
        uint32_t n_layers = 0, n_experts = 0;
        for (uint32_t l = 0; l < GGML_PROF_MAX_LAYERS; ++l) {
            for (uint32_t e = 0; e < GGML_PROF_MAX_EXPERTS; ++e) {
                const size_t k = (size_t) l * GGML_PROF_MAX_EXPERTS + e;
                if (gen[k] || prompt[k]) {
                    n_layers  = std::max(n_layers, l + 1);
                    n_experts = std::max(n_experts, e + 1);
                }
            }
        }
        fwrite(&n_layers, 4, 1, f);
        fwrite(&n_experts, 4, 1, f);
        for (const auto * a : { &gen, &prompt }) {
            for (uint32_t l = 0; l < n_layers; ++l) {
                fwrite(a->data() + (size_t) l * GGML_PROF_MAX_EXPERTS, 4, n_experts, f);
            }
        }

        // expert trace (generation tokens, in execution order)
        const uint32_t n_trace = (uint32_t) trace.size();
        fwrite(&n_trace, 4, 1, f);
        fwrite(trace.data(), sizeof(ggml_prof_trace_entry), n_trace, f);
        fclose(f);
        fprintf(stderr, "ggml-profile: %llu entries (%llu dropped), %u names, %u gpu markers -> %s\n", (unsigned long long) total,
                (unsigned long long) dropped, n_names, n_gpu, path.c_str());
    }
};

std::mutex g_mutex;   // rare flush calls only, never the hot path

profiler & prof() {
    static profiler p;
    return p;
}

} // namespace

extern "C" {

ggml_prof_slot * ggml_prof_slots(int * level, int * n_slots) {
    profiler & p = prof();
    *level   = p.level;
    *n_slots = kMaxSlots;
    return p.level ? p.slots : nullptr;
}

void ggml_prof_name_bytes(uint32_t hash, const char * name, uint64_t bytes) {
    profiler & p = prof();
    if (!p.level || hash == 0) {
        return;
    }
    for (size_t i = hash & (kDictSize - 1), n = 0; n < kDictSize; i = (i + 1) & (kDictSize - 1), ++n) {
        dict_entry & d = p.dict[i];
        if (d.hash == hash) {
            if (bytes) {
                d.bytes = bytes;
            }
            return;
        }
        if (d.hash == 0) {
            d.hash  = hash;
            d.bytes = bytes;
            strncpy(d.name, name, GGML_MAX_NAME - 1);
            return;
        }
    }
}

void ggml_prof_name(uint32_t hash, const char * name) {
    ggml_prof_name_bytes(hash, name, 0);
}

int ggml_prof_is_topk(uint32_t prefix_hash) {
    return prof().block_end[1] == prefix_hash;
}

int ggml_prof_layer(const char * name) {
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return -1;
    }
    const int l = atoi(dash + 1);
    return l >= 0 && l < GGML_PROF_MAX_LAYERS ? l : -1;
}

void ggml_prof_add_gpu_markers(const ggml_prof_gpu_marker * m, uint32_t n, uint64_t ref_tsc, uint64_t ref_ns) {
    profiler & p = prof();
    if (!p.level || n == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    p.gpu_refs.push_back({ p.gpu.size(), { ref_tsc, ref_ns } });
    p.gpu.insert(p.gpu.end(), m, m + n);
}

void ggml_prof_add_expert_counts(const uint32_t * gen, const uint32_t * prompt) {
    profiler & p = prof();
    if (!p.level) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (size_t i = 0; i < p.gen.size(); ++i) {
        p.gen[i]    += gen[i];
        p.prompt[i] += prompt[i];
    }
}

void ggml_prof_add_trace(const ggml_prof_trace_entry * t, uint32_t n) {
    profiler & p = prof();
    if (!p.level || n == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    p.trace.insert(p.trace.end(), t, t + n);
}

void ggml_prof_count_experts(const struct ggml_tensor * topk) {
    profiler & p = prof();
    const int layer = ggml_prof_layer(topk->name);
    if (!p.level || layer < 0 || topk->type != GGML_TYPE_I32 || !topk->data) {
        return;
    }
    auto & counts = topk->ne[1] == 1 ? p.gen : p.prompt;
    uint32_t * row = counts.data() + (size_t) layer * GGML_PROF_MAX_EXPERTS;
    for (int64_t t = 0; t < topk->ne[1]; ++t) {
        for (int64_t i = 0; i < topk->ne[0]; ++i) {
            const int32_t e = *(const int32_t *) ((const char *) topk->data + i * topk->nb[0] + t * topk->nb[1]);
            if (e >= 0 && e < GGML_PROF_MAX_EXPERTS) {
                row[e]++;
            }
        }
    }
}

int ggml_prof_is_block_end(uint32_t prefix_hash) {
    const profiler & p = prof();
    for (uint32_t h : p.block_end) {
        if (h == prefix_hash) {
            return 1;
        }
    }
    return 0;
}

}
