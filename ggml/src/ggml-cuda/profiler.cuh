#pragma once

#include "common.cuh"

// GPU part of the ggml profiler (GGML_PROFILE): marker kernels at block ends and expert counts at topk nodes.
// All launches go to the main stream, so they work inside captured CUDA graphs.
void ggml_cuda_prof_prepare(ggml_backend_cuda_context & ctx);
void ggml_cuda_prof_begin(ggml_backend_cuda_context & ctx);
void ggml_cuda_prof_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
void ggml_cuda_prof_flush(ggml_backend_cuda_context & ctx);
