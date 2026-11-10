// Runs a list of chat prompts through one loaded model (greedy, -n tokens each) for profiling.
// Prompts come from a JSONL file (one {"prompt": "..."} per line), set with LLAMASCOPE_PROMPTS.
// LLAMASCOPE_RANGE=start:end selects lines. All other options are the usual llama.cpp ones (-m, -ngl, -n, ...).

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * file = getenv("LLAMASCOPE_PROMPTS");
    if (!file) {
        fprintf(stderr, "set LLAMASCOPE_PROMPTS to a JSONL file\n");
        return 1;
    }
    int first = 0, last = 1 << 30;
    if (const char * range = getenv("LLAMASCOPE_RANGE")) {
        sscanf(range, "%d:%d", &first, &last);
    }

    std::vector<std::string> prompts;
    {
        std::ifstream in(file);
        std::string line;
        for (int i = 0; std::getline(in, line); i++) {
            if (i < first || i >= last || line.empty()) {
                continue;
            }
            prompts.push_back(nlohmann::json::parse(line).value("prompt", ""));
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    params.warmup = false;
    {
        auto init = common_init_from_params(params);
        llama_model   * model = init->model();
        llama_context * ctx   = init->context();
        if (!model || !ctx) {
            fprintf(stderr, "failed to load the model\n");
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        auto tmpls = common_chat_templates_init(model, params.chat_template);
        llama_sampler * smpl = llama_sampler_init_greedy();
        const int n_ctx   = (int) llama_n_ctx(ctx);
        const int n_batch = (int) llama_n_batch(ctx);
        const int n_gen   = params.n_predict > 0 ? params.n_predict : 128;

        long long n_prompt_tok = 0, n_gen_tok = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t p = 0; p < prompts.size(); p++) {
            common_chat_templates_inputs inputs;
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = prompts[p];
            inputs.messages.push_back(msg);
            inputs.add_generation_prompt = true;
            inputs.enable_thinking       = false;
            const std::string text = common_chat_templates_apply(tmpls.get(), inputs).prompt.str();

            std::vector<llama_token> toks = common_tokenize(vocab, text, true, true);
            if ((int) toks.size() + n_gen > n_ctx) {
                toks.resize(std::max(0, n_ctx - n_gen - 1));
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_sampler_reset(smpl);

            bool ok = true;
            for (int i = 0; i < (int) toks.size() && ok; i += n_batch) {
                const int n = std::min(n_batch, (int) toks.size() - i);
                ok = llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) == 0;
            }
            n_prompt_tok += toks.size();
            for (int g = 0; g < n_gen && ok; g++) {
                llama_token t = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, t)) {
                    break;
                }
                ok = llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0;
                n_gen_tok++;
            }
            if (!ok) {
                fprintf(stderr, "decode failed on prompt %zu\n", p + first);
            }
            if ((p + 1) % 10 == 0 || p + 1 == prompts.size()) {
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                fprintf(stderr, "llamascope-batch: %zu/%zu prompts, %lld prompt + %lld generated tokens, %.0f s\n", p + 1,
                        prompts.size(), n_prompt_tok, n_gen_tok, s);
            }
        }
        llama_sampler_free(smpl);
    }
    llama_backend_free();
    return 0;
}
