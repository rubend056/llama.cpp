#include "models.h"

void llama_model_glm_dsa_mtp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);

    // DSA parameters (preserved on the MTP block but not used by the simplified MTP forward)
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head,   false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k,    false);

    // Expert gating function (GLM-DSA uses sigmoid)
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
    GGML_ASSERT(hparams.nextn_predict_layers > 0  && "GLM_DSA_MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers <= hparams.n_layer);

    // only the MTP layers get a KV cache; trunk layers are owned by the sibling GLM_DSA model.
    hparams.kv_only_nextn         = true;
    hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_glm_dsa_mtp::load_arch_tensors(llama_model_loader & /*ml*/) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("GLM_DSA_MTP architecture requires MLA");
    }

    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    // Required: shared with the trunk via the same .gguf, so these tensors are always present.
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd },          TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output  = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    const uint32_t n_main = n_layer - hparams.nextn_predict_layers;
    for (int i = 0; i < n_layer; ++i) {
        if (static_cast<uint32_t>(i) < n_main) {
            continue;  // trunk layer — owned by the sibling GLM_DSA model
        }

        auto & layer = layers[i];

        // GLM-DSA decoder block (MLA attention + MoE FFN + DSA indexer tensors).
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_embd },        0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), { q_lora_rank },   0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), { kv_lora_rank },  0);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), { n_embd, q_lora_rank }, 0);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), { q_lora_rank, n_head * n_embd_head_k_mla }, 0);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), { n_embd, kv_lora_rank + n_embd_head_qk_rope }, 0);
        layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,      "weight", i), { n_embd_head_qk_nope, kv_lora_rank, n_head }, 0);
        layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,      "weight", i), { kv_lora_rank, n_embd_head_v_mla, n_head }, 0);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_head * n_embd_head_v_mla, n_embd }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        // DSA indexer tensors (loaded but not used by the MTP forward — kept so the file fully unloads)
        if (hparams.indexer_n_head > 0 && hparams.indexer_head_size > 0) {
            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), { hparams.indexer_head_size }, TENSOR_SKIP | TENSOR_NOT_REQUIRED);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), { hparams.indexer_head_size }, TENSOR_SKIP | TENSOR_NOT_REQUIRED);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), { n_embd, hparams.indexer_n_head }, TENSOR_SKIP | TENSOR_NOT_REQUIRED);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), { n_embd, hparams.indexer_head_size }, TENSOR_SKIP | TENSOR_NOT_REQUIRED);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), { q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size }, TENSOR_SKIP | TENSOR_NOT_REQUIRED);
        }

        // MoE FFN
        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), { n_embd, n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), { n_expert },         TENSOR_NOT_REQUIRED);

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0");
        }

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd,   n_expert }, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);

        // Shared expert
        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_exp * n_expert_shared, n_embd }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), { 2 * n_embd, n_embd }, 0);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), { n_embd },              0);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), { n_embd },              0);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab },     TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd },              TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_glm_dsa_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// LLM_ARCH_GLM_DSA_MTP draft head for GLM 5.x DSA series.
//
// Forward (single MTP block):
//   h_in       = previous-step hidden (input embedding slot, "mtp_h_input")
//   tok_embd   = nextn.embed_tokens (if present) else model.tok_embd applied to drafted tokens
//   concat     = [ RMSNorm(tok_embd, enorm) ; RMSNorm(h_in, hnorm) ]   (over n_embd dim)
//   x          = eh_proj(concat)           // {2*n_embd -> n_embd}
//   x          = GLM-DSA decoder block @ layer il (MLA attention + MoE FFN, residuals on x)
//   x          = RMSNorm(x, shared_head_norm | output_norm)
//   logits     = (shared_head_head | output) @ x
llama_model_glm_dsa_mtp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_ASSERT(hparams.nextn_predict_layers > 0  && "GLM_DSA_MTP requires nextn_predict_layers > 0");
    GGML_ASSERT(hparams.nextn_predict_layers == 1 && "GLM_DSA_MTP currently only supports a single MTP block");

    GGML_ASSERT(hparams.is_mla() && "GLM_DSA_MTP architecture requires MLA");

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // pre-scale kq_scale / attn_factor for YaRN — mirrors deepseek2 graph
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
    const float mscale          = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale        = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    // The MTP block lives at the source file's original layer index.
    const int il = (int) hparams.n_layer - (int) hparams.nextn_predict_layers;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    // input: previous-step hidden state (from trunk) + drafted token ids
    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos    = build_inp_pos();
    auto * inp_attn_k        = build_attn_inp_k();

    // NextN preprocessing: RMSNorm(tok_embd) and RMSNorm(h) concatenated then projected.
    ggml_tensor * h_norm = build_norm(h_input, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(cur, "mtp_eh_proj", il);

    // ----- GLM-DSA decoder block (MLA + MoE), residuals on `cur` -----
    ggml_tensor * inpSA = cur;

    // self-attention: pre-norm
    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // MLA Q (with q-LoRA)
    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    cb(q, "mtp_q_a", il);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "mtp_q_a_norm", il);
    q = ggml_mul_mat(ctx0, layer.wq_b, q);
    cb(q, "mtp_q_b", il);

    ggml_tensor * q_nope = ggml_view_3d(
        ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
        ggml_row_size(q->type, n_embd_head_k),
        ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
    cb(q_nope, "mtp_q_nope", il);

    ggml_tensor * q_pe = ggml_view_3d(
        ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
        ggml_row_size(q->type, n_embd_head_k),
        ggml_row_size(q->type, n_embd_head_k) * n_head,
        ggml_row_size(q->type, n_embd_head_qk_nope));
    cb(q_pe, "mtp_q_pe", il);

    // KV down-projection (compressed kv + rope k)
    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

    ggml_tensor * kv_cmpr = ggml_view_2d(
        ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
    cb(kv_cmpr, "mtp_kv_cmpr", il);

    ggml_tensor * k_pe = ggml_view_3d(
        ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
    cb(k_pe, "mtp_k_pe", il);

    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                         freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q_pe, "mtp_q_pe_rope", il);
    k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                         freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_pe, "mtp_k_pe_rope", il);

    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "mtp_kv_cmpr_norm", il);

    // MLA absorption path (MQA-shaped attention)
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    cb(q_nope, "mtp_q_nope_perm", il);

    ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);
    q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
    cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

    ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
    cb(Qcur, "mtp_Qcur", il);

    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

    ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
    cb(Kcur, "mtp_Kcur", il);

    ggml_tensor * Vcur = kv_cmpr;
    cb(Vcur, "mtp_Vcur", il);

    cur = build_attn(inp_attn_k,
                     layer.wo, nullptr, layer.wo_s,
                     Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    // ----- MoE FFN with shared expert (sigmoid-gated softmax routing, GLM-DSA style) -----
    ggml_tensor * ffn_residual = cur;

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
        layer.ffn_gate_inp,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        n_expert, n_expert_used,
        LLM_FFN_SILU, hparams.expert_weights_norm,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il,
        nullptr,
        layer.ffn_gate_up_exps);
    cb(moe_out, "mtp_ffn_moe_out", il);

    {
        ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "mtp_ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
    }
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    // snapshot post-FFN hidden so the AR MTP loop can re-use it
    res->t_mtp_out = cur;

    // ----- final norm + LM head -----
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "GLM_DSA_MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "GLM_DSA_MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
