#pragma once

#include "llama.h"

#include <deque>
#include <vector>

// MTP (Multi-Token Prediction) shadow-context glue.
//
// MTP runs a small auxiliary context that mirrors the trunk timeline.
// The MTP model takes (token_p, h_{p-1}) as input and produces its own
// KV cache entry at position p. The h_{p-1} handed to a k=0 draft step
// comes from the trunk model; all subsequent draft steps use the MTP's own
// output. Because of this, the trunk must "bridge" the MTP cache whenever
// positions become stale (e.g. after a checkpoint restore following a
// partial draft acceptance).
//
// The invariant we enforce:
//      After every trunk decode batch at positions [p0 .. p0+n-1],
//      the MTP KV cache covers positions [0 .. p0+n-1] with values that
//      were written by llama_decode on the MTP context — either during the
//      hook write described below or during the preceding draft() AR loop.
//
// How the hook (handle_mtp_for_ubatch) achieves this:
//   1. Every trunk ubatch contributes its hidden-state rows to a sliding
//      ring buffer (size = n_ubatch + 1).
//   2. If the MTP cache already has positions >= p0 (from drafting), those
//      tail entries are trimmed so the write can proceed contiguously.
//   3. We look up h_{p0-1} in the ring buffer.
//      - Found  → full write: p0 uses (h_{p0-1}, token[p0]),
//                               p0+1..p0+n-1 use (h[0..n-2], token[p0+1..p0+n-1]).
//      - Missing → partial write: p0+1..p0+n-1 only; position p0 is skipped.
//
// This avoids the fragile three-way state machine (AHEAD / IN-SYNC / DRIFT)
// and the single-float-vector pending_h stash that went stale across
// checkpoint-restore cycles.

//
// Per-hook-call timing breakdown for optimization profiling
//
struct mtp_hook_timing {
    int64_t t_total_us       = 0; // wall time of handle_mtp_for_ubatch
    int64_t t_copy_ring_us   = 0; // copying trunk h rows into the ring buffer
    int64_t t_reconcile_us   = 0; // reconciling MTP cache (pos_max checks)
    int64_t t_lookup_us      = 0; // searching ring buffer for h_{need_pos}
    int64_t t_build_batch_us = 0; // building the hook batch from ring + tensor
    int64_t t_decode_us      = 0; // llama_decode(ctx_mtp, hook_batch)
    int64_t t_sync_us        = 0; // trunk synchronize() before copying

    int32_t n_calls       = 0; // number of hook invocations
    int32_t n_rows_copied = 0; // total rows copied into ring
    int32_t n_rows_decoded = 0; // total rows decoded in hook batches
    int32_t n_ring_hits   = 0; // times h_{need_pos} found in ring
    int32_t n_ring_misses = 0; // times h_{need_pos} NOT found in ring
    int32_t n_cache_wipes = 0; // times MTP cache was wiped (gap/drift)

    void reset() { *this = {}; }
};

//
// Trunk decode timing — breaks down where time goes in the main model's
// forward passes: compute (graph dispatch) vs overhead (memory mgmt,
// logit/embd extraction, hook overhead). Split by prompt processing (PP)
// and generation (verification + single-token decodes).
//
struct trunk_dec_timing {
    // PP — large batches (prompt ingestion)
    int64_t t_pp_total_us    = 0; // wall time in decode() calls
    int64_t t_pp_compute_us  = 0; // time in process_ubatch (graph compute)
    int32_t n_pp_calls       = 0; // number of PP decode() calls
    int32_t n_pp_tokens      = 0; // total tokens processed
    int32_t n_pp_ubatches    = 0; // total ubatches dispatched
    int32_t n_pp_tokens_ub   = 0; // total tokens across ubatches

    // GEN — small batches (generation, verification, single-token decode)
    int64_t t_gen_total_us    = 0;
    int64_t t_gen_compute_us  = 0;
    int32_t n_gen_calls       = 0;
    int32_t n_gen_tokens      = 0;
    int32_t n_gen_ubatches    = 0;
    int32_t n_gen_tokens_ub   = 0;

    void reset() { *this = {}; }
};

struct llama_context; // forward

struct llama_mtp {
    llama_context * ctx_mtp    = nullptr; // non-owning
    llama_batch     hook_batch = {};      // sized to n_ubatch

    // Ring buffer of recent trunk hidden-state rows, indexed by position.
    // Always contains the most recent N h-rows seen by the hook, where
    // N ≤ n_ubatch. The position-based lookup in handle_mtp_for_ubatch
    // finds the correct h_{pos_start-1} even after the MTP cache has
    // been trimmed by accept() or a checkpoint restore.
    struct h_entry {
        llama_pos          pos;
        std::vector<float> h;
    };
    std::deque<h_entry> h_ring;

    // accumulated hook timing for profiling
    mtp_hook_timing hook_timing;
};
