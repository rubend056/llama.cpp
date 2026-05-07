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
//      ring buffer (size = n_ubatch).
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
};
