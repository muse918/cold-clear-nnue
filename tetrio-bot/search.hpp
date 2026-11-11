#pragma once

#include "engine.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tetrio {

// A fixed-size, allocation-free cache for the existing native NNUE accumulator.
// Adapters can use [0,128) for position and [128,272) for local sums. The search
// treats it as opaque and never assumes that equal scores imply equal caches.
struct Evaluation {
    double value = 0.0;
    alignas(64) std::array<float, 272> accumulator{};
    bool valid_accumulator = false;
};

class SearchEvaluator {
public:
    virtual ~SearchEvaluator() = default;
    virtual Evaluation evaluate(const State &state, const State *parent,
                                const Evaluation *parent_evaluation,
                                const Transition *transition) const = 0;
    virtual double reward(const Transition &transition) const = 0;
};

struct SearchOptions {
    unsigned max_depth = 4;
    std::size_t beam_width = 64;
    // Hard limit on processed candidate edges, including cached edges. The
    // root's initial evaluation is additional and is reported separately.
    std::size_t max_nodes = 12000;
    // Zero disables the wall-clock deadline. A deadline is cooperative: one
    // geometry enumeration plus one child evaluation can finish after it.
    double time_budget_ms = 50.0;
    std::size_t max_cache_nodes = 32768;
    bool allow_hold = true;
    bool deduplicate = true;
    int frames_per_piece = 0;
};

struct RootCandidate {
    Action action;
    double value = 0.0;
    unsigned depth = 1;
    bool survives = true;
};

struct SearchStats {
    std::size_t nodes = 0;             // Candidate edges processed this call.
    std::size_t evaluations = 0;       // Fresh evaluator calls, including root.
    std::size_t expanded = 0;          // Geometry enumerations this call.
    std::size_t cache_hits = 0;        // Reused child states/evaluations.
    std::size_t duplicate_paths = 0;   // Dominated paths removed within a layer.
    std::size_t beam_pruned = 0;
    std::size_t unknown_cutoffs = 0;
    std::size_t terminal_cutoffs = 0;
    std::size_t reused_nodes = 0;
    std::size_t cache_nodes = 0;
    unsigned completed_depth = 0;
    unsigned reached_depth = 0;
    bool node_budget_exhausted = false;
    bool deadline_exhausted = false;
    bool cache_budget_exhausted = false;
    double elapsed_ms = 0.0;
};

struct SearchResult {
    std::optional<Action> action;
    std::vector<Action> principal_variation;
    std::vector<RootCandidate> candidates;
    double value = 0.0;
    SearchStats stats;
    // completed, node_budget, deadline, cache_budget, unknown_queue, dead,
    // no_legal_actions, no_surviving_continuation. A dead searched continuation
    // is not a proof that every pruned move loses. No hidden pieces are sampled.
    std::string termination;
};

// The rules and evaluator are immutable for the lifetime of a searcher.
// Exact state keys include queue/hold/battle/timing/RNG context. A changed
// observation that does not match a retained state starts a fresh DAG; an
// identical state can reuse generated edges and accumulators. Preview extension
// is not treated as an already-known random draw or an upstream speculation.
class Searcher {
public:
    Searcher(Rules rules, const SearchEvaluator &evaluator);
    ~Searcher();
    Searcher(Searcher &&) noexcept;
    Searcher &operator=(Searcher &&) noexcept;
    Searcher(const Searcher &) = delete;
    Searcher &operator=(const Searcher &) = delete;

    SearchResult search(const State &state, const SearchOptions &options = {});
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tetrio
