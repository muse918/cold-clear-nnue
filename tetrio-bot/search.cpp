#include "search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tetrio {
namespace {
constexpr double DEATH_VALUE = -1.0e12;
constexpr std::size_t NO_ID = std::numeric_limits<std::size_t>::max();
bool known(Piece piece) { return static_cast<int>(piece) >= 0 && static_cast<int>(piece) < 7; }

void check_options(const SearchOptions &o) {
    if (!o.max_depth || !o.beam_width || !o.max_nodes || o.max_cache_nodes < 2 ||
        o.frames_per_piece < 0 || !std::isfinite(o.time_budget_ms) || o.time_budget_ms < 0)
        throw std::invalid_argument("Invalid search depth, width, budget, cache size or elapsed time");
}
} // namespace

struct Searcher::Impl {
    struct Edge {
        Action action;
        double reward = 0;
        std::size_t child = 0;
    };
    struct Node {
        State state;
        std::string key;
        Evaluation evaluation;
        std::vector<Edge> children;
        bool expanded = false;
        bool no_legal_actions = false;
    };
    struct Path {
        std::size_t node = 0;
        std::size_t parent = NO_ID;
        std::size_t parent_node = NO_ID;
        std::size_t edge = NO_ID;
        std::size_t root_action = NO_ID;
        unsigned depth = 0;
        double reward = 0;
    };

    Rules rules;
    const SearchEvaluator *evaluator;
    std::vector<Node> nodes;
    // The complete canonical state bytes are keys, not an unchecked 64-bit hash.
    std::unordered_map<std::string, std::size_t> index;
    bool configured = false;
    bool allow_hold = true;
    bool deduplicate = true;
    int frames_per_piece = 0;

    Impl(Rules r, const SearchEvaluator &e) : rules(std::move(r)), evaluator(&e) {
        rules.validate();
    }

    void clear() { nodes.clear(); index.clear(); }

    void retain_descendants(std::size_t root) {
        if (!root) return;
        std::vector<std::size_t> remap(nodes.size(), NO_ID), order{root};
        remap[root] = 0;
        for (std::size_t i = 0; i < order.size(); ++i) {
            for (const Edge &edge : nodes[order[i]].children) {
                if (remap[edge.child] == NO_ID) {
                    remap[edge.child] = order.size();
                    order.push_back(edge.child);
                }
            }
        }
        std::vector<Node> retained;
        retained.reserve(order.size());
        for (std::size_t id : order) retained.push_back(std::move(nodes[id]));
        for (Node &node : retained)
            for (Edge &edge : node.children) edge.child = remap[edge.child];
        nodes = std::move(retained);
        index.clear();
        for (std::size_t i = 0; i < nodes.size(); ++i) index.emplace(nodes[i].key, i);
    }

    bool terminal(const Node &node) const { return node.state.dead || node.no_legal_actions; }
    double score(const Path &path) const {
        const Node &node = nodes[path.node];
        return path.reward + (terminal(node) ? DEATH_VALUE : node.evaluation.value);
    }

    SearchResult run(const State &state, const SearchOptions &o) {
        check_options(o);
        const auto started = std::chrono::steady_clock::now();
        SearchResult out;
        auto elapsed = [&]() {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        };
        if (!configured || allow_hold != o.allow_hold || deduplicate != o.deduplicate ||
            frames_per_piece != o.frames_per_piece || nodes.size() > o.max_cache_nodes) {
            clear();
            configured = true;
            allow_hold = o.allow_hold;
            deduplicate = o.deduplicate;
            frames_per_piece = o.frames_per_piece;
        }

        const std::string root_key = state_key(state);
        auto root_found = index.find(root_key);
        std::size_t root_id;
        if (root_found == index.end()) {
            // An unexpected garbage/timing/preview observation invalidates the
            // retained graph. No attempt is made to guess an unobserved queue.
            clear();
            Evaluation evaluation;
            if (state.dead) evaluation.value = DEATH_VALUE;
            else {
                evaluation = evaluator->evaluate(state, nullptr, nullptr, nullptr);
                ++out.stats.evaluations;
                if (!std::isfinite(evaluation.value))
                    throw std::runtime_error("Non-finite root evaluation");
            }
            nodes.push_back(Node{state, root_key, std::move(evaluation), {}, false, false});
            index.emplace(root_key, 0);
            root_id = 0;
        } else {
            root_id = root_found->second;
            retain_descendants(root_id);
            root_id = 0;
            out.stats.reused_nodes = nodes.size();
        }

        std::vector<Path> paths;
        paths.reserve(std::min<std::size_t>(o.max_nodes, 999999) + 1);
        paths.push_back(Path{root_id, NO_ID, NO_ID, NO_ID, NO_ID, 0, 0});
        std::vector<std::size_t> frontier{0};
        std::size_t best_complete = NO_ID;
        std::size_t best_root_fallback = NO_ID;
        bool interrupted = false;

        auto better = [&](std::size_t a, std::size_t b) {
            if (b == NO_ID) return true;
            const double va = score(paths[a]), vb = score(paths[b]);
            return va > vb || (va == vb && a < b);
        };
        auto budget_available = [&]() {
            if (out.stats.nodes >= o.max_nodes) {
                out.stats.node_budget_exhausted = true;
                return false;
            }
            // Always permit a legal root fallback after geometry enumeration.
            if (out.stats.nodes && o.time_budget_ms > 0 && elapsed() >= o.time_budget_ms) {
                out.stats.deadline_exhausted = true;
                return false;
            }
            return true;
        };
        auto materialize = [&](std::size_t id) {
            std::vector<Action> result;
            while (paths[id].parent != NO_ID) {
                const Path &path = paths[id];
                result.push_back(nodes[path.parent_node].children[path.edge].action);
                id = path.parent;
            }
            std::reverse(result.begin(), result.end());
            return result;
        };

        for (unsigned depth = 1; depth <= o.max_depth; ++depth) {
            std::vector<std::size_t> next;
            bool made_child = false;
            for (std::size_t path_id : frontier) {
                const Path parent_path = paths[path_id];
                const std::size_t node_id = parent_path.node;
                if (terminal(nodes[node_id])) {
                    ++out.stats.terminal_cutoffs;
                    next.push_back(path_id);
                    continue;
                }
                if (!known(nodes[node_id].state.current)) {
                    ++out.stats.unknown_cutoffs;
                    next.push_back(path_id);
                    continue;
                }
                if (!budget_available()) { interrupted = true; break; }

                // Local parent copies stay valid if the node vector reallocates
                // while new children are appended. One copy per parent, not per child.
                const State parent_state = nodes[node_id].state;
                const Evaluation parent_evaluation = nodes[node_id].evaluation;
                std::vector<Action> actions;
                const bool already_expanded = nodes[node_id].expanded;
                if (!already_expanded) {
                    actions = enumerate_actions(parent_state, rules, o.allow_hold);
                    for (auto &action : actions) action.elapsed_frames = o.frames_per_piece;
                    ++out.stats.expanded;
                    if (actions.size() < nodes[node_id].children.size())
                        throw std::logic_error("Placement enumeration changed for an identical state");
                    if (actions.empty()) {
                        nodes[node_id].expanded = true;
                        nodes[node_id].no_legal_actions = true;
                        ++out.stats.terminal_cutoffs;
                        next.push_back(path_id);
                        continue;
                    }
                }
                const std::size_t count = already_expanded ? nodes[node_id].children.size() : actions.size();
                for (std::size_t edge_id = 0; edge_id < count; ++edge_id) {
                    if (!budget_available()) { interrupted = true; break; }
                    if (edge_id >= nodes[node_id].children.size()) {
                        const Action &action = actions[edge_id];
                        Transition transition = apply_action(parent_state, action, rules);
                        if (!transition.valid)
                            throw std::logic_error("Generated action was rejected: " + transition.error);
                        const double reward = evaluator->reward(transition);
                        if (!std::isfinite(reward)) throw std::runtime_error("Non-finite edge reward");
                        const std::string key = state_key(transition.state);
                        auto found = o.deduplicate ? index.find(key) : index.end();
                        std::size_t child_id;
                        if (found != index.end()) {
                            child_id = found->second;
                            ++out.stats.cache_hits;
                        } else {
                            if (nodes.size() >= o.max_cache_nodes) {
                                out.stats.cache_budget_exhausted = true;
                                interrupted = true;
                                break;
                            }
                            Evaluation evaluation;
                            if (transition.state.dead) evaluation.value = DEATH_VALUE;
                            else {
                                evaluation = evaluator->evaluate(transition.state, &parent_state,
                                                                 &parent_evaluation, &transition);
                                ++out.stats.evaluations;
                                if (!std::isfinite(evaluation.value))
                                    throw std::runtime_error("Non-finite child evaluation");
                            }
                            child_id = nodes.size();
                            nodes.push_back(Node{std::move(transition.state), key,
                                                 std::move(evaluation), {}, false, false});
                            if (o.deduplicate) index.emplace(key, child_id);
                        }
                        nodes[node_id].children.push_back(Edge{action, reward, child_id});
                    } else ++out.stats.cache_hits;

                    ++out.stats.nodes;
                    const Edge &edge = nodes[node_id].children[edge_id];
                    const std::size_t root_action = parent_path.depth ? parent_path.root_action : edge_id;
                    const std::size_t child_path_id = paths.size();
                    paths.push_back(Path{edge.child, path_id, node_id, edge_id, root_action,
                                         parent_path.depth + 1, parent_path.reward + edge.reward});
                    next.push_back(child_path_id);
                    made_child = true;
                    out.stats.reached_depth = std::max(out.stats.reached_depth, paths.back().depth);
                    if (!parent_path.depth) {
                        if (out.candidates.size() != edge_id)
                            throw std::logic_error("Root candidate order is inconsistent");
                        out.candidates.push_back(RootCandidate{edge.action, score(paths.back()), 1,
                                                              !terminal(nodes[edge.child])});
                        if (better(child_path_id, best_root_fallback)) best_root_fallback = child_path_id;
                    }
                }
                if (!interrupted && !already_expanded) nodes[node_id].expanded = true;
                if (interrupted) break;
            }
            if (interrupted) break;
            if (!made_child) break; // Only unknown or terminal states remain.

            // Scores for a root move are replaced at each completed layer, not
            // maximized across incompatible search horizons. Pruned moves retain
            // their last searched depth explicitly in RootCandidate::depth.
            std::vector<std::size_t> root_best(out.candidates.size(), NO_ID);
            for (std::size_t id : next) {
                const auto root = paths[id].root_action;
                if (root != NO_ID && better(id, root_best[root])) root_best[root] = id;
            }
            for (std::size_t i = 0; i < root_best.size(); ++i) {
                if (root_best[i] == NO_ID) continue;
                const auto &path = paths[root_best[i]];
                out.candidates[i].value = score(path);
                out.candidates[i].depth = path.depth;
                out.candidates[i].survives = !terminal(nodes[path.node]);
            }

            if (o.deduplicate) {
                std::unordered_map<std::string, std::size_t> unique;
                std::vector<std::size_t> merged;
                merged.reserve(next.size());
                for (std::size_t id : next) {
                    const auto &key = nodes[paths[id].node].key;
                    auto inserted = unique.emplace(key, merged.size());
                    if (inserted.second) merged.push_back(id);
                    else {
                        ++out.stats.duplicate_paths;
                        auto &retained = merged[inserted.first->second];
                        if (better(id, retained)) retained = id;
                    }
                }
                next = std::move(merged);
            }
            std::sort(next.begin(), next.end(), better);
            if (next.size() > o.beam_width) {
                out.stats.beam_pruned += next.size() - o.beam_width;
                next.resize(o.beam_width);
            }
            if (next.empty()) break;
            best_complete = next.front();
            out.stats.completed_depth = depth;
            frontier = std::move(next);
        }

        const std::size_t best = best_complete != NO_ID ? best_complete : best_root_fallback;
        if (best != NO_ID) {
            out.principal_variation = materialize(best);
            if (!out.principal_variation.empty()) out.action = out.principal_variation.front();
            out.value = score(paths[best]);
        } else out.value = terminal(nodes[root_id]) ? DEATH_VALUE : nodes[root_id].evaluation.value;

        if (out.stats.cache_budget_exhausted) out.termination = "cache_budget";
        else if (out.stats.node_budget_exhausted) out.termination = "node_budget";
        else if (out.stats.deadline_exhausted) out.termination = "deadline";
        else if (state.dead) out.termination = "dead";
        else if (nodes[root_id].no_legal_actions) out.termination = "no_legal_actions";
        else if (best != NO_ID && terminal(nodes[paths[best].node]))
            out.termination = "no_surviving_continuation";
        else if (!known(state.current) || (out.stats.completed_depth < o.max_depth && out.stats.unknown_cutoffs))
            out.termination = "unknown_queue";
        else if (!out.action) out.termination = "no_legal_actions";
        else out.termination = "completed";
        out.stats.cache_nodes = nodes.size();
        out.stats.elapsed_ms = elapsed();
        return out;
    }
};

Searcher::Searcher(Rules rules, const SearchEvaluator &evaluator)
    : impl_(std::make_unique<Impl>(std::move(rules), evaluator)) {}
Searcher::~Searcher() = default;
Searcher::Searcher(Searcher &&) noexcept = default;
Searcher &Searcher::operator=(Searcher &&) noexcept = default;
SearchResult Searcher::search(const State &state, const SearchOptions &options) {
    return impl_->run(state, options);
}
void Searcher::clear() { impl_->clear(); }

} // namespace tetrio
