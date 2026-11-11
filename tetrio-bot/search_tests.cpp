#include "search.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tetrio;
namespace {
constexpr double DEATH_VALUE = -1e12;
void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
int occupied(const Board &board) {
    int count = 0;
    for (auto row : board.rows) count += __builtin_popcount(static_cast<unsigned>(row));
    return count;
}
double heuristic(const State &state) {
    int total = 0, maximum = 0, holes = 0;
    for (int x = 0; x < WIDTH; ++x) {
        int height = 0;
        for (int y = HEIGHT - 1; y >= 0; --y) if (state.board.rows[y] & (1u << x)) {
            height = y + 1;
            break;
        }
        for (int y = 0; y < height; ++y) if (!(state.board.rows[y] & (1u << x))) ++holes;
        total += height;
        maximum = std::max(maximum, height);
    }
    int incoming = 0;
    for (const auto &packet : state.incoming) incoming += packet.amount;
    return -total - 4.0 * holes - 0.5 * maximum * maximum
           + .25 * state.b2b + .125 * state.combo - .25 * incoming;
}

struct CountingEvaluator : SearchEvaluator {
    mutable std::size_t refreshes = 0, updates = 0;
    Evaluation evaluate(const State &state, const State *parent,
                        const Evaluation *parent_value, const Transition *transition) const override {
        Evaluation result;
        if (parent) {
            ++updates;
            require(parent_value && parent_value->valid_accumulator && transition,
                    "A child evaluation did not receive its parent cache and transition");
            require(parent_value->accumulator[0] == occupied(parent->board),
                    "Parent cache belongs to another board");
            const int difference = occupied(state.board) - occupied(parent->board);
            result.accumulator[0] = parent_value->accumulator[0] + difference;
        } else {
            ++refreshes;
            require(!parent_value && !transition, "Root received a partial parent cache");
            result.accumulator[0] = static_cast<float>(occupied(state.board));
        }
        require(result.accumulator[0] == occupied(state.board), "Updated cache is incorrect");
        result.valid_accumulator = true;
        result.value = heuristic(state);
        return result;
    }
    double reward(const Transition &transition) const override {
        return 8.0 * transition.sent + 4.0 * transition.cancelled + 2.0 * transition.lines
               + (transition.perfect_clear ? 12.0 : 0.0) - .25 * transition.received;
    }
};

State empty_state() {
    State state;
    state.current = Piece::I;
    state.queue = {Piece::O, Piece::T, Piece::L, Piece::S, Piece::Z};
    state.pieces_placed = 20;
    return state;
}
State quad_well(Piece current = Piece::I) {
    State state = empty_state();
    state.current = current;
    for (int y = 0; y < 4; ++y) state.board.rows[y] = FULL_ROW ^ 1u;
    return state;
}
SearchOptions exhaustive_options(unsigned depth = 2) {
    SearchOptions options;
    options.max_depth = depth;
    options.beam_width = 200000;
    options.max_nodes = 1000000;
    options.max_cache_nodes = 200000;
    options.time_budget_ms = 0;
    return options;
}
bool same_action(const Action &a, const Action &b) {
    return a.use_hold == b.use_hold && a.elapsed_frames == b.elapsed_frames &&
           a.placement.piece == b.placement.piece && a.placement.spin == b.placement.spin &&
           a.placement.pose.x == b.placement.pose.x && a.placement.pose.y == b.placement.pose.y &&
           a.placement.pose.rotation == b.placement.pose.rotation;
}

// This reference recursively enumerates every visible action without a beam,
// graph, memoization, accumulator reuse, or the search implementation's code.
double exhaustive_value(const State &state, const Rules &rules, const CountingEvaluator &evaluator,
                        unsigned depth, bool allow_hold, int elapsed_frames = 0) {
    if (state.dead) return DEATH_VALUE;
    if (!depth || state.current == Piece::None) return heuristic(state);
    const auto actions = enumerate_actions(state, rules, allow_hold);
    if (actions.empty()) return DEATH_VALUE;
    double best = -std::numeric_limits<double>::infinity();
    for (auto action : actions) {
        action.elapsed_frames = elapsed_frames;
        const Transition transition = apply_action(state, action, rules);
        require(transition.valid, "Exhaustive reference encountered an invalid generated action");
        best = std::max(best, evaluator.reward(transition) +
            exhaustive_value(transition.state, rules, evaluator, depth - 1, allow_hold, elapsed_frames));
    }
    return best;
}
void assert_pv(const State &root, const Rules &rules, const SearchResult &result) {
    State state = root;
    for (const auto &action : result.principal_variation) {
        const auto transition = apply_action(state, action, rules);
        require(transition.valid, "Returned principal variation contains an invalid action");
        state = transition.state;
    }
    require(result.action && !result.principal_variation.empty() &&
            same_action(*result.action, result.principal_variation.front()), "Root action and PV differ");
}

} // namespace

int main() {
    unsigned passed = 0;
    try {
        const Rules rules = Rules::tetra_league();
        {
            CountingEvaluator evaluator;
            for (bool hold : {false, true}) {
                State state = empty_state();
                state.board.rows[0] = 0b0000000011;
                state.board.rows[1] = 0b0000000001;
                auto options = exhaustive_options(2);
                options.allow_hold = hold;
                Searcher searcher(rules, evaluator);
                const auto result = searcher.search(state, options);
                const double reference = exhaustive_value(state, rules, evaluator, 2, hold);
                require(result.stats.completed_depth == 2 && result.termination == "completed",
                        "Exhaustive-width search did not complete two plies");
                require(std::abs(result.value - reference) < 1e-9,
                        "Beam/DAG search differs from independent exhaustive two-ply value");
                require(result.stats.beam_pruned == 0, "Exhaustive-width fixture was pruned");
                assert_pv(state, rules, result);
                const auto chosen = apply_action(state, *result.action, rules);
                const double chosen_value = evaluator.reward(chosen) +
                    exhaustive_value(chosen.state, rules, evaluator, 1, hold);
                require(std::abs(chosen_value - reference) < 1e-9,
                        "Returned root action does not attain the exhaustive optimum");
                ++passed;
            }
        }
        {
            CountingEvaluator evaluator;
            auto state = empty_state();
            state.queue = {Piece::O, Piece::T};
            auto options = exhaustive_options(3);
            options.allow_hold = false;
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, options);
            const double reference = exhaustive_value(state, rules, evaluator, 3, false);
            require(result.stats.completed_depth == 3 && result.principal_variation.size() == 3 &&
                    std::abs(result.value - reference) < 1e-9,
                    "Three-ply search differs from independent exhaustive recursion");
            assert_pv(state, rules, result);
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            auto state = quad_well(Piece::O);
            state.queue = {Piece::I, Piece::T, Piece::L};
            auto options = exhaustive_options(1);
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, options);
            require(result.action && result.action->use_hold, "Empty hold failed to access the next I");
            const auto transition = apply_action(state, *result.action, rules);
            require(transition.lines == 4 && transition.perfect_clear, "Empty-hold quad was not chosen");
            require(transition.state.hold == Piece::O && transition.state.current == Piece::T &&
                    transition.state.queue == std::vector<Piece>{Piece::L},
                    "Empty hold consumed the wrong number of observed pieces");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            auto state = quad_well(Piece::O);
            state.hold = Piece::I;
            state.queue = {Piece::T, Piece::L};
            auto options = exhaustive_options(1);
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, options);
            require(result.action && result.action->use_hold, "Filled hold failed to use the I");
            const auto transition = apply_action(state, *result.action, rules);
            require(transition.lines == 4 && transition.state.current == Piece::T &&
                    transition.state.hold == Piece::O && transition.state.queue.size() == 1,
                    "Filled hold consumed the wrong queue state");
            state.hold_used = true;
            const auto unavailable = searcher.search(state, options);
            require(unavailable.action && !unavailable.action->use_hold,
                    "Search ignored authoritative hold availability");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            auto state = empty_state();
            state.queue.clear();
            auto options = exhaustive_options(4);
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, options);
            require(result.action && !result.action->use_hold && result.principal_variation.size() == 1,
                    "Search invented an unobserved queue piece");
            require(result.stats.completed_depth == 1 && result.stats.unknown_cutoffs > 0 &&
                    result.termination == "unknown_queue", "Unknown horizon was not reported");
            State unknown;
            const auto none = searcher.search(unknown, options);
            require(!none.action && none.termination == "unknown_queue", "Unknown current piece was played");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            const auto state = empty_state();
            auto options = exhaustive_options(4);
            options.max_nodes = 1;
            Searcher a(rules, evaluator), b(rules, evaluator);
            const auto first = a.search(state, options), second = b.search(state, options);
            require(first.action && second.action && same_action(*first.action, *second.action) &&
                    first.value == second.value, "Deterministic node-cap fallback changed");
            require(first.stats.nodes == 1 && first.stats.node_budget_exhausted &&
                    first.stats.completed_depth == 0, "Node cap was exceeded or partial root marked complete");
            assert_pv(state, rules, first);
            options.max_nodes = 10000;
            options.time_budget_ms = 1e-9;
            Searcher deadline(rules, evaluator);
            const auto timed = deadline.search(state, options);
            require(timed.action && timed.stats.deadline_exhausted && timed.stats.nodes == 1,
                    "Cooperative deadline failed to return one legal fallback");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            const auto state = empty_state();
            auto greedy_options = exhaustive_options(1);
            Searcher greedy(rules, evaluator);
            const auto one = greedy.search(state, greedy_options);
            auto options = exhaustive_options(3);
            options.max_nodes = one.stats.nodes + 1;
            Searcher interrupted(rules, evaluator);
            const auto result = interrupted.search(state, options);
            require(result.stats.completed_depth == 1 && result.stats.reached_depth == 2 &&
                    result.stats.node_budget_exhausted && result.value == one.value &&
                    same_action(*result.action, *one.action),
                    "Interrupted next layer replaced the last completed horizon");
            options.max_nodes = 10000;
            options.max_cache_nodes = 2;
            Searcher constrained(rules, evaluator);
            const auto limited = constrained.search(state, options);
            require(limited.action && limited.stats.cache_budget_exhausted && limited.stats.cache_nodes == 2,
                    "Cache bound was exceeded or discarded every legal fallback");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            const auto state = empty_state();
            auto options = exhaustive_options(2);
            Searcher searcher(rules, evaluator);
            const auto first = searcher.search(state, options);
            const auto repeated = searcher.search(state, options);
            require(repeated.stats.reused_nodes > 0 && repeated.stats.evaluations == 0 &&
                    repeated.stats.cache_hits > 0 && first.value == repeated.value &&
                    same_action(*first.action, *repeated.action), "Exact-state DAG reuse is inconsistent");
            const auto child = apply_action(state, *first.action, rules);
            options.max_depth = 1;
            const auto rerooted = searcher.search(child.state, options);
            Searcher fresh(rules, evaluator);
            const auto reference = fresh.search(child.state, options);
            require(rerooted.stats.reused_nodes > 0 && rerooted.stats.evaluations == 0 &&
                    rerooted.value == reference.value && same_action(*rerooted.action, *reference.action),
                    "Re-rooting retained an incompatible cache");
            auto extended = child.state;
            extended.queue.push_back(Piece::J);
            const auto observed = searcher.search(extended, options);
            require(observed.stats.reused_nodes == 0 && observed.stats.evaluations > 0,
                    "New preview was silently treated as an already-known state");
            options.frames_per_piece = 3;
            const auto timed = searcher.search(extended, options);
            require(timed.stats.reused_nodes == 0 && timed.action && timed.action->elapsed_frames == 3,
                    "Changing external time advance reused incompatible edges");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            const auto state = empty_state();
            auto options = exhaustive_options(2);
            Searcher merged(rules, evaluator), tree(rules, evaluator);
            const auto dag = merged.search(state, options);
            options.deduplicate = false;
            const auto full_tree = tree.search(state, options);
            require(std::abs(dag.value - full_tree.value) < 1e-9,
                    "Dominance dedup changed the exhaustive optimum");
            require(dag.stats.duplicate_paths > 0 && dag.stats.evaluations < full_tree.stats.evaluations,
                    "Fixture did not exercise state deduplication");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            auto state = quad_well();
            GarbagePacket packet;
            packet.amount = 4; packet.hole = 5; packet.id = 31;
            state.incoming.push_back(packet);
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, exhaustive_options(1));
            require(result.action.has_value(), "Cancellation fixture has no action");
            const auto transition = apply_action(state, *result.action, rules);
            require(transition.lines == 4 && transition.cancelled == 4 && transition.received == 0,
                    "Search did not evaluate official cancellation before garbage receipt");
            ++passed;
        }
        {
            CountingEvaluator evaluator;
            auto state = empty_state();
            state.board.rows.fill(FULL_ROW);
            Searcher searcher(rules, evaluator);
            const auto result = searcher.search(state, exhaustive_options(2));
            require(!result.action, "Search returned a placement in an entirely blocked board");
            state.dead = true;
            const auto dead = searcher.search(state, exhaustive_options(2));
            require(!dead.action && dead.termination == "dead", "Authoritative death state was ignored");
            ++passed;
        }
        {
            State base = empty_state();
            base.board.rows[0] = 3;
            base.incoming.push_back(GarbagePacket{2, 3, 1, true, true, false, false, 11});
            const std::string original = state_key(base);
            const std::vector<std::function<void(State&)>> changes = {
                [](State &s){s.board.rows[0] = 7;}, [](State &s){s.board.garbage[0] = 1;},
                [](State &s){s.board.solid[0] = 1;}, [](State &s){s.current = Piece::Z;},
                [](State &s){s.hold = Piece::J;}, [](State &s){s.queue.push_back(Piece::I);},
                [](State &s){s.hold_used = true;}, [](State &s){s.last_clear = true;},
                [](State &s){s.dead = true;}, [](State &s){++s.combo;}, [](State &s){++s.b2b;},
                [](State &s){++s.pieces_placed;}, [](State &s){s.total_sent += 1;},
                [](State &s){++s.frame;}, [](State &s){++s.last_tank;},
                [](State &s){s.current_multiplier = 1;}, [](State &s){s.current_garbage_cap = 8;},
                [](State &s){++s.rng_seed;}, [](State &s){s.last_hole = 3;},
                [](State &s){s.hole_changed = true;}, [](State &s){++s.enemies;},
                [](State &s){s.garbage_bonus = 1;}, [](State &s){++s.incoming[0].amount;},
                [](State &s){++s.incoming[0].hole;}, [](State &s){++s.incoming[0].width;},
                [](State &s){s.incoming[0].active = false;}, [](State &s){s.incoming[0].spawn_ready = false;},
                [](State &s){s.incoming[0].hardened = true;}, [](State &s){s.incoming[0].shielded = true;},
                [](State &s){++s.incoming[0].id;}, [](State &s){s.incoming.push_back(s.incoming[0]);}
            };
            for (const auto &change : changes) {
                State modified = base;
                change(modified);
                require(state_key(modified) != original, "Battle context omitted from state key");
            }
            ++passed;
        }
        std::cout << "{\"search_test_groups\":" << passed
                  << ",\"passed\":true,\"scope\":\"exhaustive-reference, hold, visible horizon, budgets, DAG reuse, context keys and battle cancellation\"}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Search test group " << passed + 1 << " failed: " << error.what() << '\n';
        return 1;
    }
}
