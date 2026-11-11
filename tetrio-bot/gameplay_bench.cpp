#include "evaluator.hpp"
#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tetrio;
using json = nlohmann::json;

namespace {
constexpr unsigned GAMES = 8;
constexpr unsigned PIECE_CAP = 120;
constexpr unsigned PREVIEWS = 5;
constexpr int FRAMES_PER_PIECE = 30;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// Test-fixture RNG, not TETR.IO's replay/bag RNG. An explicit Fisher-Yates
// shuffle avoids implementation-dependent std::shuffle sequences. Each game
// gives all policies the same referee-owned stream, never its hidden suffix.
struct FixtureRng {
    uint32_t state;
    explicit FixtureRng(uint32_t seed) : state(seed ? seed : 1) {}
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    uint32_t below(uint32_t bound) {
        const uint32_t threshold = uint32_t(-bound) % bound;
        uint32_t value;
        do { value = next(); } while (value < threshold);
        return value % bound;
    }
};

std::vector<Piece> fixture_queue(uint32_t seed) {
    FixtureRng rng(seed);
    std::vector<Piece> queue;
    while (queue.size() < PIECE_CAP + PREVIEWS + 16) {
        std::array<Piece, 7> bag{{Piece::I, Piece::O, Piece::T, Piece::L,
                                 Piece::J, Piece::S, Piece::Z}};
        for (unsigned i = 6; i; --i) std::swap(bag[i], bag[rng.below(i + 1)]);
        queue.insert(queue.end(), bag.begin(), bag.end());
    }
    return queue;
}

struct Checksum {
    uint64_t value = UINT64_C(14695981039346656037);
    void bytes(const std::string &bytes) {
        for (unsigned char byte : bytes) {
            value ^= byte;
            value *= UINT64_C(1099511628211);
        }
    }
    void number(uint64_t number) {
        for (unsigned i = 0; i < 8; ++i) {
            value ^= uint8_t(number >> (8 * i));
            value *= UINT64_C(1099511628211);
        }
    }
    std::string hex() const {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << value;
        return output.str();
    }
};

bool same_action(const Action &a, const Action &b) {
    return a.use_hold == b.use_hold && a.placement.piece == b.placement.piece &&
           a.placement.pose.x == b.placement.pose.x &&
           a.placement.pose.y == b.placement.pose.y &&
           a.placement.pose.rotation == b.placement.pose.rotation &&
           a.placement.spin == b.placement.spin &&
           a.placement.last_rotation == b.placement.last_rotation &&
           a.placement.rotated180 == b.placement.rotated180 &&
           a.placement.kick_index == b.placement.kick_index &&
           a.placement.path == b.placement.path;
}

void hash_action(Checksum &hash, const Action &action, const Transition &transition) {
    hash.number(static_cast<uint64_t>(action.placement.piece));
    hash.number(static_cast<uint64_t>(action.placement.pose.x));
    hash.number(static_cast<uint64_t>(action.placement.pose.y));
    hash.number(static_cast<uint64_t>(action.placement.pose.rotation));
    hash.number(static_cast<uint64_t>(action.placement.spin));
    hash.number(action.use_hold);
    hash.number(action.elapsed_frames);
    hash.number(transition.lines);
    hash.number(transition.attack);
    hash.number(transition.sent);
    hash.number(transition.cancelled);
    hash.number(transition.received);
    hash.bytes(state_key(transition.state));
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double index = fraction * (values.size() - 1);
    const auto lower = static_cast<std::size_t>(index);
    const auto upper = std::min(values.size() - 1, lower + 1);
    return values[lower] + (index - lower) * (values[upper] - values[lower]);
}

struct Policy {
    std::string label;
    const BotEvaluator *evaluator;
    unsigned depth;
};

json play(const Policy &policy, unsigned game) {
    const Rules rules = Rules::tetra_league();
    const uint32_t queue_seed = UINT32_C(20261112) + UINT32_C(104729) * game;
    const uint32_t garbage_seed = UINT32_C(0x9e3779b9) ^ (UINT32_C(7919) * (game + 1));
    const auto stream = fixture_queue(queue_seed);
    FixtureRng garbage_rng(garbage_seed);
    std::array<int, PIECE_CAP / 12> holes{};
    for (auto &hole : holes) hole = static_cast<int>(garbage_rng.below(WIDTH));
    const bool with_garbage = game % 2 != 0;

    Checksum queue_hash, schedule_hash;
    for (auto piece : stream) queue_hash.number(static_cast<uint64_t>(piece));
    if (with_garbage) for (unsigned i = 0; i < holes.size(); ++i) {
        schedule_hash.number((i + 1) * 12);
        schedule_hash.number(2);
        schedule_hash.number(static_cast<uint64_t>(holes[i]));
    }

    State state;
    state.current = stream[0];
    state.queue.assign(stream.begin() + 1, stream.begin() + PREVIEWS + 1);
    state.rng_seed = UINT32_C(1000003) + game;
    std::size_t next_draw = PREVIEWS + 1;
    SearchOptions options;
    options.max_depth = policy.depth;
    options.beam_width = 16;
    options.max_nodes = 2000;
    options.time_budget_ms = 0;
    options.frames_per_piece = FRAMES_PER_PIECE;
    Searcher searcher(rules, *policy.evaluator);

    json result = {
        {"policy", policy.label}, {"game", game}, {"scenario", with_garbage ? "fixed_incoming" : "clean"},
        {"queue_seed", queue_seed}, {"garbage_seed", garbage_seed},
        {"referee_queue_fnv1a64", queue_hash.hex()}, {"garbage_schedule_fnv1a64", schedule_hash.hex()}
    };
    uint64_t pieces = 0, lines = 0, attack = 0, sent = 0, cancelled = 0, received = 0;
    uint64_t spins = 0, full_spins = 0, mini_spins = 0, spin_clears = 0, holds = 0;
    uint64_t empty_hold_uses = 0, perfect_clears = 0, incoming_rows = 0, surge = 0;
    uint64_t nodes = 0, evaluations = 0, cache_hits = 0, reused_nodes = 0;
    uint64_t duplicate_paths = 0, expanded = 0, node_budget_stops = 0;
    unsigned max_height = 0, max_b2b = 0, max_combo = 0;
    std::map<std::string, uint64_t> completed_depth, termination_counts;
    std::vector<double> decision_ms;
    Checksum trajectory;
    trajectory.bytes(state_key(state));
    std::string end = "piece_cap";
    const auto started = std::chrono::steady_clock::now();

    for (unsigned lock = 1; lock <= PIECE_CAP; ++lock) {
        if (state.dead) { end = "dead"; break; }
        require(state.current != Piece::None && state.queue.size() == PREVIEWS,
                "Referee failed to provide exactly the visible queue");
        if (with_garbage && lock % 12 == 0) {
            GarbagePacket packet;
            packet.amount = 2;
            packet.hole = holes[lock / 12 - 1];
            packet.id = static_cast<int64_t>(game * 1000 + lock / 12);
            incoming_rows += enqueue_garbage(state, packet, rules);
        }
        // Search receives only this public snapshot. The hidden stream, its RNG
        // and the upcoming garbage schedule never enter State or Searcher.
        const SearchResult decision = searcher.search(state, options);
        nodes += decision.stats.nodes;
        evaluations += decision.stats.evaluations;
        cache_hits += decision.stats.cache_hits;
        reused_nodes += decision.stats.reused_nodes;
        duplicate_paths += decision.stats.duplicate_paths;
        expanded += decision.stats.expanded;
        node_budget_stops += decision.stats.node_budget_exhausted;
        ++completed_depth[std::to_string(decision.stats.completed_depth)];
        ++termination_counts[decision.termination];
        decision_ms.push_back(decision.stats.elapsed_ms);
        require(decision.stats.nodes <= options.max_nodes, "Search exceeded its hard candidate budget");
        if (!decision.action) { end = decision.termination; break; }
        const auto &action = *decision.action;
        const auto legal = enumerate_actions(state, rules, true);
        require(std::any_of(legal.begin(), legal.end(), [&](const Action &candidate) {
                    return same_action(action, candidate);
                }), "Search selected an action outside the authoritative geometric action set");
        require(action.elapsed_frames == FRAMES_PER_PIECE, "External clock advance was lost");
        const bool empty_hold = action.use_hold && state.hold == Piece::None;
        const std::size_t expected_remaining = PREVIEWS - 1 - static_cast<unsigned>(empty_hold);
        const Transition transition = apply_action(state, action, rules);
        require(transition.valid, "Chosen action failed to apply: " + transition.error);
        require(transition.state.queue.size() == expected_remaining,
                "Hold/queue consumption differs from the referee contract");
        require(transition.state.pieces_placed == static_cast<int>(pieces + 1),
                "Authoritative lock counter is inconsistent");
        hash_action(trajectory, action, transition);

        ++pieces;
        lines += transition.lines;
        attack += transition.attack;
        sent += transition.sent;
        cancelled += transition.cancelled;
        received += transition.received;
        surge += transition.surge;
        holds += action.use_hold;
        empty_hold_uses += empty_hold;
        spins += transition.spin != Spin::None;
        full_spins += transition.spin == Spin::Full;
        mini_spins += transition.spin == Spin::Mini;
        spin_clears += transition.spin != Spin::None && transition.lines != 0;
        perfect_clears += transition.perfect_clear;
        state = transition.state;
        max_height = std::max(max_height, static_cast<unsigned>(state.board.height()));
        max_b2b = std::max(max_b2b, static_cast<unsigned>(state.b2b));
        max_combo = std::max(max_combo, static_cast<unsigned>(state.combo));

        // New previews become public only after the actual lock. They invalidate
        // exact-state graph reuse when they were absent from the search tree.
        while (state.queue.size() < PREVIEWS) {
            require(next_draw < stream.size(), "Referee stream exhausted");
            state.queue.push_back(stream[next_draw++]);
        }
        require(next_draw == PREVIEWS + 1 + pieces + empty_hold_uses,
                "The observation window skipped or repeated a piece");
        require(state.frame == static_cast<int64_t>(pieces * FRAMES_PER_PIECE),
                "External elapsed frame counter is inconsistent");
        trajectory.bytes(state_key(state));
    }
    if (state.dead) end = "dead";
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const double search_ms = std::accumulate(decision_ms.begin(), decision_ms.end(), 0.0);
    uint64_t pending = 0;
    for (const auto &packet : state.incoming) pending += packet.amount;
    require(cancelled + received + pending == incoming_rows,
            "Fixed incoming rows were lost or duplicated");
    result.update({
        {"pieces", pieces}, {"lines", lines}, {"attack", attack}, {"sent", sent},
        {"cancelled", cancelled}, {"received", received}, {"pending", pending},
        {"scheduled_incoming_rows", incoming_rows}, {"surge", surge},
        {"spins", spins}, {"full_spins", full_spins}, {"mini_spins", mini_spins},
        {"spin_clears", spin_clears}, {"holds", holds}, {"empty_hold_uses", empty_hold_uses},
        {"perfect_clears", perfect_clears}, {"max_height", max_height},
        {"final_height", state.board.height()}, {"max_b2b", max_b2b}, {"max_combo", max_combo},
        {"nodes", nodes}, {"evaluations", evaluations}, {"cache_hits", cache_hits},
        {"reused_nodes", reused_nodes}, {"duplicate_paths", duplicate_paths}, {"expanded", expanded},
        {"node_budget_stops", node_budget_stops}, {"completed_depth_histogram", completed_depth},
        {"termination_counts", termination_counts}, {"decisions", decision_ms.size()},
        {"search_ms", search_ms}, {"mean_search_ms", search_ms / std::max<std::size_t>(1, decision_ms.size())},
        {"p50_search_ms", percentile(decision_ms, .50)}, {"p95_search_ms", percentile(decision_ms, .95)},
        {"max_search_ms", percentile(decision_ms, 1)}, {"elapsed_ms", elapsed_ms},
        {"revealed_pieces", next_draw}, {"max_exposed_previews", PREVIEWS},
        {"all_chosen_actions_valid", true}, {"hard_node_budget_respected", true},
        {"queue_consumption_verified", true}, {"garbage_accounting_verified", true},
        {"trajectory_fnv1a64", trajectory.hex()}, {"final_state_fnv1a64", [&]() {
            Checksum hash; hash.bytes(state_key(state)); return hash.hex(); }()},
        {"ending", end}, {"reached_piece_cap_alive", pieces == PIECE_CAP && !state.dead},
        {"final_frame", state.frame}, {"final_total_sent", state.total_sent}
    });
    return result;
}

json summarize(const json &games, const std::string &policy, const std::string &scenario) {
    json result = {{"policy", policy}, {"scenario", scenario}};
    const std::vector<std::string> summed = {
        "pieces", "lines", "attack", "sent", "cancelled", "received", "pending",
        "scheduled_incoming_rows", "surge", "spins", "full_spins", "mini_spins",
        "spin_clears", "holds", "empty_hold_uses", "perfect_clears", "nodes",
        "evaluations", "cache_hits", "reused_nodes", "duplicate_paths", "expanded",
        "node_budget_stops", "decisions"
    };
    for (const auto &field : summed) result[field] = uint64_t(0);
    unsigned count = 0, capped = 0;
    double search_ms = 0, elapsed_ms = 0;
    std::map<std::string, uint64_t> depth_histogram;
    for (const auto &game : games) {
        if (game.at("policy") != policy || (scenario != "all" && game.at("scenario") != scenario)) continue;
        ++count;
        capped += game.at("reached_piece_cap_alive").get<bool>();
        for (const auto &field : summed)
            result[field] = result.at(field).get<uint64_t>() + game.at(field).get<uint64_t>();
        search_ms += game.at("search_ms").get<double>();
        elapsed_ms += game.at("elapsed_ms").get<double>();
        for (auto it = game.at("completed_depth_histogram").begin();
             it != game.at("completed_depth_histogram").end(); ++it)
            depth_histogram[it.key()] += it.value().get<uint64_t>();
    }
    result["games"] = count;
    result["reached_piece_cap_alive"] = capped;
    result["search_ms"] = search_ms;
    result["elapsed_ms"] = elapsed_ms;
    result["mean_search_ms"] = search_ms / std::max<uint64_t>(1, result.at("decisions").get<uint64_t>());
    result["completed_depth_histogram"] = depth_histogram;
    result["sent_per_piece"] = double(result.at("sent").get<uint64_t>()) /
                               std::max<uint64_t>(1, result.at("pieces").get<uint64_t>());
    return result;
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc <= 2, "Usage: gameplay_bench [attention.snn]");
        const std::string model = argc == 2 ? argv[1] :
            "experiments/spatial-nnue/results/final/seed918/attention.snn";
        EvaluatorOptions heuristic_options;
        heuristic_options.mode = EvaluationMode::Heuristic;
        BotEvaluator heuristic({}, heuristic_options);
        EvaluatorOptions hybrid_options;
        hybrid_options.mode = EvaluationMode::Hybrid;
        hybrid_options.update = UpdatePolicy::RefreshEvents;
        hybrid_options.nnue_weight = .25;
        BotEvaluator hybrid(model, hybrid_options);
        require(hybrid.has_model(), "The hybrid gameplay run requires an actual loaded NNUE model");
        const std::array<Policy, 3> policies{{
            {"greedy_heuristic", &heuristic, 1},
            {"beam_heuristic", &heuristic, 3},
            {"beam_hybrid_nnue", &hybrid, 3}
        }};
        json games = json::array();
        // Interleave policies by game rather than running one entire policy
        // first. Timings are still coarse functional measurements, not isolated.
        for (unsigned game = 0; game < GAMES; ++game) {
            for (const auto &policy : policies) {
                auto result = play(policy, game);
                std::cerr << policy.label << " game=" << game << " pieces=" << result.at("pieces")
                          << " sent=" << result.at("sent") << " ending=" << result.at("ending") << '\n';
                games.push_back(std::move(result));
            }
        }
        json summaries = json::array();
        for (const auto &policy : policies)
            for (const std::string scenario : {"all", "clean", "fixed_incoming"})
                summaries.push_back(summarize(games, policy.label, scenario));
        const json output = {
            {"schema", 1}, {"experiment", "placement-level TETR.IO League offline gameplay smoke"},
            {"environment", "ported supplied tetrio.js Beta 1.7.8 engine 19"},
            {"games_per_policy", GAMES}, {"policies", policies.size()}, {"piece_cap", PIECE_CAP},
            {"rules", "Rules::tetra_league()"}, {"model", model},
            {"model_training_scope", "existing seed918 attention student trained on prior surrogate labels; not battle-trained"},
            {"nnue_weight", .25}, {"update_policy", "refresh events; incremental non-clear placements"},
            {"search", {{"greedy_depth", 1}, {"beam_depth", 3}, {"beam_width", 16},
                         {"max_nodes", 2000}, {"time_budget_ms", 0}, {"deduplicate", true},
                         {"allow_hold", true}, {"max_cache_nodes", 32768}}},
            {"observations", {{"previews_excluding_current", PREVIEWS},
                               {"hidden_queue_passed_to_search", false},
                               {"future_garbage_passed_to_search", false},
                               {"preview_refill", "after each real lock only"}}},
            {"fixture", {{"queue", "independent xorshift32 Fisher-Yates 7-bags with rejection-sampled bounded draws"},
                         {"clean_games", GAMES / 2}, {"incoming_games", GAMES / 2},
                         {"garbage", "two ready known-hole rows before locks 12,24,...,120; independent RNG"},
                         {"elapsed_frames_per_lock", FRAMES_PER_PIECE}, {"assumed_locks_per_second", 2},
                         {"frame_physics", false}, {"clock", "externally supplied fixed cadence"}}},
            {"limitations", json::array({
                "No opponent, networking, live service, input timing or gravity simulation.",
                "Small capped smoke sample; survival and attack totals do not establish competitive strength.",
                "Elapsed timings are shared-host functional measurements, not isolated kernel benchmarks.",
                "Node limits can return the last completed shallower layer; completed depths are reported.",
                "New previews invalidate exact-state cross-move caches; cache_hits also count within-search DAG reuse.",
                "Queue generation is an explicit fixture, not an implementation of official replay bag RNG.",
                "FNV-1a checksums detect trajectory differences and are not cryptographic hashes."
            })},
            {"checks_passed", true}, {"summary", summaries}, {"games", games}
        };
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Gameplay benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
