// End-to-end one-ply expansion benchmark using the experiment's actual move
// generator and native evaluator. No continuation/teacher labels are computed.
// Build: g++ -O3 -DNDEBUG -std=c++17 -march=native search_bench.cpp -o search_bench
// Verify without timing: ./search_bench --data records.bin --model model.snn --verify-only
#define SPATIAL_DATA_LIBRARY
#include "generate.cpp"
#define SPATIAL_KERNEL_LIBRARY
#include "kernels.cpp"

namespace search_bench {

struct Options {
    std::string data = "experiments/spatial-nnue/data/heldout/records.bin";
    std::string model, output;
    int repeats = 7;
    size_t limit_roots = 0;
    bool verify_only = false;
};

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "search_bench --model model.snn [--data records.bin] "
                         "[--output results.json] [--repeats 7] [--limit-roots N] "
                         "[--verify-only]\n";
            std::exit(0);
        }
        if (arg == "--verify-only") { o.verify_only = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("Missing argument for " + arg);
        const std::string value = argv[++i];
        if (arg == "--data") o.data = value;
        else if (arg == "--model") o.model = value;
        else if (arg == "--output") o.output = value;
        else if (arg == "--repeats") o.repeats = std::stoi(value);
        else if (arg == "--limit-roots") o.limit_roots = std::stoull(value);
        else throw std::runtime_error("Unknown option " + arg);
    }
    if (o.model.empty() || o.repeats < 1)
        throw std::runtime_error("A model path and positive repeat count are required");
    return o;
}

struct Root {
    size_t representative;
    std::vector<size_t> recorded_children;
};

std::vector<Root> collect_roots(const std::vector<spatial::Record> &records, size_t limit) {
    std::vector<Root> roots;
    std::unordered_map<uint64_t, size_t> by_id;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto &r = records[i];
        if (r.event != 0) continue;
        const uint64_t key = (uint64_t(r.game_id) << 32) | r.parent_id;
        const auto found = by_id.find(key);
        if (found == by_id.end()) {
            by_id.emplace(key, roots.size());
            roots.push_back({i, {i}});
        } else {
            auto &root = roots[found->second];
            const auto &first = records[root.representative];
            if (r.piece != first.piece || r.next != first.next ||
                r.step != first.step ||
                std::memcmp(r.parent_rows, first.parent_rows, sizeof(r.parent_rows)))
                throw std::runtime_error("One root ID has inconsistent board or piece metadata");
            root.recorded_children.push_back(i);
        }
    }
    if (limit && roots.size() > limit) roots.resize(limit);
    if (roots.empty()) throw std::runtime_error("No lock roots in dataset");
    return roots;
}

Board parent_board(const spatial::Record &r) {
    Board board;
    std::copy(std::begin(r.parent_rows), std::end(r.parent_rows), board.begin());
    return board;
}

// Construct the evaluator's transition record from a freshly generated move.
// This work is included in every timed method, including the full-refresh path.
spatial::Record transition_record(const spatial::Record &root, const Candidate &move) {
    spatial::Record r{};
    r.game_id = root.game_id;
    r.parent_id = root.parent_id;
    r.step = root.step;
    r.event = 0;
    r.piece = root.piece;
    r.next = root.next;
    r.clears = uint8_t(move.clears);
    r.clear_mask = move.clear_mask;
    r.reward = move.reward;
    std::copy(std::begin(root.parent_rows), std::end(root.parent_rows), r.parent_rows);
    std::copy(move.child.begin(), move.child.end(), r.child_rows);
    std::copy(move.source.begin(), move.source.end(), r.row_source);
    std::copy(move.cells.begin(), move.cells.end(), r.placed_cells);
    return r;
}

constexpr double ACTION_TIE_ABS = 1e-4, ACTION_TIE_REL = 1e-6;
struct Verification {
    size_t roots = 0, candidates = 0;
    std::array<spatial::Parity, 3> parity;
    std::array<size_t, 3> identical_choices{}, numeric_tie_choices{}, non_tie_mismatches{};
    std::array<double, 3> max_reference_choice_gap{};
    bool passed = true;
};

Verification verify(const spatial::Model &model, const std::vector<spatial::Record> &records,
                    const std::vector<Root> &roots) {
    Verification result;
    spatial::Accumulator parent, full, changed;
    for (const auto &root : roots) {
        const auto &source = records[root.representative];
        if (source.piece < 0 || source.piece >= 7 || source.next >= 7)
            throw std::runtime_error("Lock root requires current/next piece IDs 0..6");
        const Board board = parent_board(source);
        const auto moves = placements(board, source.piece);
        if (moves.empty() || moves.size() != root.recorded_children.size())
            throw std::runtime_error("Regenerated candidate count differs from dataset");
        spatial::refresh(model, source.parent_rows, parent);
        std::array<std::vector<float>, 3> scores;
        for (auto &v : scores) v.reserve(moves.size());
        for (size_t j = 0; j < moves.size(); ++j) {
            const auto &expected = records[root.recorded_children[j]];
            const auto r = transition_record(source, moves[j]);
            if (r.clears != expected.clears || r.clear_mask != expected.clear_mask ||
                r.reward != expected.reward ||
                std::memcmp(r.child_rows, expected.child_rows, sizeof(r.child_rows)) ||
                std::memcmp(r.row_source, expected.row_source, sizeof(r.row_source)) ||
                std::memcmp(r.placed_cells, expected.placed_cells, sizeof(r.placed_cells))) {
                throw std::runtime_error("Regenerated candidate/order differs at game " +
                    std::to_string(source.game_id) + " parent " +
                    std::to_string(source.parent_id) + " candidate " + std::to_string(j));
            }
            spatial::refresh(model, r.child_rows, full);
            const float reference = spatial::forward(model, full, r.next);
            if (!std::isfinite(reference)) throw std::runtime_error("Non-finite reference score");
            scores[0].push_back(r.reward + reference);
            for (int id = 1; id < 3; ++id) {
                spatial::transition(model, r, parent, changed, static_cast<spatial::Method>(id));
                auto &p = result.parity[id];
                spatial::compare_acc(model, full, changed, p);
                const float value = spatial::forward(model, changed, r.next);
                const double error = std::abs(double(reference) - value);
                p.max_score_abs = std::max(p.max_score_abs, error);
                p.sum_score_abs += error;
                if (!std::isfinite(value) || error > 1e-3 + 1e-5 * std::abs(double(reference)))
                    ++p.failed_scores;
                scores[id].push_back(r.reward + value);
            }
            ++result.candidates;
        }
        const size_t reference_choice = size_t(std::max_element(scores[0].begin(), scores[0].end()) - scores[0].begin());
        for (int id = 1; id < 3; ++id) {
            const size_t choice = size_t(std::max_element(scores[id].begin(), scores[id].end()) - scores[id].begin());
            const double gap = double(scores[0][reference_choice]) - scores[0][choice];
            result.max_reference_choice_gap[id] = std::max(result.max_reference_choice_gap[id], gap);
            const double tolerance = ACTION_TIE_ABS + ACTION_TIE_REL *
                std::max(std::abs(double(scores[0][reference_choice])), std::abs(double(scores[0][choice])));
            if (choice == reference_choice) ++result.identical_choices[id];
            else if (gap <= tolerance) ++result.numeric_tie_choices[id];
            else ++result.non_tie_mismatches[id];
        }
        ++result.roots;
    }
    for (int id = 1; id < 3; ++id)
        result.passed = result.passed && !result.parity[id].failed_acc_components &&
                        !result.parity[id].failed_scores && !result.non_tie_mismatches[id];
    return result;
}

struct Pass {
    size_t candidates = 0;
    double best_q_sum = 0;
    uint64_t selection_checksum = 14695981039346656037ULL;
};

Pass expand_all(const spatial::Model &model, const std::vector<spatial::Record> &records,
                const std::vector<Root> &roots, spatial::Method method) {
    Pass pass;
    spatial::Accumulator parent, child;
    for (const auto &root : roots) {
        const auto &source = records[root.representative];
        // Board reconstruction, generation/allocation, locks and row compaction,
        // transition metadata, evaluator, reward addition and argmax are timed.
        const Board board = parent_board(source);
        if (method != spatial::Method::Full) spatial::refresh(model, board.data(), parent);
        const auto moves = placements(board, source.piece);
        float best_q = -std::numeric_limits<float>::infinity();
        size_t best_choice = 0;
        for (size_t j = 0; j < moves.size(); ++j) {
            const auto r = transition_record(source, moves[j]);
            spatial::transition(model, r, parent, child, method);
            const float q = r.reward + spatial::forward(model, child, r.next);
            if (q > best_q) { best_q = q; best_choice = j; }
        }
        pass.candidates += moves.size();
        pass.best_q_sum += best_q;
        const uint64_t key = (uint64_t(source.game_id) << 32) | source.parent_id;
        pass.selection_checksum = (pass.selection_checksum ^ key) * 1099511628211ULL;
        pass.selection_checksum = (pass.selection_checksum ^ best_choice) * 1099511628211ULL;
    }
    return pass;
}

volatile uint64_t choice_sink = 0;
struct Timing {
    std::vector<double> total_ns;
    Pass pass;
};

std::array<Timing, 3> benchmark(const spatial::Model &model,
                              const std::vector<spatial::Record> &records,
                              const std::vector<Root> &roots, int repeats) {
    std::array<Timing, 3> timings;
    // One untimed complete warm-up per method. Rotate method order each repeat
    // to avoid always placing the same method first or last in the run.
    for (int id = 0; id < 3; ++id) {
        const auto pass = expand_all(model, records, roots, static_cast<spatial::Method>(id));
        spatial::benchmark_sink = spatial::benchmark_sink + pass.best_q_sum;
        choice_sink = choice_sink ^ pass.selection_checksum;
    }
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int slot = 0; slot < 3; ++slot) {
            const int id = (repeat + slot) % 3;
            const auto start = spatial::Clock::now();
            const auto pass = expand_all(model, records, roots, static_cast<spatial::Method>(id));
            const auto end = spatial::Clock::now();
            auto &timing = timings[id];
            if (!timing.total_ns.empty() && (pass.candidates != timing.pass.candidates ||
                pass.best_q_sum != timing.pass.best_q_sum ||
                pass.selection_checksum != timing.pass.selection_checksum))
                throw std::runtime_error("Expansion changed between identical benchmark repetitions");
            timing.pass = pass;
            timing.total_ns.push_back(std::chrono::duration<double, std::nano>(end - start).count());
            spatial::benchmark_sink = spatial::benchmark_sink + pass.best_q_sum;
            choice_sink = choice_sink ^ pass.selection_checksum;
        }
    }
    return timings;
}

void json_string(std::ostream &out, const std::string &s) {
    const char hex[] = "0123456789abcdef";
    out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << char(c);
    }
    out << '"';
}

void report(std::ostream &out, const Options &options, const spatial::Model &model,
            size_t loaded_records, const Verification &v, const std::array<Timing, 3> &timings) {
    out << std::setprecision(12)
        << "{\n\"schema\":\"spatial-nnue-one-ply-v1\",\n\"dataset\":";
    json_string(out, options.data);
    out << ",\n\"model\":"; json_string(out, options.model);
    out << ",\n\"compiler\":"; json_string(out, __VERSION__);
    out << ",\n\"record_bytes\":240,\"width\":10,\"total_height\":40,\"loaded_records\":" << loaded_records
        << ",\"root_count\":" << v.roots << ",\"generated_candidates\":" << v.candidates
        << ",\"mean_candidates_per_root\":" << double(v.candidates) / v.roots
        << ",\"model_kind\":" << model.kind << ",\"model_bytes\":" << model.serialized_bytes
        << ",\"verify_only\":" << (options.verify_only ? "true" : "false")
        << ",\n\"scope\":\"One-ply hard-drop expansion over heldout parent states: move generation, lock and clear board transitions, evaluator transition and head, current reward and argmax. No teacher continuation labels, deeper search, hold, spins, attack rules or playing-strength claim.\",\n"
        << "\"verification\":{\"passed\":" << (v.passed ? "true" : "false")
        << ",\"regenerated_candidates_match_dataset\":true,\"candidate_order_matches\":true,"
           "\"action_tie_tolerance\":{\"absolute\":" << ACTION_TIE_ABS << ",\"relative\":" << ACTION_TIE_REL << "}";
    for (int id = 1; id < 3; ++id) {
        const auto &p = v.parity[id];
        out << ','; json_string(out, spatial::method_name(static_cast<spatial::Method>(id)));
        out << ":{\"max_acc_abs\":" << p.max_acc_abs
            << ",\"failed_acc_components\":" << p.failed_acc_components
            << ",\"max_evaluator_abs\":" << p.max_score_abs
            << ",\"mean_evaluator_abs\":" << p.sum_score_abs / v.candidates
            << ",\"failed_scores\":" << p.failed_scores
            << ",\"identical_choices\":" << v.identical_choices[id]
            << ",\"different_choices_within_numeric_tie\":" << v.numeric_tie_choices[id]
            << ",\"non_tie_choice_mismatches\":" << v.non_tie_mismatches[id]
            << ",\"max_reference_choice_gap\":" << v.max_reference_choice_gap[id] << '}';
    }
    out << "},\n\"timings\":{";
    if (!options.verify_only && v.passed) {
        for (int id = 0; id < 3; ++id) {
            if (id) out << ',';
            json_string(out, spatial::method_name(static_cast<spatial::Method>(id)));
            const auto &t = timings[id];
            const double median = spatial::median(t.total_ns);
            const double minimum = *std::min_element(t.total_ns.begin(), t.total_ns.end());
            const double maximum = *std::max_element(t.total_ns.begin(), t.total_ns.end());
            out << ":{\"repeats\":" << t.total_ns.size()
                << ",\"generated_candidates_per_repeat\":" << t.pass.candidates
                << ",\"median_ns_per_candidate\":" << median / t.pass.candidates
                << ",\"min_ns_per_candidate\":" << minimum / t.pass.candidates
                << ",\"max_ns_per_candidate\":" << maximum / t.pass.candidates
                << ",\"median_us_per_root\":" << median / (1000.0 * v.roots)
                << ",\"median_candidates_per_second\":" << 1e9 * t.pass.candidates / median
                << ",\"best_q_sum\":" << t.pass.best_q_sum << ",\"selection_checksum\":";
            json_string(out, std::to_string(t.pass.selection_checksum));
            out << ",\"total_ns_samples\": [";
            for (size_t j = 0; j < t.total_ns.size(); ++j) { if (j) out << ','; out << t.total_ns[j]; }
            out << "]}";
        }
    }
    out << "}\n}\n";
}

int run(const Options &options) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
        throw std::runtime_error("This dataset/model schema requires a little-endian host");
    ensure_shapes();
    const auto model = spatial::Model::load(options.model);
    const auto records = spatial::load_records(options.data, 0);
    const auto roots = collect_roots(records, options.limit_roots);
    const auto verification = verify(model, records, roots);
    std::array<Timing, 3> timings;
    if (verification.passed && !options.verify_only)
        timings = benchmark(model, records, roots, options.repeats);
    if (options.output.empty()) report(std::cout, options, model, records.size(), verification, timings);
    else {
        std::ofstream file(options.output);
        if (!file) throw std::runtime_error("Cannot write output: " + options.output);
        report(file, options, model, records.size(), verification, timings);
        if (!file) throw std::runtime_error("Failed writing output: " + options.output);
        std::cout << "Wrote " << options.output << " for " << verification.roots << " roots and "
                  << verification.candidates << " regenerated candidates\n";
    }
    return verification.passed ? 0 : 2;
}
} // namespace search_bench

int main(int argc, char **argv) {
    try { return search_bench::run(search_bench::parse(argc, argv)); }
    catch (const std::exception &e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
