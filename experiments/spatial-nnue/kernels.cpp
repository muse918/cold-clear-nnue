// Reproducible, dependency-free inference and transition benchmark.
// Build: g++ -O3 -DNDEBUG -std=c++17 -march=native kernels.cpp -o kernels
// The model is trained/exported separately. This program never trains weights.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace spatial {
constexpr int W = 10, H = 40, PX = 8, PY = 37;
constexpr int MAX_POS = 128, MAX_LOCAL = 144, MAX_FEATURES = 167;

#pragma pack(push, 1)
struct Record {
    uint32_t game_id, parent_id;
    uint16_t step;
    uint8_t event;
    int8_t piece;
    uint8_t next, clears, garbage, selected;
    uint64_t clear_mask;
    float reward, value_target;
    uint16_t parent_rows[H], child_rows[H];
    int8_t row_source[H];
    uint16_t placed_cells[4];
};
#pragma pack(pop)
static_assert(sizeof(Record) == 240, "Dataset ABI must be 240 bytes");

template <class T> void read_exact(std::istream &in, T *ptr, size_t n) {
    if (n && !in.read(reinterpret_cast<char *>(ptr), static_cast<std::streamsize>(n * sizeof(T))))
        throw std::runtime_error("Unexpected end of input");
}
template <class T> T read_one(std::istream &in) {
    T result;
    read_exact(in, &result, 1);
    return result;
}

struct Model {
    uint32_t kind = 0, localdim = 0, posdim = 0, featdim = 0, headdim = 0;
    float output_mean = 0, output_std = 1;
    // lut contains residuals relative to the empty token. Empty-token mass is
    // retained in local_zero; in particular attention denominators are NOT masked.
    std::vector<float> lut, posw, posb, headw, headb, outw;
    std::array<float, 18> local_zero{};
    float outb = 0;
    size_t serialized_bytes = 0;

    static Model load(const std::string &path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open model: " + path);
        char magic[16];
        read_exact(in, magic, 16);
        if (std::memcmp(magic, "SPATIALNNUEv1", 13))
            throw std::runtime_error("Unsupported model magic");
        Model m;
        m.kind = read_one<uint32_t>(in);
        m.localdim = read_one<uint32_t>(in);
        m.posdim = read_one<uint32_t>(in);
        m.featdim = read_one<uint32_t>(in);
        m.headdim = read_one<uint32_t>(in);
        m.output_mean = read_one<float>(in);
        m.output_std = read_one<float>(in);
        if (m.kind > 2 || m.headdim != 32 ||
            (m.kind == 0 && (m.localdim != 0 || m.posdim != 128 || m.featdim != 135)) ||
            (m.kind == 1 && (m.localdim != 16 || m.posdim != 32 || m.featdim != 167)) ||
            (m.kind == 2 && (m.localdim != 18 || m.posdim != 32 || m.featdim != 167)))
            throw std::runtime_error("Model dimensions disagree with v1 schema");
        auto readv = [&in](std::vector<float> &v, size_t size) {
            v.resize(size);
            read_exact(in, v.data(), size);
            for (float x : v) if (!std::isfinite(x)) throw std::runtime_error("Non-finite weight");
        };
        readv(m.lut, size_t(4096) * m.localdim);
        readv(m.posw, size_t(400) * m.posdim);
        readv(m.posb, m.posdim);
        readv(m.headw, size_t(m.featdim) * m.headdim);
        readv(m.headb, m.headdim);
        readv(m.outw, m.headdim);
        m.outb = read_one<float>(in);
        m.serialized_bytes = static_cast<size_t>(in.tellg());
        if (in.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Unexpected trailing model data");
        for (uint32_t j = 0; j < m.localdim; ++j) m.local_zero[j] = m.lut[j];
        for (int p = 0; p < 4096; ++p)
            for (uint32_t j = 0; j < m.localdim; ++j)
                m.lut[size_t(p) * m.localdim + j] -= m.local_zero[j];
        return m;
    }
};

struct alignas(64) Accumulator {
    std::array<float, MAX_POS> pos{};
    std::array<float, MAX_LOCAL> local{};
};

inline unsigned pattern(const uint16_t *rows, int x, int y) {
    return ((rows[y] >> x) & 7u)
        | (((rows[y + 1] >> x) & 7u) << 3)
        | (((rows[y + 2] >> x) & 7u) << 6)
        | (((rows[y + 3] >> x) & 7u) << 9);
}

inline void add_local(const Model &m, Accumulator &a, int x, unsigned p, float sign) {
    if (!p) return;
    float *dst = a.local.data() + size_t(x) * m.localdim;
    const float *src = m.lut.data() + size_t(p) * m.localdim;
    for (uint32_t j = 0; j < m.localdim; ++j) dst[j] += sign * src[j];
}

inline void replace_local(const Model &m, Accumulator &a, int x, unsigned oldp, unsigned newp) {
    if (oldp == newp) return;
    float *dst = a.local.data() + size_t(x) * m.localdim;
    const float *oldv = m.lut.data() + size_t(oldp) * m.localdim;
    const float *newv = m.lut.data() + size_t(newp) * m.localdim;
    for (uint32_t j = 0; j < m.localdim; ++j) dst[j] += newv[j] - oldv[j];
}

inline void add_cell(const Model &m, Accumulator &a, unsigned cell, float sign) {
    const float *src = m.posw.data() + size_t(cell) * m.posdim;
    for (uint32_t j = 0; j < m.posdim; ++j) a.pos[j] += sign * src[j];
}

void refresh(const Model &m, const uint16_t *rows, Accumulator &a) {
    std::copy(m.posb.begin(), m.posb.end(), a.pos.begin());
    for (int y = 0; y < H; ++y) {
        unsigned mask = rows[y] & 1023u;
        while (mask) {
            unsigned x = unsigned(__builtin_ctz(mask));
            mask &= mask - 1;
            add_cell(m, a, y * W + x, 1.0f);
        }
    }
    if (!m.kind) return;
    for (int x = 0; x < PX; ++x)
        for (uint32_t j = 0; j < m.localdim; ++j)
            a.local[size_t(x) * m.localdim + j] = PY * m.local_zero[j];
    for (int y = 0; y < PY; ++y)
        for (int x = 0; x < PX; ++x)
            add_local(m, a, x, pattern(rows, x, y), 1.0f);
}

inline bool ordinary_lock(const Record &r) {
    return r.event == 0 && r.clears == 0 && r.garbage == 0;
}

void copy_acc(const Model &m, const Accumulator &parent, Accumulator &child) {
    std::copy_n(parent.pos.data(), m.posdim, child.pos.data());
    if (m.kind) std::copy_n(parent.local.data(), PX * m.localdim, child.local.data());
}

void update_pos(const Model &m, const Record &r, Accumulator &a) {
    for (int y = 0; y < H; ++y) {
        unsigned removed = r.parent_rows[y] & ~r.child_rows[y] & 1023u;
        unsigned added = r.child_rows[y] & ~r.parent_rows[y] & 1023u;
        while (removed) {
            unsigned x = unsigned(__builtin_ctz(removed));
            removed &= removed - 1;
            add_cell(m, a, y * W + x, -1.0f);
        }
        while (added) {
            unsigned x = unsigned(__builtin_ctz(added));
            added &= added - 1;
            add_cell(m, a, y * W + x, 1.0f);
        }
    }
}

void dirty_lock_anchors(const Record &r, std::array<uint64_t, PX> &dirty) {
    dirty.fill(0);
    // Four placements are supplied in the pre-clear coordinate system. They
    // are directly usable only on the ordinary-lock fast path.
    bool supplied = true;
    for (unsigned cell : r.placed_cells) supplied = supplied && cell < W * H;
    auto mark = [&dirty](int cx, int cy) {
        int lo = std::max(0, cy - 3), hi = std::min(PY - 1, cy);
        if (lo > hi) return;
        uint64_t ymask = ((uint64_t(1) << (hi - lo + 1)) - 1) << lo;
        for (int x = std::max(0, cx - 2); x <= std::min(PX - 1, cx); ++x)
            dirty[x] |= ymask;
    };
    if (supplied) {
        for (unsigned cell : r.placed_cells) mark(cell % W, cell / W);
    } else {
        for (int y = 0; y < H; ++y) {
            unsigned mask = (r.parent_rows[y] ^ r.child_rows[y]) & 1023u;
            while (mask) {
                unsigned x = unsigned(__builtin_ctz(mask));
                mask &= mask - 1;
                mark(int(x), y);
            }
        }
    }
}

void update_local_lock(const Model &m, const Record &r, Accumulator &a) {
    std::array<uint64_t, PX> dirty;
    dirty_lock_anchors(r, dirty);
    for (int x = 0; x < PX; ++x) {
        uint64_t ys = dirty[x];
        while (ys) {
            int y = __builtin_ctzll(ys);
            ys &= ys - 1;
            replace_local(m, a, x, pattern(r.parent_rows, x, y), pattern(r.child_rows, x, y));
        }
    }
}

// Marks intact four-row strips. A strip can be reused only when all four
// source rows are consecutive AND their complete row masks are unchanged.
// This is conservative: a changed bit outside a particular x-window can make
// that strip dirty, but cannot cause an incorrect reuse.
void reusable_strips(const Record &r, uint64_t &old_keep, uint64_t &new_keep) {
    old_keep = new_keep = 0;
    for (int y = 0; y < PY; ++y) {
        int oldy = r.row_source[y];
        if (oldy < 0 || oldy >= PY) continue;
        bool keep = true;
        for (int dy = 0; dy < 4; ++dy)
            if (r.row_source[y + dy] != oldy + dy ||
                r.child_rows[y + dy] != r.parent_rows[oldy + dy]) keep = false;
        if (keep) {
            old_keep |= uint64_t(1) << oldy;
            new_keep |= uint64_t(1) << y;
        }
    }
}

void update_local_seams(const Model &m, const Record &r, Accumulator &a) {
    uint64_t old_keep, new_keep;
    reusable_strips(r, old_keep, new_keep);
    const uint64_t valid = (uint64_t(1) << PY) - 1;
    uint64_t old_dirty = valid & ~old_keep, new_dirty = valid & ~new_keep;
    while (old_dirty) {
        int y = __builtin_ctzll(old_dirty);
        old_dirty &= old_dirty - 1;
        for (int x = 0; x < PX; ++x) add_local(m, a, x, pattern(r.parent_rows, x, y), -1.0f);
    }
    while (new_dirty) {
        int y = __builtin_ctzll(new_dirty);
        new_dirty &= new_dirty - 1;
        for (int x = 0; x < PX; ++x) add_local(m, a, x, pattern(r.child_rows, x, y), 1.0f);
    }
}

enum class Method { Full = 0, RefreshEvents = 1, Seams = 2 };
const char *method_name(Method method) {
    switch (method) {
        case Method::Full: return "full_refresh";
        case Method::RefreshEvents: return "incremental_refresh_events";
        case Method::Seams: return "incremental_row_seams";
    }
    return "unknown";
}

void transition(const Model &m, const Record &r, const Accumulator &parent,
                Accumulator &child, Method method) {
    if (method == Method::Full || (method == Method::RefreshEvents && !ordinary_lock(r))) {
        refresh(m, r.child_rows, child);
        return;
    }
    copy_acc(m, parent, child);
    update_pos(m, r, child);
    if (!m.kind) return;
    if (ordinary_lock(r)) update_local_lock(m, r, child);
    else update_local_seams(m, r, child);
}

float forward(const Model &m, const Accumulator &a, unsigned next) {
    std::array<float, MAX_FEATURES> feature{};
    unsigned cursor = 0;
    if (m.kind == 1) {
        for (int j = 0; j < PX * 16; ++j) feature[cursor++] = a.local[j] / float(PY);
    } else if (m.kind == 2) {
        for (int x = 0; x < PX; ++x) {
            for (int h = 0; h < 2; ++h) {
                const float *pool = a.local.data() + x * 18 + h * 9;
                float denom = pool[8];
                if (!(denom > 0) || !std::isfinite(denom))
                    throw std::runtime_error("Attention denominator must remain finite and positive");
                float inv = 1.0f / denom;
                for (int j = 0; j < 8; ++j) feature[cursor++] = pool[j] * inv;
            }
        }
    }
    for (unsigned j = 0; j < m.posdim; ++j) feature[cursor++] = std::max(a.pos[j], 0.0f);
    if (next < 7) feature[cursor + next] = 1.0f;
    cursor += 7;
    if (cursor != m.featdim) throw std::runtime_error("Feature layout mismatch");
    std::array<float, 32> hidden;
    std::copy(m.headb.begin(), m.headb.end(), hidden.begin());
    for (unsigned i = 0; i < m.featdim; ++i) {
        const float *weight = m.headw.data() + size_t(i) * 32;
        float f = feature[i];
        for (int j = 0; j < 32; ++j) hidden[j] += f * weight[j];
    }
    float value = m.outb;
    for (int j = 0; j < 32; ++j) value += std::max(hidden[j], 0.0f) * m.outw[j];
    return value * m.output_std + m.output_mean;
}

std::vector<Record> load_records(const std::string &path, size_t limit) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open dataset: " + path);
    size_t bytes = static_cast<size_t>(in.tellg());
    if (bytes % sizeof(Record)) throw std::runtime_error("Dataset length is not a multiple of 240");
    size_t count = bytes / sizeof(Record);
    if (limit) count = std::min(count, limit);
    std::vector<Record> records(count);
    in.seekg(0);
    read_exact(in, records.data(), count);
    for (const auto &r : records) {
        if (r.event > 1 || r.next >= 7 || r.clears > 4)
            throw std::runtime_error("Invalid dataset enum/count");
        int last_source = -1;
        for (int y = 0; y < H; ++y) {
            if ((r.parent_rows[y] | r.child_rows[y]) & ~1023u)
                throw std::runtime_error("Board has bits beyond ten columns");
            if (r.row_source[y] < -1 || r.row_source[y] >= H)
                throw std::runtime_error("Invalid row mapping");
            if (r.row_source[y] >= 0) {
                if (r.row_source[y] <= last_source)
                    throw std::runtime_error("Row mapping is not injective/order preserving");
                last_source = r.row_source[y];
            }
        }
    }
    return records;
}

struct Group {
    size_t representative = 0;
    std::vector<size_t> children;
    Accumulator acc;
};
struct Parents {
    std::vector<Group> groups;
    std::vector<size_t> group_of_record;
};
Parents prepare_parents(const Model &m, const std::vector<Record> &records) {
    Parents result;
    result.group_of_record.resize(records.size());
    std::unordered_map<uint64_t, size_t> ids;
    for (size_t i = 0; i < records.size(); ++i) {
        const auto &r = records[i];
        uint64_t key = (uint64_t(r.game_id) << 32) | r.parent_id;
        auto found = ids.find(key);
        size_t group_id;
        if (found == ids.end()) {
            group_id = result.groups.size();
            ids.emplace(key, group_id);
            result.groups.emplace_back();
            auto &g = result.groups.back();
            g.representative = i;
            refresh(m, r.parent_rows, g.acc);
        } else {
            group_id = found->second;
            const Record &original = records[result.groups[group_id].representative];
            if (std::memcmp(original.parent_rows, r.parent_rows, sizeof(r.parent_rows)))
                throw std::runtime_error("Same parent ID has different boards");
        }
        result.group_of_record[i] = group_id;
        result.groups[group_id].children.push_back(i);
    }
    return result;
}

struct Parity {
    double max_acc_abs = 0, max_score_abs = 0, sum_score_abs = 0;
    size_t failed_acc_components = 0, failed_scores = 0;
};

struct EncoderParity {
    bool enabled = false;
    size_t patterns = 0, components = 0, failed_components = 0;
    double max_abs = 0;
};

// Independent scalar execution of the train-time local MLP, using source
// weights rather than reading the compiled LUT to produce the reference.
// Source file is raw input-major float32, in this exact order:
// W1[12,16], b1[16], W2[16,16], b2[16]; for attention additionally
// logitsW[16,2], logitsB[2], valuesW[16,16], valuesB[16].
EncoderParity verify_encoder(const Model &m, const std::string &path) {
    EncoderParity report;
    if (path.empty()) return report;
    if (!m.kind) throw std::runtime_error("Cell model has no local encoder to verify");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open encoder source: " + path);
    std::array<float, 12 * 16> w1;
    std::array<float, 16> b1, b2, valueb{};
    std::array<float, 16 * 16> w2, valuew{};
    std::array<float, 16 * 2> logitw{};
    std::array<float, 2> logitb{};
    read_exact(in, w1.data(), w1.size());
    read_exact(in, b1.data(), b1.size());
    read_exact(in, w2.data(), w2.size());
    read_exact(in, b2.data(), b2.size());
    if (m.kind == 2) {
        read_exact(in, logitw.data(), logitw.size());
        read_exact(in, logitb.data(), logitb.size());
        read_exact(in, valuew.data(), valuew.size());
        read_exact(in, valueb.data(), valueb.size());
    }
    if (in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected trailing encoder source data");
    report.enabled = true;
    for (unsigned p = 0; p < 4096; ++p) {
        std::array<float, 16> layer1 = b1, layer2 = b2;
        for (int i = 0; i < 12; ++i) {
            float bit = float((p >> i) & 1u);
            for (int j = 0; j < 16; ++j) layer1[j] += bit * w1[i * 16 + j];
        }
        for (float &x : layer1) x = std::max(x, 0.0f);
        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) layer2[j] += layer1[i] * w2[i * 16 + j];
        for (float &x : layer2) x = std::max(x, 0.0f);
        std::array<float, 18> reference{};
        if (m.kind == 1) std::copy(layer2.begin(), layer2.end(), reference.begin());
        else {
            std::array<float, 2> logits = logitb;
            std::array<float, 16> values = valueb;
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 2; ++j) logits[j] += layer2[i] * logitw[i * 2 + j];
                for (int j = 0; j < 16; ++j) values[j] += layer2[i] * valuew[i * 16 + j];
            }
            for (int head = 0; head < 2; ++head) {
                float weight = std::exp(std::max(-5.0f, std::min(logits[head], 5.0f)));
                for (int j = 0; j < 8; ++j)
                    reference[head * 9 + j] = weight * std::tanh(values[head * 8 + j]);
                reference[head * 9 + 8] = weight;
            }
        }
        ++report.patterns;
        for (unsigned j = 0; j < m.localdim; ++j) {
            float compiled = m.lut[size_t(p) * m.localdim + j] + m.local_zero[j];
            double err = std::abs(double(reference[j]) - compiled);
            report.max_abs = std::max(report.max_abs, err);
            ++report.components;
            if (!std::isfinite(compiled) || err > 2e-5 + 3e-5 * std::abs(double(reference[j])))
                ++report.failed_components;
        }
    }
    return report;
}

void compare_acc(const Model &m, const Accumulator &ref, const Accumulator &got, Parity &p) {
    auto check = [&p](float expected, float actual) {
        double err = std::abs(double(expected) - actual);
        p.max_acc_abs = std::max(p.max_acc_abs, err);
        if (!std::isfinite(actual) || err > 1e-4 + 3e-5 * std::abs(double(expected)))
            ++p.failed_acc_components;
    };
    for (unsigned j = 0; j < m.posdim; ++j) check(ref.pos[j], got.pos[j]);
    for (unsigned j = 0; j < PX * m.localdim; ++j) check(ref.local[j], got.local[j]);
}

struct ChainedParity {
    size_t transitions = 0, segments = 0, discontinuities = 0, max_segment_length = 0;
    std::array<Parity, 3> methods;
};

// Carry the actual updated accumulator along selected game trajectories.
// This catches cumulative floating-point drift that fresh-parent edge checks
// cannot observe. Garbage-only transitions are forced edges and also included.
ChainedParity verify_chains(const Model &m, const std::vector<Record> &records) {
    ChainedParity report;
    bool have_previous = false;
    uint32_t previous_game = 0;
    size_t segment_length = 0;
    std::array<uint16_t, H> previous_rows{};
    std::array<Accumulator, 3> carried;
    Accumulator reference, updated;
    for (const Record &r : records) {
        if (!r.selected && r.event != 1) continue;
        bool same_game = have_previous && r.game_id == previous_game;
        bool contiguous = same_game &&
            std::equal(previous_rows.begin(), previous_rows.end(), r.parent_rows);
        if (!contiguous) {
            if (same_game) ++report.discontinuities;
            ++report.segments;
            segment_length = 0;
            for (int method = 1; method < 3; ++method) refresh(m, r.parent_rows, carried[method]);
        }
        ++report.transitions;
        ++segment_length;
        report.max_segment_length = std::max(report.max_segment_length, segment_length);
        refresh(m, r.child_rows, reference);
        float reference_score = forward(m, reference, r.next);
        for (int method = 1; method < 3; ++method) {
            transition(m, r, carried[method], updated, static_cast<Method>(method));
            auto &p = report.methods[method];
            compare_acc(m, reference, updated, p);
            double err = std::abs(double(reference_score) - forward(m, updated, r.next));
            p.max_score_abs = std::max(p.max_score_abs, err);
            p.sum_score_abs += err;
            if (!std::isfinite(err) || err > 1e-3 + 1e-5 * std::abs(double(reference_score)))
                ++p.failed_scores;
            copy_acc(m, updated, carried[method]);
        }
        std::copy_n(r.child_rows, H, previous_rows.begin());
        previous_game = r.game_id;
        have_previous = true;
    }
    return report;
}

volatile double benchmark_sink = 0;
using Clock = std::chrono::steady_clock;
double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

struct Timing {
    double median_ns = 0, min_ns = 0, max_ns = 0, checksum = 0;
    size_t count = 0;
};

Timing benchmark_subset(const Model &m, const std::vector<Record> &records,
                        const Parents &parents, const std::vector<size_t> &indices,
                        Method method, int repeats) {
    Timing out;
    out.count = indices.size();
    if (indices.empty()) return out;
    Accumulator child;
    for (size_t k = 0; k < std::min<size_t>(indices.size(), 128); ++k) {
        size_t i = indices[k];
        transition(m, records[i], parents.groups[parents.group_of_record[i]].acc, child, method);
        benchmark_sink = benchmark_sink + forward(m, child, records[i].next);
    }
    std::vector<double> timings;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        double checksum = 0;
        auto start = Clock::now();
        for (size_t i : indices) {
            transition(m, records[i], parents.groups[parents.group_of_record[i]].acc, child, method);
            checksum += forward(m, child, records[i].next);
        }
        auto stop = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(stop - start).count() / indices.size();
        timings.push_back(ns);
        out.checksum = checksum;
        benchmark_sink = benchmark_sink + checksum;
    }
    out.median_ns = median(timings);
    out.min_ns = *std::min_element(timings.begin(), timings.end());
    out.max_ns = *std::max_element(timings.begin(), timings.end());
    return out;
}

Timing benchmark_siblings(const Model &m, const std::vector<Record> &records,
                          const Parents &parents, Method method, int repeats) {
    Timing out;
    out.count = records.size();
    if (records.empty()) return out;
    Accumulator parent, child;
    std::vector<double> timings;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        double checksum = 0;
        auto start = Clock::now();
        for (const auto &group : parents.groups) {
            // Full child inference does not require an otherwise unused parent.
            if (method != Method::Full) refresh(m, records[group.representative].parent_rows, parent);
            for (size_t i : group.children) {
                transition(m, records[i], parent, child, method);
                checksum += forward(m, child, records[i].next);
            }
        }
        auto stop = Clock::now();
        timings.push_back(std::chrono::duration<double, std::nano>(stop - start).count() / records.size());
        out.checksum = checksum;
        benchmark_sink = benchmark_sink + checksum;
    }
    out.median_ns = median(timings);
    out.min_ns = *std::min_element(timings.begin(), timings.end());
    out.max_ns = *std::max_element(timings.begin(), timings.end());
    return out;
}

std::string event_name(const Record &r) {
    if (r.event == 1 || r.garbage) return "garbage";
    if (r.clears) return "clear_" + std::to_string(r.clears);
    return "no_clear";
}
void print_timing(std::ostream &out, const Timing &t) {
    out << "{\"count\":" << t.count << ",\"median_ns\":" << t.median_ns
        << ",\"min_ns\":" << t.min_ns << ",\"max_ns\":" << t.max_ns
        << ",\"checksum\":" << t.checksum << "}";
}

struct Options {
    std::string data, model, output, predictions, encoder_source;
    size_t limit = 0;
    int repeats = 5;
    bool verify_only = false;
};
Options options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "--data records.bin --model model.bin [--output bench.json] "
                         "[--predictions scores.f32] [--encoder-source weights.f32] "
                         "[--limit N] [--repeats N] [--verify-only]\n";
            std::exit(0);
        }
        if (arg == "--verify-only") { o.verify_only = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("Missing argument for " + arg);
        std::string value = argv[++i];
        if (arg == "--data") o.data = value;
        else if (arg == "--model") o.model = value;
        else if (arg == "--output") o.output = value;
        else if (arg == "--predictions") o.predictions = value;
        else if (arg == "--encoder-source") o.encoder_source = value;
        else if (arg == "--limit") o.limit = std::stoull(value);
        else if (arg == "--repeats") o.repeats = std::stoi(value);
        else throw std::runtime_error("Unknown option " + arg);
    }
    if (o.data.empty() || o.model.empty() || o.repeats < 1)
        throw std::runtime_error("--data, --model, and positive --repeats required");
    return o;
}

int run(const Options &o) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
        throw std::runtime_error("This v1 loader requires a little-endian machine");
    Model m = Model::load(o.model);
    EncoderParity encoder = verify_encoder(m, o.encoder_source);
    std::vector<Record> records = load_records(o.data, o.limit);
    if (records.empty()) throw std::runtime_error("No records");
    Parents parents = prepare_parents(m, records);
    ChainedParity chained = verify_chains(m, records);
    std::array<Parity, 3> parity;
    std::vector<float> predictions(records.size());
    Accumulator full, changed;
    for (size_t i = 0; i < records.size(); ++i) {
        const Record &r = records[i];
        refresh(m, r.child_rows, full);
        float reference = forward(m, full, r.next);
        if (!std::isfinite(reference)) throw std::runtime_error("Non-finite full inference");
        predictions[i] = reference;
        for (int method_id = 1; method_id < 3; ++method_id) {
            transition(m, r, parents.groups[parents.group_of_record[i]].acc,
                       changed, static_cast<Method>(method_id));
            compare_acc(m, full, changed, parity[method_id]);
            float score = forward(m, changed, r.next);
            double error = std::abs(double(reference) - score);
            parity[method_id].max_score_abs = std::max(parity[method_id].max_score_abs, error);
            parity[method_id].sum_score_abs += error;
            if (!std::isfinite(score) || error > 1e-3 + 1e-5 * std::abs(double(reference)))
                ++parity[method_id].failed_scores;
        }
    }
    if (!o.predictions.empty()) {
        std::ofstream prediction_file(o.predictions, std::ios::binary);
        if (!prediction_file) throw std::runtime_error("Cannot write predictions");
        prediction_file.write(reinterpret_cast<const char *>(predictions.data()),
                              std::streamsize(predictions.size() * sizeof(float)));
    }
    const std::array<std::string, 7> categories = {
        "all", "no_clear", "clear_1", "clear_2", "clear_3", "clear_4", "garbage"};
    std::array<std::vector<size_t>, 7> subsets;
    for (size_t i = 0; i < records.size(); ++i) {
        subsets[0].push_back(i);
        std::string category = event_name(records[i]);
        for (size_t c = 1; c < categories.size(); ++c)
            if (category == categories[c]) subsets[c].push_back(i);
    }
    std::array<std::array<Timing, 7>, 3> timings;
    std::array<Timing, 3> siblings;
    if (!o.verify_only) {
        for (int method_id = 0; method_id < 3; ++method_id) {
            Method method = static_cast<Method>(method_id);
            for (size_t category = 0; category < categories.size(); ++category)
                timings[method_id][category] = benchmark_subset(m, records, parents, subsets[category], method, o.repeats);
            siblings[method_id] = benchmark_siblings(m, records, parents, method, o.repeats);
        }
    }
    std::ofstream file;
    if (!o.output.empty()) {
        file.open(o.output);
        if (!file) throw std::runtime_error("Cannot write benchmark output");
    }
    std::ostream &out = o.output.empty() ? std::cout : file;
    out << std::setprecision(12);
    out << "{\n\"schema\":\"spatial-nnue-native-v1\",\n"
        << "\"kind\":" << m.kind << ",\"records\":" << records.size()
        << ",\"parent_groups\":" << parents.groups.size()
        << ",\"mean_siblings\":" << double(records.size()) / parents.groups.size()
        << ",\"repeats\":" << o.repeats << ",\"model_bytes\":" << m.serialized_bytes
        << ",\"timings_measured\":" << (o.verify_only ? "false" : "true")
        << ",\"accumulator_active_bytes\":" << 4 * (m.posdim + PX * m.localdim)
        << ",\"accumulator_storage_bytes\":" << sizeof(Accumulator)
        << ",\"precomputed_parent_accumulator_bytes\":" << sizeof(Accumulator) * parents.groups.size()
        << ",\"empty_baseline_folded\":true,\"attention_empty_tokens_retained\":true,\n"
        << "\"arithmetic\":\"float32 embeddings, accumulators, and head\",\n"
        << "\"encoder_source_parity\":{\"enabled\":" << (encoder.enabled ? "true" : "false")
        << ",\"patterns\":" << encoder.patterns << ",\"components\":" << encoder.components
        << ",\"failed_components\":" << encoder.failed_components << ",\"max_abs\":" << encoder.max_abs << "},\n"
        << "\"parity_tolerance\":{\"acc_abs\":0.0001,\"acc_rel\":0.00003,"
           "\"score_abs\":0.001,\"score_rel\":0.00001},\n\"parity\":{";
    for (int method_id = 1; method_id < 3; ++method_id) {
        if (method_id > 1) out << ',';
        const auto &p = parity[method_id];
        out << '"' << method_name(static_cast<Method>(method_id)) << "\":{"
            << "\"max_acc_abs\":" << p.max_acc_abs
            << ",\"failed_acc_components\":" << p.failed_acc_components
            << ",\"max_score_abs\":" << p.max_score_abs
            << ",\"mean_score_abs\":" << p.sum_score_abs / records.size()
            << ",\"failed_scores\":" << p.failed_scores << '}';
    }
    out << "},\n\"chained_selected_parity\":{\"transitions\":" << chained.transitions
        << ",\"segments\":" << chained.segments << ",\"discontinuities\":" << chained.discontinuities
        << ",\"max_segment_length\":" << chained.max_segment_length << ",\"methods\":{";
    for (int method_id = 1; method_id < 3; ++method_id) {
        if (method_id > 1) out << ',';
        const auto &p = chained.methods[method_id];
        out << '"' << method_name(static_cast<Method>(method_id)) << "\":{"
            << "\"max_acc_abs\":" << p.max_acc_abs
            << ",\"failed_acc_components\":" << p.failed_acc_components
            << ",\"max_score_abs\":" << p.max_score_abs
            << ",\"mean_score_abs\":" << (chained.transitions ? p.sum_score_abs / chained.transitions : 0)
            << ",\"failed_scores\":" << p.failed_scores << '}';
    }
    out << "}},\n\"update_and_head_parent_precomputed\":{";
    for (int method_id = 0; method_id < 3; ++method_id) {
        if (method_id) out << ',';
        out << '\n' << '"' << method_name(static_cast<Method>(method_id)) << "\":{";
        for (size_t category = 0; category < categories.size(); ++category) {
            if (category) out << ',';
            out << '"' << categories[category] << "\":";
            print_timing(out, timings[method_id][category]);
        }
        out << '}';
    }
    out << "},\n\"sibling_pipeline_including_parent_refresh\":{";
    for (int method_id = 0; method_id < 3; ++method_id) {
        if (method_id) out << ',';
        out << '"' << method_name(static_cast<Method>(method_id)) << "\":";
        print_timing(out, siblings[method_id]);
    }
    out << "}\n}\n";
    if (!o.output.empty()) std::cout << "Wrote " << (o.verify_only ? "validation" : "benchmark")
        << " for " << records.size() << " records to " << o.output << '\n';
    bool good = !encoder.failed_components;
    for (int i = 1; i < 3; ++i)
        good = good && !parity[i].failed_scores && !parity[i].failed_acc_components &&
            !chained.methods[i].failed_scores && !chained.methods[i].failed_acc_components;
    return good ? 0 : 2;
}
}  // namespace spatial

// Minimal C ABI for the Python rollout harness. A loaded model is immutable;
// each prediction call owns its scratch accumulator and can run concurrently.
namespace {
thread_local std::string spatial_error;
}
extern "C" {
const char *spatial_last_error() { return spatial_error.c_str(); }

void *spatial_model_load(const char *path) {
    try {
        spatial_error.clear();
        if (!path) throw std::runtime_error("Null model path");
        return new spatial::Model(spatial::Model::load(path));
    } catch (const std::exception &error) {
        spatial_error = error.what();
        return nullptr;
    }
}

void spatial_model_free(void *handle) {
    delete static_cast<spatial::Model *>(handle);
}

int spatial_model_predict(void *handle, const uint16_t *boards,
                          const uint8_t *next, float *out, size_t count) {
    try {
        spatial_error.clear();
        if (!handle || (count && (!boards || !next || !out)))
            throw std::runtime_error("Null inference argument");
        const auto &model = *static_cast<spatial::Model *>(handle);
        spatial::Accumulator accumulator;
        for (size_t i = 0; i < count; ++i) {
            if (next[i] >= 7) throw std::runtime_error("Next piece must be in 0..6");
            spatial::refresh(model, boards + i * spatial::H, accumulator);
            out[i] = spatial::forward(model, accumulator, next[i]);
            if (!std::isfinite(out[i])) throw std::runtime_error("Non-finite prediction");
        }
        return 0;
    } catch (const std::exception &error) {
        spatial_error = error.what();
        return -1;
    }
}
}

#ifndef SPATIAL_KERNEL_LIBRARY
int main(int argc, char **argv) {
    try { return spatial::run(spatial::options(argc, argv)); }
    catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
#endif
