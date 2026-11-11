#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tetrio {
namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

int checked_int(int64_t value, const char* message) {
    require(value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max(), message);
    return static_cast<int>(value);
}

int checked_add(int left, int right, const char* message) {
    return checked_int(static_cast<int64_t>(left) + right, message);
}

int checked_floor(double value, const char* message) {
    const double floored = std::floor(value);
    require(std::isfinite(value) && floored >= std::numeric_limits<int>::min() &&
            floored <= std::numeric_limits<int>::max(), message);
    return static_cast<int>(floored);
}

double next_float(State& state) {
    constexpr uint64_t modulus = 2147483647;
    uint64_t seed = state.rng_seed % modulus;
    if (seed == 0) seed = modulus - 1;
    state.rng_seed = static_cast<uint32_t>((16807 * seed) % modulus);
    return (static_cast<double>(state.rng_seed) - 1) / 2147483646.0;
}

int auto_round(double value, State& state, const Rules& rules) {
    const double base = std::floor(value);
    const int rounded = checked_floor(value, "attack is outside supported integer range");
    if (rules.garbage_rounding == GarbageRounding::Down || base == value)
        return rounded;
    return checked_add(rounded, next_float(state) < value - base ? 1 : 0,
                       "rounded attack is outside supported integer range");
}

void init_dynamic(State& state, const Rules& rules) {
    if (state.current_multiplier < 0) state.current_multiplier = rules.garbage_multiplier;
    if (state.current_garbage_cap < 0) state.current_garbage_cap = rules.garbage_cap;
}

int reroll_column(State& state, const Rules& rules) {
    const int padding = rules.messiness_center ? 2 : 0;
    const int width = WIDTH - (rules.garbage_hole_width - 1);
    int column;
    if (rules.messiness_no_same && state.last_hole >= 0) {
        column = padding + static_cast<int>(std::floor(next_float(state) * (width - 1 - 2 * padding)));
        if (column >= state.last_hole) ++column;
    } else {
        column = padding + static_cast<int>(std::floor(next_float(state) * (width - 2 * padding)));
    }
    state.last_hole = column;
    return column;
}

int pending_count(const State& state) {
    int count = 0;
    for (const auto& packet : state.incoming)
        if (!packet.shielded) count += packet.amount;
    return count;
}

void packet_exhausted(State& state, const Rules& rules) {
    // The comparison itself consumes RNG even at probability zero or one.
    if (next_float(state) < rules.messiness_change) {
        reroll_column(state, rules);
        state.hole_changed = true;
    }
}

void offence(int amount, State& state, BattleResult& result, const Rules& rules) {
    if (!rules.has_garbage || amount < 1) return;
    result.sent = checked_add(result.sent, amount, "total sent attack exceeds supported integer range");
    state.total_sent += amount;
    require(std::isfinite(state.total_sent), "total_sent overflow");
    result.outgoing.push_back(amount);
}

void fight(int amount, State& state, BattleResult& result, const Rules& rules) {
    if (!rules.has_garbage) return;
    result.attack = checked_add(result.attack, amount, "total attack exceeds supported integer range");
    int extra = 0;
    if (rules.garbage_target_bonus == GarbageTargetBonus::Defensive)
        extra = auto_round(state.garbage_bonus * state.current_multiplier, state, rules);
    extra = checked_add(extra, auto_round(rules.cancel_multiplier * amount - amount, state, rules),
                        "additional cancellation exceeds supported integer range");
    if (state.pieces_placed <= rules.opener_phase && pending_count(state) >= state.total_sent)
        extra = checked_add(extra, amount, "opener cancellation exceeds supported integer range");

    // Official cancellation also sees unconfirmed and non-spawn-ready packets.
    // Hardened packets are skipped; extra cancellation cannot become outgoing attack.
    size_t index = 0;
    while (amount > 0 || extra > 0) {
        if (index >= state.incoming.size()) break;
        auto& packet = state.incoming[index];
        if (packet.hardened) { ++index; continue; }
        --packet.amount;
        if (packet.amount == 0) {
            state.incoming.erase(state.incoming.begin() + static_cast<std::ptrdiff_t>(index));
            packet_exhausted(state, rules);
        }
        if (amount != 0) --amount; else --extra;
        result.cancelled = checked_add(result.cancelled, 1, "cancelled lines exceed supported integer range");
    }
    offence(amount, state, result, rules);
}

void emit_attack(int amount, State& state, BattleResult& result, const Rules& rules) {
    if (rules.garbage_blocking == GarbageBlocking::None) {
        result.attack = checked_add(result.attack, amount, "total attack exceeds supported integer range");
        offence(amount, state, result, rules);
    } else {
        fight(amount, state, result, rules);
    }
}

void tank_damage(State& state, BattleResult& result, const Rules& rules) {
    if (!rules.has_garbage) return;
    const int cap = checked_floor(std::min(static_cast<double>(rules.garbage_cap_max), state.current_garbage_cap),
                                  "garbage cap is outside supported integer range");
    if (!state.hole_changed && rules.messiness_timeout &&
        state.frame >= state.last_tank && state.frame - state.last_tank >= rules.messiness_timeout) {
        reroll_column(state, rules);
        state.hole_changed = true;
    }

    // A not-ready packet does not block later ready packets. The official engine
    // appends waiting packets after leftover ready packets following every tank.
    std::vector<GarbagePacket> ready, waiting;
    ready.reserve(state.incoming.size());
    waiting.reserve(state.incoming.size());
    for (const auto& packet : state.incoming) {
        (packet.active && packet.spawn_ready ? ready : waiting).push_back(packet);
    }
    size_t index = 0;
    for (int count = 0; count < cap && index < ready.size(); ++count) {
        auto& packet = ready[index];
        --packet.amount;
        result.received = checked_add(result.received, 1, "received lines exceed supported integer range");
        int hole = state.last_hole;
        if (packet.hole >= 0) hole = packet.hole;
        else if (hole < 0 || next_float(state) < rules.messiness_inner) {
            if (!state.hole_changed) {
                hole = reroll_column(state, rules);
                state.hole_changed = true;
            }
        }
        result.tank.push_back({hole, packet.width, packet.hardened, packet.id});
        state.hole_changed = false;
        if (packet.amount == 0) {
            ++index;
            packet_exhausted(state, rules);
        }
    }
    state.incoming.assign(ready.begin() + static_cast<std::ptrdiff_t>(index), ready.end());
    state.incoming.insert(state.incoming.end(), waiting.begin(), waiting.end());
    if (result.received) state.last_tank = state.frame;
}

void validate_packet(const GarbagePacket& packet) {
    require(packet.amount > 0 && packet.amount <= 1000000, "garbage packet amount must be in 1..1000000");
    require(packet.hole >= -1 && packet.hole < WIDTH, "garbage hole must be -1 or a board column");
    require(packet.width >= 1 && packet.width <= WIDTH, "garbage width must be in 1..10");
}

void validate_state(const State& state) {
    require(state.combo >= 0 && state.b2b >= 0 && state.pieces_placed >= 0,
            "official combo/b2b/piece counters must be nonnegative");
    require(std::isfinite(state.total_sent) && state.total_sent >= 0, "total_sent must be nonnegative");
    require(state.frame >= 0 && state.last_tank >= 0, "frame counters must be nonnegative");
    require(std::isfinite(state.current_multiplier) && std::isfinite(state.current_garbage_cap),
            "dynamic garbage settings must be finite");
    require(state.last_hole >= -1 && state.last_hole < WIDTH, "last_hole must be -1 or a board column");
    require(!state.hole_changed || state.last_hole >= 0, "hole_changed requires a known last_hole");
    require(state.enemies >= 0 && std::isfinite(state.garbage_bonus) && state.garbage_bonus >= 0,
            "target bonus state must be nonnegative");
    int64_t pending = 0;
    for (const auto& packet : state.incoming) { validate_packet(packet); pending += packet.amount; }
    require(pending <= std::numeric_limits<int>::max(), "pending garbage exceeds supported integer range");
}

template<class T> void key_integer(std::string& out, T value) {
    using U = std::make_unsigned_t<T>;
    U bits = static_cast<U>(value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        out.push_back(static_cast<char>(bits & 255));
        bits >>= 8;
    }
}
void key_double(std::string& out, double value) {
    uint64_t bits;
    static_assert(sizeof(bits) == sizeof(value), "expected binary64 double");
    std::memcpy(&bits, &value, sizeof(bits));
    key_integer(out, bits);
}

} // namespace

Rules Rules::tetra_league() { return Rules{}; }

void Rules::validate() const {
    require(visible_height == 20, "this placement environment supports 10x20+20 boards");
    require(preview_count >= 0, "preview_count must be nonnegative");
    require(b2b_charge_at >= 0 && b2b_charge_base >= 0 && pc_garbage >= 0 && pc_b2b >= 0,
            "attack and B2B settings must be nonnegative");
    for (const double value : {garbage_multiplier, garbage_increase, garbage_cap,
                              garbage_cap_increase, cancel_multiplier, receive_multiplier})
        require(std::isfinite(value) && value >= 0, "garbage multipliers and caps must be finite/nonnegative");
    require(garbage_margin >= 0 && garbage_cap_margin >= 0 && garbage_cap_max >= 0 &&
            garbage_absolute_cap >= 0 && opener_phase >= 0 && attack_cap >= 0,
            "garbage counters and thresholds must be nonnegative");
    require(garbage_hole_width >= 1 && garbage_hole_width <= WIDTH, "garbage_hole_width must be in 1..10");
    require(std::isfinite(messiness_change) && messiness_change >= 0 && messiness_change <= 1 &&
            std::isfinite(messiness_inner) && messiness_inner >= 0 && messiness_inner <= 1,
            "messiness probabilities must be in 0..1");
    require(messiness_timeout >= 0, "messiness_timeout must be nonnegative");
    const int available = WIDTH - (garbage_hole_width - 1) - (messiness_center ? 4 : 0);
    require(available >= (messiness_no_same ? 2 : 1), "hole width/center/no-same leaves no legal garbage column");
}

void advance_frames(State& state, const Rules& rules, int elapsed_frames) {
    require(elapsed_frames >= 0, "elapsed_frames must be nonnegative");
    require(state.frame <= std::numeric_limits<int64_t>::max() - elapsed_frames,
            "frame counter overflow");
    init_dynamic(state, rules);
    for (int i = 0; i < elapsed_frames; ++i) {
        ++state.frame;
        if (rules.garbage_increase && state.frame > rules.garbage_margin)
            state.current_multiplier += rules.garbage_increase / 60;
        if (rules.garbage_cap_increase && state.frame > rules.garbage_cap_margin)
            state.current_garbage_cap += rules.garbage_cap_increase / 60;
        require(std::isfinite(state.current_multiplier) && std::isfinite(state.current_garbage_cap),
                "dynamic garbage setting overflow");
    }
}

int enqueue_garbage(State& state, GarbagePacket packet, const Rules& rules) {
    rules.validate();
    validate_packet(packet);
    init_dynamic(state, rules);
    packet.amount = auto_round(packet.amount * rules.receive_multiplier, state, rules);
    if (rules.garbage_absolute_cap) {
        int64_t total = 0;
        for (const auto& queued : state.incoming) total += queued.amount;
        // A packet received while already at the cap is retained as shielded
        // until the client processes its hit/confirmation event.
        if (total >= rules.garbage_absolute_cap) packet.shielded = true;
        else packet.amount = std::min(packet.amount, rules.garbage_absolute_cap - static_cast<int>(total));
    }
    const int accepted = packet.amount;
    if (accepted > 0) state.incoming.push_back(packet);
    return accepted;
}

BattleResult resolve_battle(State& state, const BattleClear& input, const Rules& rules, bool tank) {
    rules.validate();
    validate_state(state);
    require(input.lines >= 0 && input.lines <= 4 && input.garbage_rows >= 0 &&
            input.garbage_rows <= input.lines, "clear counts must be valid for standard tetrominoes");
    init_dynamic(state, rules);
    BattleResult result;
    const int lines = input.lines;
    const bool pc = input.perfect_clear && rules.all_clears;
    const bool spin = input.spin != Spin::None;
    const bool mini = input.spin == Spin::Mini;
    if (input.garbage_rows) state.last_tank = state.frame;

    int b2b_added = pc ? rules.pc_b2b : 0;
    if (lines) {
        if (rules.combo_mode != ComboMode::None)
            state.combo = checked_add(state.combo, 1, "combo counter overflow");
        if (lines >= 4 || spin) b2b_added = checked_add(b2b_added, 1, "B2B reward overflow");
        if (pc && !rules.allclear_b2b_dupes)
            b2b_added = std::max(rules.pc_b2b, b2b_added - rules.pc_b2b);
        if (pc && rules.allclear_charges)
            b2b_added = checked_int(std::max<int64_t>(b2b_added,
                static_cast<int64_t>(rules.b2b_charge_at) + 1 - state.b2b), "B2B charge reward overflow");
    } else {
        state.combo = 0;
    }
    if (b2b_added > 0) state.b2b = checked_add(state.b2b, b2b_added, "B2B counter overflow");
    else if (lines) {
        if (rules.b2b_charging && state.b2b > rules.b2b_charge_at) {
            const int64_t charge = static_cast<int64_t>(state.b2b) - rules.b2b_charge_at + rules.b2b_charge_base;
            const int amount = checked_floor(charge * state.current_multiplier, "B2B surge exceeds supported integer range");
            const int third = checked_floor(amount / 3.0 + .5, "B2B surge slice exceeds supported integer range");
            const int before = result.attack;
            for (const int packet : {third, third, amount - 2 * third})
                if (packet) emit_attack(packet, state, result, rules);
            result.surge = checked_add(result.surge, result.attack - before, "surge total exceeds supported integer range");
        }
        state.b2b = 0;
    }

    static constexpr int normal_attack[5] = {0,0,1,2,4};
    static constexpr int mini_attack[5] = {0,0,1,2,4};
    static constexpr int spin_attack[5] = {0,2,4,6,10};
    double attack = mini ? mini_attack[lines] : spin ? spin_attack[lines] : normal_attack[lines];
    if (spin && rules.spin_mode == SpinMode::Handheld && input.piece != Piece::T) attack /= 2;
    const bool b2b_sends = rules.allclear_b2b_sends || !(rules.pc_b2b == b2b_added && pc);
    if ((lines || b2b_added) && state.b2b > 1) {
        if (rules.b2b_chaining) {
            const int chain = state.b2b - 1;
            const double log = std::log1p(chain * .8);
            const double gain = std::floor(1 + log) + (chain == 1 ? 0 : (1 + std::fmod(log, 1)) / 3);
            if (b2b_sends) attack += gain;
        } else if (b2b_sends) {
            attack += rules.b2b_extras && (lines == 4 || (lines >= 2 && spin)) ? 2 : 1;
        }
    }
    if (state.combo > 1) {
        if (rules.combo_mode == ComboMode::Multiplier) {
            attack *= 1 + .25 * (state.combo - 1);
            if (state.combo > 2) attack = std::max(std::log1p((state.combo - 1) * 1.25), attack);
        } else if (rules.combo_mode == ComboMode::Classic) {
            static constexpr int table[] = {0,1,1,2,2,3,3,4,4,4,5};
            attack += table[std::clamp(state.combo - 2, 0, 10)];
        } else if (rules.combo_mode == ComboMode::Modern) {
            static constexpr int table[] = {0,1,1,2,2,2,3,3,3,3,3,3,4};
            attack += table[std::clamp(state.combo - 2, 0, 12)];
        }
    }
    if (lines && rules.garbage_target_bonus != GarbageTargetBonus::None) {
        const int targets = state.enemies <= 1 ? 0 : 2 * std::min(state.enemies, 6) - 3;
        if (rules.garbage_target_bonus == GarbageTargetBonus::Offensive) attack += targets;
        else {
            state.garbage_bonus = targets * state.current_multiplier;
            require(std::isfinite(state.garbage_bonus), "defensive target bonus overflow");
        }
    }
    int amount = auto_round(attack * state.current_multiplier, state, rules);
    if (rules.garbage_special_bonus && input.garbage_rows && (lines == 4 || spin))
        amount = checked_add(amount, 1, "downstack bonus exceeds supported integer range");
    if (rules.attack_cap) amount = std::min(amount, rules.attack_cap);
    result.blocked = rules.garbage_blocking == GarbageBlocking::Combo && (lines || amount);
    if (rules.garbage_blocking == GarbageBlocking::None || lines || amount)
        emit_attack(amount, state, result, rules);
    if (pc) emit_attack(auto_round(rules.pc_garbage * state.current_multiplier, state, rules), state, result, rules);
    if (tank && !result.blocked) tank_damage(state, result, rules);
    return result;
}

Transition apply_action(const State& parent, const Action& action, const Rules& rules) {
    Transition result;
    result.state = parent;
    try {
        rules.validate();
        parent.board.validate();
        validate_state(parent);
        require(!parent.dead, "cannot place a piece after topout");
        require(parent.current != Piece::None, "current piece is not observed");
        require(action.elapsed_frames >= 0, "elapsed_frames must be nonnegative");
        auto& state = result.state;
        Piece placed = state.current;
        if (action.use_hold) {
            require(!state.hold_used, "hold was already used for the current piece");
            if (state.hold == Piece::None) {
                require(!state.queue.empty(), "empty hold requires an observed next piece");
                placed = state.queue.front();
                state.queue.erase(state.queue.begin());
            } else placed = state.hold;
            state.hold = state.current;
        }
        require(placed == action.placement.piece, "placement piece disagrees with current/hold");
        require(spawn_pose(parent.board, placed, rules, parent.last_clear).has_value(), "piece cannot spawn");
        require(grounded(parent.board, placed, action.placement.pose), "placement is not grounded");
        Placement placement = action.placement;
        placement.spin = placement.last_rotation
            ? classify_spin(parent.board, placed, placement.pose, rules, placement.kick_index, placement.rotated180)
            : Spin::None;
        advance_frames(state, rules, action.elapsed_frames);
        const auto clear = place_and_clear(state.board, placement, rules.visible_height);
        require(clear.valid, clear.error.c_str());
        result.delta = clear.delta;
        result.lines = clear.lines;
        result.garbage_cleared = clear.garbage_rows;
        result.spin = placement.spin;
        result.perfect_clear = clear.perfect_clear && rules.all_clears;
        state.pieces_placed = checked_add(state.pieces_placed, 1, "placed-piece counter overflow");
        state.last_clear = clear.lines > 0;
        state.hold_used = false;
        const auto battle = resolve_battle(state, {placed, clear.lines, clear.garbage_rows,
                                                 placement.spin, result.perfect_clear}, rules);
        result.attack = battle.attack;
        result.sent = battle.sent;
        result.cancelled = battle.cancelled;
        result.received = battle.received;
        result.surge = battle.surge;
        result.outgoing = battle.outgoing;
        if (insert_garbage(state.board, battle.tank, result.delta)) state.dead = true;
        if (rules.lockout && clear.locked_above && (!clear.lines || !rules.clutch)) state.dead = true;
        if (state.queue.empty()) state.current = Piece::None;
        else {
            state.current = state.queue.front();
            state.queue.erase(state.queue.begin());
            if (!spawn_pose(state.board, state.current, rules, state.last_clear)) state.dead = true;
        }
        result.valid = true;
    } catch (const std::invalid_argument& error) {
        result = Transition{};
        result.state = parent;
        result.error = error.what();
    }
    return result;
}

std::string state_key(const State& state) {
    std::string key;
    key.reserve(400 + state.queue.size() + state.incoming.size() * 32);
    for (const auto* rows : {&state.board.rows, &state.board.garbage, &state.board.solid})
        for (const uint16_t row : *rows) key_integer(key, row);
    key_integer(key, static_cast<int8_t>(state.current));
    key_integer(key, static_cast<int8_t>(state.hold));
    key_integer(key, static_cast<uint32_t>(state.queue.size()));
    for (const Piece piece : state.queue) key_integer(key, static_cast<int8_t>(piece));
    key_integer(key, static_cast<uint8_t>(state.hold_used));
    key_integer(key, static_cast<uint8_t>(state.last_clear));
    key_integer(key, static_cast<uint8_t>(state.dead));
    key_integer(key, state.combo); key_integer(key, state.b2b); key_integer(key, state.pieces_placed);
    key_double(key, state.total_sent);
    key_integer(key, state.frame); key_integer(key, state.last_tank);
    key_double(key, state.current_multiplier); key_double(key, state.current_garbage_cap);
    key_integer(key, state.rng_seed); key_integer(key, state.last_hole);
    key_integer(key, static_cast<uint8_t>(state.hole_changed));
    key_integer(key, state.enemies); key_double(key, state.garbage_bonus);
    key_integer(key, static_cast<uint32_t>(state.incoming.size()));
    for (const auto& packet : state.incoming) {
        key_integer(key, packet.amount); key_integer(key, packet.hole); key_integer(key, packet.width);
        key_integer(key, static_cast<uint8_t>(packet.active));
        key_integer(key, static_cast<uint8_t>(packet.spawn_ready));
        key_integer(key, static_cast<uint8_t>(packet.hardened));
        key_integer(key, static_cast<uint8_t>(packet.shielded));
        key_integer(key, packet.id);
    }
    return key;
}

} // namespace tetrio
