#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Placement-level TETR.IO environment. No gravity, DAS, lock-delay or rendering.
// Coordinates are bottom-first; occupancy includes the 20 hidden buffer rows.
namespace tetrio {
constexpr int WIDTH = 10;
constexpr int HEIGHT = 40;
constexpr uint16_t FULL_ROW = (1u << WIDTH) - 1;

enum class Piece : int8_t { None=-1, I=0, O=1, T=2, L=3, J=4, S=5, Z=6 };
enum class Spin : uint8_t { None, Mini, Full };
enum class SpinMode : uint8_t {
    None, TSpins, TSpinsPlus, All, AllPlus, AllMini, AllMiniPlus,
    MiniOnly, Handheld, Stupid
};
enum class KickSystem : uint8_t { SRSPlus, SRS };
enum class Move : uint8_t { Left, Right, Down, CW, CCW, Half };
enum class ComboMode : uint8_t { Multiplier, Classic, None, Modern };
enum class GarbageBlocking : uint8_t { Combo, Limited, None };
enum class GarbageRounding : uint8_t { Down, Random };
enum class GarbageTargetBonus : uint8_t { None, Offensive, Defensive };

struct Cell { int x=0, y=0; };
struct Pose { int x=0, y=0, rotation=0; };

struct Board {
    std::array<uint16_t, HEIGHT> rows{};
    std::array<uint16_t, HEIGHT> garbage{};
    std::array<uint16_t, HEIGHT> solid{};
    bool occupied(int x, int y) const;
    bool empty() const;
    int height() const;
    void validate() const;
};

struct Placement {
    Piece piece = Piece::None;
    Pose pose{}; // Top-left matrix anchor: cells are (x+shape_x, y-shape_y).
    Spin spin = Spin::None;
    bool last_rotation = false;
    bool rotated180 = false;
    int kick_index = -1;
    std::vector<Move> path; // Geometric witness; not a frame-accurate key script.
};

struct Rules {
    SpinMode spin_mode = SpinMode::AllMiniPlus;
    KickSystem kicks = KickSystem::SRSPlus;
    bool spin_180 = true;
    bool clutch = true;
    bool lockout = false;
    int visible_height = 20;
    int preview_count = 5;

    ComboMode combo_mode = ComboMode::Multiplier;
    bool b2b_charging = true;
    bool b2b_chaining = false;
    bool b2b_extras = false;
    int b2b_charge_at = 4;
    int b2b_charge_base = 0;
    int pc_garbage = 5;
    int pc_b2b = 1;
    bool all_clears = true;
    bool allclear_b2b_dupes = true;
    bool allclear_b2b_sends = false;
    bool allclear_charges = false;
    bool garbage_special_bonus = true;
    double garbage_multiplier = 1;
    double garbage_increase = .008;
    int garbage_margin = 10800;
    double garbage_cap = 8;
    double garbage_cap_increase = 0;
    int garbage_cap_margin = 0;
    int garbage_cap_max = 40;
    int garbage_absolute_cap = 0;
    double cancel_multiplier = 1;
    double receive_multiplier = 1;
    int opener_phase = 14;
    int attack_cap = 0;
    GarbageBlocking garbage_blocking = GarbageBlocking::Combo;
    GarbageRounding garbage_rounding = GarbageRounding::Down;
    GarbageTargetBonus garbage_target_bonus = GarbageTargetBonus::None;
    bool has_garbage = true;
    int garbage_hole_width = 1;
    double messiness_change = 1;
    double messiness_inner = 0;
    bool messiness_no_same = false;
    bool messiness_center = false;
    int messiness_timeout = 0;

    static Rules tetra_league();
    void validate() const;
};

struct GarbagePacket {
    int amount = 0;
    int hole = -1; // -1 uses the supplied local garbage RNG state.
    int width = 1;
    bool active = true;
    bool spawn_ready = true;
    bool hardened = false;
    bool shielded = false;
    int64_t id = 0;
};

struct State {
    Board board{};
    Piece current = Piece::None;
    Piece hold = Piece::None;
    std::vector<Piece> queue; // Observed previews only, EXCLUDING current.
    bool hold_used = false;
    bool last_clear = false;
    bool dead = false;
    int combo = 0; // Official zero-based *counter state*, not Triangle's -1 sentinel.
    int b2b = 0;
    int pieces_placed = 0;
    double total_sent = 0;
    int64_t frame = 0;
    int64_t last_tank = 0;
    double current_multiplier = -1;
    double current_garbage_cap = -1;
    uint32_t rng_seed = 1;
    int last_hole = -1;
    bool hole_changed = false;
    int enemies = 1;
    double garbage_bonus = 0;
    std::vector<GarbagePacket> incoming;
};

struct Action {
    Placement placement{};
    bool use_hold = false;
    int elapsed_frames = 0; // External time advance; no physical movement simulation.
};

struct BoardDelta {
    std::array<uint16_t,4> placed_cells{{65535,65535,65535,65535}};
    std::array<int8_t,HEIGHT> row_source{}; // Child row -> original parent row, -1=new.
    uint64_t clear_mask = 0;
    int clears = 0;
    int garbage = 0;
};

struct TankRow { int hole=0; int width=1; bool hardened=false; int64_t packet_id=0; };

// Counter values are the official values AFTER pieces_placed was incremented.
// resolve_battle does not change the board, draw pieces or increment pieces_placed.
struct BattleClear {
    Piece piece = Piece::T;
    int lines = 0;
    int garbage_rows = 0;
    Spin spin = Spin::None;
    bool perfect_clear = false;
};
struct BattleResult {
    int attack=0, sent=0, cancelled=0, received=0, surge=0;
    bool blocked=false;
    std::vector<int> outgoing;
    std::vector<TankRow> tank;
};

struct ClearResult {
    BoardDelta delta{};
    int lines = 0;
    int garbage_rows = 0;
    bool perfect_clear = false;
    bool locked_above = false;
    bool valid = false;
    std::string error;
};

struct Transition {
    State state{};
    BoardDelta delta{};
    int lines = 0;
    int garbage_cleared = 0;
    Spin spin = Spin::None;
    bool perfect_clear = false;
    int attack = 0;
    int sent = 0;
    int cancelled = 0;
    int received = 0;
    int surge = 0;
    std::vector<int> outgoing;
    bool valid = false;
    std::string error;
};

// Geometry implementation. Rotation/spin provenance is computed on the parent board.
struct RotationResult { Pose pose{}; int kick_index=-1; };
std::array<Cell,4> piece_cells(Piece piece, const Pose& pose);
bool collides(const Board& board, Piece piece, const Pose& pose);
bool grounded(const Board& board, Piece piece, const Pose& pose);
std::optional<RotationResult> rotate_pose(const Board& board, Piece piece, const Pose& pose,
                                         const Rules& rules, Move move);
std::optional<Pose> spawn_pose(const Board& board, Piece piece, const Rules& rules,
                               bool last_clear=false);
Spin classify_spin(const Board& board, Piece piece, const Pose& pose, const Rules& rules,
                   int kick_index, bool rotated180);
std::vector<Placement> generate_placements(const Board& board, Piece piece,
                                          const Rules& rules, bool last_clear=false);

// Board primitives preserve garbage provenance and row-source mapping.
ClearResult place_and_clear(Board& board, const Placement& placement, int visible_height=20);
// Returns true when the official FULL top-buffer-row guard blocks an insertion.
// A partially occupied discarded top row does not itself trigger that guard.
bool insert_garbage(Board& board, const std::vector<TankRow>& rows, BoardDelta& delta);

// Queue/hold action enumeration. A depleted observed queue is unknown, not a hidden draw.
std::vector<Action> enumerate_actions(const State& state, const Rules& rules,
                                      bool allow_hold=true);
// apply_action accepts placement provenance produced by generate_placements.
// Public protocol requests are matched to enumerated actions before applying.
Transition apply_action(const State& state, const Action& action, const Rules& rules);
BattleResult resolve_battle(State& state, const BattleClear& clear, const Rules& rules,
                           bool tank=true);
// Incoming packets are already reconciled by the client transport; this applies
// receive multiplier and absolute cap. Readiness is supplied by that client.
int enqueue_garbage(State& state, GarbagePacket packet, const Rules& rules);
void advance_frames(State& state, const Rules& rules, int elapsed_frames);
std::string state_key(const State& state);

Piece parse_piece(const std::string& value);
std::string piece_name(Piece piece);
std::string spin_name(Spin spin);
SpinMode parse_spin_mode(const std::string& value);
std::string spin_mode_name(SpinMode mode);
std::string move_name(Move move);
} // namespace tetrio
