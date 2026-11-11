#include "engine.hpp"
#include "geometry_tables.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_set>

namespace tetrio {
namespace {
bool valid_piece(Piece p) { return int(p) >= 0 && int(p) < 7; }

// Official falling-y starts at a .96 phase. Its occupancy routine rejects a
// negative fractional coordinate before ceil(), so a mino whose normalized
// coordinate is the very top buffer row cannot be entered at that phase.
// Keep Board/collides as integer geometry, but conservatively use this phase
// for generated actions rather than promise a frame-dependent ceiling kick.
bool phase_collides(const Board& board, Piece piece, const Pose& pose) {
    if (collides(board,piece,pose)) return true;
    if (pose.y < HEIGHT-1) return false;
    for (const auto& cell : piece_cells(piece,pose))
        if (cell.y >= HEIGHT-1) return true;
    return false;
}

Spin corner_spin(const Board& board, Piece piece, const Pose& pose, int kick_index) {
    if (int(piece) < int(Piece::T) || !grounded(board, piece, pose)) return Spin::None;
    const int r = pose.rotation & 3;
    const auto& offsets = geometry_tables::corners[int(piece)-int(Piece::T)][r];
    int occupied = 0, front = 0;
    for (int i = 0; i < 4; ++i) {
        // Every corner-tested standard piece has dx=dy=1.
        const int y=pose.y-1-offsets[i][1];
        if (y >= HEIGHT-1 || board.occupied(pose.x + 1 + offsets[i][0],y)) {
            ++occupied;
            // T corner i faces orientations i-1 and i.
            if (piece == Piece::T && (r == i || r == ((i+3)&3))) ++front;
        }
    }
    if (occupied < 3) return Spin::None;
    if (piece == Piece::T && front != 2 && kick_index != 3) return Spin::Mini;
    return Spin::Full;
}

bool immobile(const Board& board, Piece piece, const Pose& pose) {
    Pose q = pose;
    --q.x; if (!phase_collides(board, piece, q)) return false;
    q.x += 2; if (!phase_collides(board, piece, q)) return false;
    --q.x; --q.y; if (!phase_collides(board, piece, q)) return false;
    q.y += 2; return phase_collides(board, piece, q);
}

uint64_t placement_key(Piece piece, const Pose& pose, Spin spin) {
    auto cells = piece_cells(piece, pose);
    std::array<int,4> ids{};
    for (int i=0; i<4; ++i) ids[i] = cells[i].y*WIDTH + cells[i].x;
    std::sort(ids.begin(), ids.end());
    uint64_t key = unsigned(spin);
    for (int id : ids) key = (key << 9) | unsigned(id);
    return key;
}
} // namespace

std::array<Cell,4> piece_cells(Piece piece, const Pose& pose) {
    if (!valid_piece(piece)) throw std::invalid_argument("piece_cells needs a standard tetromino");
    std::array<Cell,4> result{};
    const auto& shape = geometry_tables::shapes[int(piece)][pose.rotation & 3];
    for (int i=0; i<4; ++i) result[i] = {pose.x + shape[i][0], pose.y - shape[i][1]};
    return result;
}

bool collides(const Board& board, Piece piece, const Pose& pose) {
    if (!valid_piece(piece)) return true;
    const auto& shape = geometry_tables::shapes[int(piece)][pose.rotation & 3];
    for (const auto& cell : shape)
        if (board.occupied(pose.x + cell[0], pose.y - cell[1])) return true;
    return false;
}

bool grounded(const Board& board, Piece piece, const Pose& pose) {
    if (collides(board, piece, pose)) return false;
    Pose below = pose;
    --below.y;
    return collides(board, piece, below);
}

std::optional<RotationResult> rotate_pose(const Board& board, Piece piece, const Pose& pose,
                                         const Rules& rules, Move move) {
    if (phase_collides(board, piece, pose)) return std::nullopt;
    int delta = 0;
    if (move == Move::CW) delta = 1;
    else if (move == Move::CCW) delta = 3;
    else if (move == Move::Half && rules.spin_180) delta = 2;
    else return std::nullopt;
    Pose q = pose;
    q.rotation = (pose.rotation + delta) & 3;
    if (!phase_collides(board, piece, q)) return RotationResult{q, -1};
    if (piece == Piece::O) return std::nullopt; // O permits rotation, not displaced kicks.
    const auto& table = piece == Piece::I
        ? (rules.kicks == KickSystem::SRSPlus ? geometry_tables::i_plus : geometry_tables::i_srs)
        : geometry_tables::common;
    const auto& kicks = table[pose.rotation & 3][q.rotation];
    for (int i=0; i<kicks.count; ++i) {
        q.x = pose.x + kicks.xy[i][0];
        q.y = pose.y - kicks.xy[i][1];
        if (!phase_collides(board, piece, q)) return RotationResult{q, i};
    }
    return std::nullopt;
}

std::optional<Pose> spawn_pose(const Board& board, Piece piece, const Rules& rules,
                               bool last_clear) {
    if (!valid_piece(piece)) return std::nullopt;
    Pose pose{piece == Piece::O ? 4 : 3, rules.visible_height + 2, 0};
    if (!phase_collides(board, piece, pose)) return pose;
    if (!last_clear || !rules.clutch) return std::nullopt;
    // Clutch raises a blocked spawn only after the previous lock cleared.
    for (++pose.y; pose.y < HEIGHT+4; ++pose.y)
        if (!phase_collides(board, piece, pose)) return pose;
    return std::nullopt;
}

Spin classify_spin(const Board& board, Piece piece, const Pose& pose, const Rules& rules,
                   int kick_index, bool rotated180) {
    // Precondition: a successful rotation has just occurred at this pose.
    // Engine 19 promotes kick index 3 for both 90 and 180 degree rotations.
    // It does not use Triangle's quarter-turn-specific kick-vector shortcut.
    (void)rotated180;
    if (phase_collides(board, piece, pose)) return Spin::None;
    const auto mode = rules.spin_mode;
    if (mode == SpinMode::None) return Spin::None;
    if (mode == SpinMode::Stupid) return grounded(board,piece,pose) ? Spin::Full : Spin::None;
    if (mode == SpinMode::TSpins || mode == SpinMode::TSpinsPlus) {
        if (piece != Piece::T) return Spin::None;
        Spin spin = corner_spin(board,piece,pose,kick_index);
        if (spin != Spin::None || mode == SpinMode::TSpins) return spin;
        return immobile(board,piece,pose) ? Spin::Mini : Spin::None;
    }
    if (mode == SpinMode::Handheld) return corner_spin(board,piece,pose,kick_index);
    if (piece == Piece::T) {
        Spin spin = corner_spin(board,piece,pose,kick_index);
        if (spin != Spin::None && mode == SpinMode::MiniOnly) return Spin::Mini;
        // Non-plus All and AllMini do NOT allow T immobility fallback.
        if (spin != Spin::None || mode == SpinMode::All || mode == SpinMode::AllMini) return spin;
    }
    if (!immobile(board,piece,pose)) return Spin::None;
    const bool mini = mode == SpinMode::MiniOnly || mode == SpinMode::AllMini ||
        mode == SpinMode::AllMiniPlus || (mode == SpinMode::AllPlus && piece == Piece::T);
    return mini ? Spin::Mini : Spin::Full;
}

std::vector<Placement> generate_placements(const Board& board, Piece piece,
                                          const Rules& rules, bool last_clear) {
    const auto spawn = spawn_pose(board,piece,rules,last_clear);
    if (!spawn) return {};
    // Allow empty matrix borders beyond the board; occupied minos still must fit.
    constexpr int MIN_X=-3, NX=WIDTH+3, NY=HEIGHT+4;
    constexpr int STATES=NX*NY*4*3;
    std::array<int,STATES> seen;
    seen.fill(-1);
    struct Node {
        Pose pose;
        Spin spin;
        int parent;
        Move move;
        int kick;
        bool last_rotation;
        bool half;
    };
    std::vector<Node> nodes;
    nodes.reserve(2048);
    auto add = [&](const Node& node) {
        const auto& q=node.pose;
        if (q.x < MIN_X || q.x >= MIN_X+NX || q.y < 0 || q.y >= NY) return;
        const int id = (((q.y*NX + q.x-MIN_X)*4 + (q.rotation&3))*3 + int(node.spin));
        if (seen[id] >= 0) return;
        seen[id] = int(nodes.size());
        nodes.push_back(node);
    };
    add({*spawn,Spin::None,-1,Move::Down,-1,false,false});
    std::vector<Placement> result;
    result.reserve(64);
    std::unordered_set<uint64_t> emitted;
    emitted.reserve(128);
    for (std::size_t head=0; head<nodes.size(); ++head) {
        const Node node=nodes[head]; // add() may reallocate the vector.
        if (grounded(board,piece,node.pose) && emitted.insert(placement_key(piece,node.pose,node.spin)).second) {
            Placement p;
            p.piece=piece; p.pose=node.pose; p.spin=node.spin;
            p.last_rotation=node.last_rotation; p.rotated180=node.half; p.kick_index=node.kick;
            for (int at=int(head); nodes[at].parent>=0; at=nodes[at].parent)
                p.path.push_back(nodes[at].move);
            std::reverse(p.path.begin(),p.path.end());
            result.push_back(std::move(p));
        }
        for (Move move : {Move::Left,Move::Right,Move::Down}) {
            Pose q=node.pose;
            if (move==Move::Left) --q.x;
            else if (move==Move::Right) ++q.x;
            else --q.y;
            if (!phase_collides(board,piece,q)) add({q,Spin::None,int(head),move,-1,false,false});
        }
        for (Move move : {Move::CW,Move::CCW,Move::Half}) {
            const auto rotation=rotate_pose(board,piece,node.pose,rules,move);
            if (!rotation) continue;
            const Spin spin=classify_spin(board,piece,rotation->pose,rules,rotation->kick_index,move==Move::Half);
            add({rotation->pose,spin,int(head),move,rotation->kick_index,true,move==Move::Half});
        }
    }
    return result;
}
} // namespace tetrio
