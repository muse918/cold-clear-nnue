#include "engine.hpp"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <stdexcept>

namespace tetrio {

bool Board::occupied(int x, int y) const {
    return x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT || (rows[y] & (1u << x));
}
bool Board::empty() const {
    return std::all_of(rows.begin(), rows.end(), [](uint16_t row){return row == 0;});
}
int Board::height() const {
    for (int y=HEIGHT-1; y>=0; --y) if (rows[y]) return y+1;
    return 0;
}
void Board::validate() const {
    for (int y=0; y<HEIGHT; ++y) {
        if ((rows[y] | garbage[y] | solid[y]) & ~FULL_ROW)
            throw std::invalid_argument("Board rows must be 10-bit masks");
        if ((garbage[y] | solid[y]) & ~rows[y])
            throw std::invalid_argument("Garbage/solid provenance must be occupied");
    }
}

ClearResult place_and_clear(Board& board, const Placement& placement, int visible_height) {
    ClearResult result;
    for (int y=0; y<HEIGHT; ++y) result.delta.row_source[y]=int8_t(y);
    if (int(placement.piece)<0 || int(placement.piece)>6 ||
        placement.pose.rotation<0 || placement.pose.rotation>3) {
        result.error="Invalid piece or rotation"; return result;
    }
    if (collides(board,placement.piece,placement.pose) ||
        !grounded(board,placement.piece,placement.pose)) {
        result.error="Placement must be collision-free and grounded"; return result;
    }
    const auto cells=piece_cells(placement.piece,placement.pose);
    result.locked_above=true;
    for (int i=0; i<4; ++i) {
        const Cell c=cells[i];
        if (c.x<0 || c.x>=WIDTH || c.y<0 || c.y>=HEIGHT) {
            result.error="Placement leaves the 10x40 board"; return result;
        }
        result.delta.placed_cells[i]=uint16_t(c.y*WIDTH+c.x);
        result.locked_above=result.locked_above && c.y>=visible_height;
    }
    for (const Cell c:cells) board.rows[c.y] |= uint16_t(1u<<c.x);
    int destination=0;
    for (int y=0; y<HEIGHT; ++y) {
        if (board.rows[y]==FULL_ROW && board.solid[y]==0) {
            result.delta.clear_mask |= uint64_t(1)<<y;
            ++result.lines;
            if (board.garbage[y] || board.solid[y]) ++result.garbage_rows;
            continue;
        }
        board.rows[destination]=board.rows[y];
        board.garbage[destination]=board.garbage[y];
        board.solid[destination]=board.solid[y];
        result.delta.row_source[destination]=int8_t(y);
        ++destination;
    }
    for (int y=destination; y<HEIGHT; ++y) {
        board.rows[y]=board.garbage[y]=board.solid[y]=0;
        result.delta.row_source[y]=-1;
    }
    result.delta.clears=result.lines;
    result.perfect_clear=true;
    for (int y=0; y<HEIGHT; ++y)
        result.perfect_clear=result.perfect_clear && !(board.rows[y] & ~board.solid[y]);
    result.valid=true;
    return result;
}

bool insert_garbage(Board& board, const std::vector<TankRow>& incoming, BoardDelta& delta) {
    bool blocked=false;
    for (const TankRow& row:incoming) {
        if (row.hole<0 || row.hole>=WIDTH || row.width<1 || row.width>WIDTH)
            throw std::invalid_argument("Garbage hole/width outside board");
        // Official PushLine refuses a FULL top row; partial top cells can leave
        // the buffer. Do not substitute the old surrogate's overflow policy.
        if (board.rows[HEIGHT-1]==FULL_ROW) { blocked=true; continue; }
        int floor=0;
        for (int y=0; y<HEIGHT; ++y) if (board.solid[y]) floor=y+1;
        // Preserve the source's negative-splice edge for a partial permanent
        // top row: after shift(), splice(-1,0,row) inserts above the bottom row.
        if (floor>=HEIGHT) floor=1;
        for (int y=HEIGHT-1; y>floor; --y) {
            board.rows[y]=board.rows[y-1];
            board.garbage[y]=board.garbage[y-1];
            board.solid[y]=board.solid[y-1];
            delta.row_source[y]=delta.row_source[y-1];
        }
        const int end=std::min(WIDTH,row.hole+row.width);
        const int begin=end-row.width;
        const uint16_t holes=uint16_t(((1u<<row.width)-1)<<begin);
        const uint16_t occupied=FULL_ROW ^ holes;
        board.rows[floor]=board.garbage[floor]=occupied;
        // Packet hardening protects cancellation; it is not permanent board material.
        board.solid[floor]=0;
        delta.row_source[floor]=-1;
        ++delta.garbage;
    }
    return blocked;
}

std::vector<Action> enumerate_actions(const State& state, const Rules& rules, bool allow_hold) {
    std::vector<Action> actions;
    if (state.dead || state.current==Piece::None) return actions;
    if (!spawn_pose(state.board,state.current,rules,state.last_clear)) return actions;
    auto append=[&](Piece piece,bool hold) {
        auto placements=generate_placements(state.board,piece,rules,state.last_clear);
        actions.reserve(actions.size()+placements.size());
        for (auto& placement:placements) actions.push_back({std::move(placement),hold,0});
    };
    append(state.current,false);
    if (allow_hold && !state.hold_used) {
        if (state.hold!=Piece::None && state.hold!=state.current) append(state.hold,true);
        else if (state.hold==Piece::None && !state.queue.empty()) append(state.queue.front(),true);
    }
    return actions;
}

Piece parse_piece(const std::string& value) {
    if (value.empty() || value=="none" || value=="null" || value=="?") return Piece::None;
    if (value.size()!=1) throw std::invalid_argument("Expected a tetromino name I/O/T/L/J/S/Z");
    const char c=char(std::toupper(static_cast<unsigned char>(value[0])));
    const std::string pieces="IOTLJSZ";
    const auto index=pieces.find(c);
    if (index==std::string::npos) throw std::invalid_argument("Unknown tetromino: "+value);
    return Piece(int(index));
}
std::string piece_name(Piece piece) {
    if (piece==Piece::None) return "none";
    if (int(piece)<0 || int(piece)>6) throw std::invalid_argument("Invalid tetromino enum");
    return std::string(1,"IOTLJSZ"[int(piece)]);
}
std::string spin_name(Spin spin) {
    switch(spin) {
        case Spin::None:return "none";case Spin::Mini:return "mini";case Spin::Full:return "full";
    }
    throw std::invalid_argument("Invalid spin enum");
}
SpinMode parse_spin_mode(const std::string& value) {
    const std::array<std::string,10> names{{"none","T-spins","T-spins+","all","all+",
        "all-mini","all-mini+","mini-only","handheld","stupid"}};
    for (size_t i=0; i<names.size(); ++i) if (value==names[i]) return SpinMode(i);
    throw std::invalid_argument("Unknown spin bonus mode: "+value);
}
std::string spin_mode_name(SpinMode mode) {
    const std::array<std::string,10> names{{"none","T-spins","T-spins+","all","all+",
        "all-mini","all-mini+","mini-only","handheld","stupid"}};
    if (size_t(mode)>=names.size()) throw std::invalid_argument("Invalid spin bonus enum");
    return names[size_t(mode)];
}
std::string move_name(Move move) {
    switch(move) {
        case Move::Left:return "left";case Move::Right:return "right";case Move::Down:return "down";
        case Move::CW:return "cw";case Move::CCW:return "ccw";case Move::Half:return "180";
    }
    throw std::invalid_argument("Invalid geometric move enum");
}
} // namespace tetrio
