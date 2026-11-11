// Reproducible hard-drop dataset for spatial NNUE architecture experiments.
// Scope: seven-bag, no hold, no spin rewards, no timed input simulation, and
// no claim to reproduce the full TETR.IO attack/topout system. Piece coordinates
// follow libtetris/src/piece.rs at repository commit 31713b0.
// Row storage is bottom-first. Clears compact complete rows as indivisible rows.
// Labels come from the explicitly documented two-placement heuristic teacher.
// Build: g++ -O3 -std=c++17 generate.cpp -o generate
// Run:   ./generate --out data/train --games 96 --steps 44 --seed 91820261111

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int W = 10, H = 40;
constexpr std::uint16_t FULL = (1u << W) - 1;
using Board = std::array<std::uint16_t, H>;

// NumPy can read this header-free format directly with align=False.
// All multibyte values are little-endian; the executable checks host byte order.
#pragma pack(push, 1)
struct Record {
    std::uint32_t game_id, parent_id;
    std::uint16_t step;
    std::uint8_t event;       // 0=piece lock, 1=garbage rise
    std::int8_t piece;        // I,O,T,L,J,S,Z = 0..6; -1 for garbage
    std::uint8_t next;        // piece used for value_target's next-ply expansion
    std::uint8_t clears, garbage, selected;
    std::uint64_t clear_mask; // original row indices after placement, before clear
    float reward, value_target;
    std::uint16_t parent_rows[H], child_rows[H];
    std::int8_t row_source[H]; // child row -> parent row; -1 for newly inserted row
    std::uint16_t placed_cells[4]; // preclear 10*y+x; 65535 for garbage
};
#pragma pack(pop)
static_assert(sizeof(Record) == 240, "Record schema must stay 240 bytes");

struct Rng {
    std::uint64_t state;
    std::uint64_t operator()() {
        std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int below(int n) { return static_cast<int>((*this)() % std::uint64_t(n)); }
};

struct Cell { int x, y; };
struct Shape {
    std::array<Cell, 4> cells;
    int width, height;
    std::array<int, 4> column_bottom;
};
std::array<std::vector<Shape>, 7> shapes;

void initialize_shapes() {
    // These are the seven North footprints from libtetris/src/piece.rs.
    const std::array<std::array<Cell, 4>, 7> north{{
        {{{-1,0},{0,0},{1,0},{2,0}}},
        {{{0,0},{1,0},{0,1},{1,1}}},
        {{{-1,0},{0,0},{1,0},{0,1}}},
        {{{-1,0},{0,0},{1,0},{1,1}}},
        {{{-1,0},{0,0},{1,0},{-1,1}}},
        {{{-1,0},{0,0},{0,1},{1,1}}},
        {{{-1,1},{0,1},{0,0},{1,0}}}
    }};
    for (int p = 0; p < 7; ++p) {
        for (int rot = 0; rot < 4; ++rot) {
            auto cells = north[p];
            for (auto &c : cells) {
                const int x = c.x, y = c.y;
                // Same N,S,E,W transforms as the original Rust macro.
                if (rot == 1) c = {-x,-y};
                if (rot == 2) c = {y,-x};
                if (rot == 3) c = {-y,x};
            }
            int minx = 9, miny = 9;
            for (auto c : cells) { minx = std::min(minx,c.x); miny = std::min(miny,c.y); }
            for (auto &c : cells) { c.x -= minx; c.y -= miny; }
            std::sort(cells.begin(),cells.end(),[](Cell a,Cell b){
                return a.y == b.y ? a.x < b.x : a.y < b.y;
            });
            bool duplicate = false;
            for (const auto &s : shapes[p]) {
                bool same = true;
                for (int i = 0; i < 4; ++i)
                    same = same && cells[i].x == s.cells[i].x && cells[i].y == s.cells[i].y;
                duplicate = duplicate || same;
            }
            if (duplicate) continue;
            Shape s{cells,0,0,{9,9,9,9}};
            for (auto c : cells) {
                s.width = std::max(s.width,c.x+1);
                s.height = std::max(s.height,c.y+1);
                s.column_bottom[c.x] = std::min(s.column_bottom[c.x],c.y);
            }
            shapes[p].push_back(s);
        }
    }
    const int expected[7] = {2,1,4,4,4,2,2};
    for (int p=0;p<7;++p) assert(int(shapes[p].size())==expected[p]);
}

void ensure_shapes() {
    // C++ guarantees thread-safe initialization of this function-local static.
    static const bool ready=[](){initialize_shapes();return true;}();
    (void)ready;
}

std::array<int,W> heights(const Board &b) {
    std::array<int,W> h{};
    for (int x=0;x<W;++x)
        for (int y=H-1;y>=0;--y)
            if (b[y] & (1u<<x)) { h[x]=y+1; break; }
    return h;
}
int max_height(const Board &b) {
    for (int y=H-1;y>=0;--y) if (b[y]) return y+1;
    return 0;
}
int occupied(const Board &b) {
    int n=0; for (auto r:b) n+=__builtin_popcount(unsigned(r)); return n;
}

// Deliberately small, transparent heuristic teacher; not Cold Clear's full
// Standard evaluator and not a claim of competitive TETR.IO strength.
float heuristic(const Board &b) {
    const auto h=heights(b);
    int sum=0, high=0, holes=0, covered=0, bump=0, wells=0;
    for (int x=0;x<W;++x) {
        sum+=h[x]; high=std::max(high,h[x]);
        int above=0;
        for (int y=h[x]-1;y>=0;--y) {
            if (b[y] & (1u<<x)) ++above;
            else { ++holes; covered+=above; }
        }
        if (x+1<W) bump+=std::abs(h[x]-h[x+1]);
        int left=x? h[x-1] : H, right=x+1<W? h[x+1] : H;
        int depth=std::max(0,std::min(left,right)-h[x]);
        wells+=depth*(depth+1)/2;
    }
    if (high>20) return -400.0f-40.0f*(high-20)-4.0f*holes;
    const int danger=std::max(0,high-12);
    return -0.45f*sum-0.65f*high-4.0f*holes-0.24f*bump
           -0.18f*wells-0.08f*covered-2.0f*danger*danger;
}
float line_reward(int n, const Board &b) {
    static constexpr float REWARD[5]={0,6,18,30,48};
    return REWARD[n]+(n && max_height(b)==0 ? 12.0f : 0.0f);
}

struct Candidate {
    Board child;
    std::array<std::int8_t,H> source;
    std::array<std::uint16_t,4> cells;
    std::uint64_t clear_mask;
    int clears;
    float reward, target;
};

std::vector<Candidate> placements(const Board &b, int piece) {
    std::vector<Candidate> out;
    out.reserve(34);
    const auto h=heights(b);
    for (const auto &s : shapes[piece]) {
        for (int x=0;x<=W-s.width;++x) {
            int y=0;
            for (int dx=0;dx<s.width;++dx)
                if(s.column_bottom[dx]!=9) y=std::max(y,h[x+dx]-s.column_bottom[dx]);
            if (y+s.height>H) continue;
            Candidate c{};
            Board locked=b;
            for (int i=0;i<4;++i) {
                const int px=x+s.cells[i].x, py=y+s.cells[i].y;
                if (locked[py]&(1u<<px)) throw std::logic_error("harddrop collision");
                locked[py]|=std::uint16_t(1u<<px);
                c.cells[i]=std::uint16_t(W*py+px);
            }
            c.source.fill(-1);
            int dst=0;
            for (int sy=0;sy<H;++sy) {
                if(locked[sy]==FULL) {c.clear_mask|=1ULL<<sy; ++c.clears;}
                else {c.child[dst]=locked[sy]; c.source[dst]=std::int8_t(sy); ++dst;}
            }
            if(c.clears>4) throw std::logic_error("more than four cleared rows");
            c.reward=line_reward(c.clears,c.child);
            c.target=0;
            out.push_back(c);
        }
    }
    return out;
}

float continuation(const Board &b,int next) {
    if(max_height(b)>20) return heuristic(b);
    auto moves=placements(b,next);
    float best=-10000;
    for(const auto &c:moves) best=std::max(best,c.reward+heuristic(c.child));
    return best;
}

struct Stats {
    std::uint64_t records=0, lock_candidates=0, garbage_records=0, parent_groups=0;
    std::uint64_t clear_candidates[5]{}, selected_clears[5]{};
    std::uint64_t occupied_before=0,occupied_after=0,cell_xor=0, selected_locks=0;
    std::uint64_t garbage_rows=0,overflow_events=0,terminated_games=0;
    std::uint64_t clean_records=0,garbage_condition_records=0;
    int max_cell_xor=0;
};

void write_record(std::ofstream &os,const Record &r,Stats &stats,bool garbage_condition) {
    os.write(reinterpret_cast<const char*>(&r),sizeof(r));
    if(!os) throw std::runtime_error("failed to write record");
    ++stats.records;
    if(garbage_condition) ++stats.garbage_condition_records; else ++stats.clean_records;
    int changes=0;
    for(int y=0;y<H;++y) {
        stats.occupied_before+=__builtin_popcount(unsigned(r.parent_rows[y]));
        stats.occupied_after+=__builtin_popcount(unsigned(r.child_rows[y]));
        changes+=__builtin_popcount(unsigned(r.parent_rows[y]^r.child_rows[y]));
    }
    stats.cell_xor+=changes;stats.max_cell_xor=std::max(stats.max_cell_xor,changes);
    if(r.event==0) {
        ++stats.lock_candidates;++stats.clear_candidates[r.clears];
        if(r.selected) {++stats.selected_locks;++stats.selected_clears[r.clears];}
    } else {++stats.garbage_records;stats.garbage_rows+=r.garbage;}
}

void copy_boards(Record &r,const Board &before,const Board &after) {
    std::copy(before.begin(),before.end(),r.parent_rows);
    std::copy(after.begin(),after.end(),r.child_rows);
}

struct Config {
    std::string out="experiments/spatial-nnue/data/train";
    std::uint64_t seed=91820261111ULL;
    int games=96,steps=44,start_game=0;
};

Config parse_args(int argc,char**argv) {
    Config c;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"generate [--out DIR] [--games 96] [--steps 44] [--seed 91820261111] [--start-game 0]\n";
            std::exit(0);
        }
        if(i+1==argc) throw std::runtime_error("missing argument for "+key);
        std::string v=argv[++i];
        if(key=="--out")c.out=v;
        else if(key=="--games")c.games=std::stoi(v);
        else if(key=="--steps")c.steps=std::stoi(v);
        else if(key=="--seed")c.seed=std::stoull(v);
        else if(key=="--start-game")c.start_game=std::stoi(v);
        else throw std::runtime_error("unknown argument "+key);
    }
    if(c.games<1||c.steps<1||c.steps>60000||c.start_game<0)
        throw std::runtime_error("invalid game/step bounds");
    return c;
}

template<class T> void json_array(std::ostream &os,const T *v,int n) {
    os<<'[';for(int i=0;i<n;++i){if(i)os<<',';os<<v[i];}os<<']';
}

void write_metadata(const Config &c,const Stats &s) {
    std::ofstream o(std::filesystem::path(c.out)/"metadata.json");
    o<<std::setprecision(10)
     <<"{\n  \"format\": \"spatial-nnue-harddrop-v1\",\n"
     <<"  \"record_bytes\": 240, \"endianness\": \"little\",\n"
     <<"  \"source_commit\": \"31713b0bd20c0bddf9dfa11daacdbd2d67888ae4\",\n"
     <<"  \"scope\": \"harddrop from sky; normalized unique footprints; seven-bag; no hold/spins/timing; surrogate teacher; not full TETR.IO\",\n"
     <<"  \"seed\": "<<c.seed<<", \"games\": "<<c.games<<", \"steps_limit\": "<<c.steps
     <<", \"start_game\": "<<c.start_game<<",\n"
     <<"  \"piece_order\": [\"I\",\"O\",\"T\",\"L\",\"J\",\"S\",\"Z\"],\n"
     <<"  \"conditions\": \"even game_id clean; odd game_id periodic 1..3 garbage rows at steps divisible by 9, if height<=12\",\n"
     <<"  \"garbage\": \"rows inserted at bottom, upper overflow discarded and detected; one shared random hole per rise\",\n"
     <<"  \"teacher\": {\"target\": \"V(child,next)=max_nextplacement(reward_next+heuristic(grandchild)); terminal child uses heuristic(child)\",\n"
     <<"    \"heuristic\": \"-.45*sum_height-.65*max_height-4*holes-.24*bumpiness-.18*triangular_wells-.08*covered_blocks-2*max(max_height-12,0)^2\",\n"
     <<"    \"terminal\": \"height>20 gives -400-40*(height-20)-4*holes; simplified surrogate condition\",\n"
     <<"    \"line_reward\": [0,6,18,30,48], \"perfect_clear_bonus\":12,\n"
     <<"    \"selection\": \"88% argmax reward+target, 10% uniform among top5, 2% uniform all placements\"},\n"
     <<"  \"records\": "<<s.records<<", \"parent_groups\": "<<s.parent_groups<<",\n"
     <<"  \"lock_candidates\": "<<s.lock_candidates<<", \"selected_locks\": "<<s.selected_locks<<",\n"
     <<"  \"clear_candidate_histogram\": ";json_array(o,s.clear_candidates,5);
    o<<",\n  \"selected_clear_histogram\": ";json_array(o,s.selected_clears,5);
    o<<",\n  \"clear_candidate_fraction\": "<<(s.lock_candidates?double(s.lock_candidates-s.clear_candidates[0])/s.lock_candidates:0)
     <<", \"selected_clear_fraction\": "<<(s.selected_locks?double(s.selected_locks-s.selected_clears[0])/s.selected_locks:0)
     <<",\n  \"garbage_records\": "<<s.garbage_records<<", \"garbage_rows_total\": "<<s.garbage_rows
     <<", \"garbage_overflow_events\": "<<s.overflow_events<<",\n"
     <<"  \"clean_condition_records\": "<<s.clean_records<<", \"garbage_condition_records\": "<<s.garbage_condition_records<<",\n"
     <<"  \"terminated_before_step_limit\": "<<s.terminated_games<<",\n"
     <<"  \"mean_occupied_parent\": "<<double(s.occupied_before)/s.records
     <<", \"mean_occupied_child\": "<<double(s.occupied_after)/s.records<<",\n"
     <<"  \"mean_cell_xor\": "<<double(s.cell_xor)/s.records<<", \"max_cell_xor\": "<<s.max_cell_xor<<",\n"
     <<"  \"dtype\": [[\"game_id\",\"<u4\"],[\"parent_id\",\"<u4\"],[\"step\",\"<u2\"],[\"event\",\"u1\"],[\"piece\",\"i1\"],[\"next\",\"u1\"],[\"clears\",\"u1\"],[\"garbage\",\"u1\"],[\"selected\",\"u1\"],[\"clear_mask\",\"<u8\"],[\"reward\",\"<f4\"],[\"value_target\",\"<f4\"],[\"parent_rows\",\"<u2\",[40]],[\"child_rows\",\"<u2\",[40]],[\"row_source\",\"i1\",[40]],[\"placed_cells\",\"<u2\",[4]]]\n}\n";
    if(!o)throw std::runtime_error("failed to write metadata");
}
} // namespace

// The same geometry/teacher can be called through ctypes for closed-loop
// surrogate evaluation. Caller allocates capacity for 34 child boards, rewards,
// targets and clear counts. out_teacher_targets may be null to skip lookahead.
// Returns a candidate count, -1 for invalid arguments, -2 for a capacity invariant
// failure, and -3 for an internal exception. No exception crosses the C ABI.
extern "C" int spatial_moves(
    const std::uint16_t* board40, int piece, int next,
    std::uint16_t* out_boards, float* out_rewards,
    float* out_teacher_targets, int* out_clears) noexcept {
    if(!board40||!out_boards||!out_rewards||!out_clears
       ||piece<0||piece>=7||next<0||next>=7) return -1;
    try {
        ensure_shapes();
        Board b;
        std::copy(board40,board40+H,b.begin());
        for(auto row:b)if(row>FULL)return -1;
        auto candidates=placements(b,piece);
        if(candidates.size()>34)return -2;
        for(std::size_t i=0;i<candidates.size();++i) {
            const auto &c=candidates[i];
            std::copy(c.child.begin(),c.child.end(),out_boards+i*H);
            out_rewards[i]=c.reward;
            out_clears[i]=c.clears;
            if(out_teacher_targets)out_teacher_targets[i]=continuation(c.child,next);
        }
        return static_cast<int>(candidates.size());
    } catch(...) {return -3;}
}

extern "C" double spatial_heuristic(const std::uint16_t* board40) noexcept {
    if(!board40)return std::numeric_limits<double>::quiet_NaN();
    Board b;
    std::copy(board40,board40+H,b.begin());
    for(auto row:b)if(row>FULL)return std::numeric_limits<double>::quiet_NaN();
    return heuristic(b);
}

#ifndef SPATIAL_DATA_LIBRARY
int main(int argc,char**argv) {
    try {
        std::uint16_t endian=1;
        if(*reinterpret_cast<std::uint8_t*>(&endian)!=1)
            throw std::runtime_error("this binary writer currently requires a little-endian host");
        const auto cfg=parse_args(argc,argv);
        ensure_shapes();
        std::filesystem::create_directories(cfg.out);
        std::ofstream out(std::filesystem::path(cfg.out)/"records.bin",std::ios::binary);
        if(!out)throw std::runtime_error("cannot create records.bin");
        Stats stats;
        std::uint32_t parent=0;
        for(int gi=0;gi<cfg.games;++gi) {
            const int game=cfg.start_game+gi;
            Rng rng{cfg.seed+std::uint64_t(game)*1000003ULL};
            std::vector<int> queue;
            while(int(queue.size())<cfg.steps+2) {
                std::array<int,7> bag{{0,1,2,3,4,5,6}};
                for(int j=6;j>0;--j)std::swap(bag[j],bag[rng.below(j+1)]);
                queue.insert(queue.end(),bag.begin(),bag.end());
            }
            Board b{};
            const bool garbage_condition=(game%2)==1;
            bool terminated=false;
            for(int step=0;step<cfg.steps;++step) {
                if(garbage_condition && step>0 && step%9==0 && max_height(b)<=12) {
                    const int g=1+rng.below(3),hole=rng.below(W);
                    Board after{};
                    Record r{};r.game_id=game;r.parent_id=parent++;r.step=step;
                    r.event=1;r.piece=-1;r.next=queue[step];r.garbage=g;r.selected=1;
                    std::fill(std::begin(r.row_source),std::end(r.row_source),std::int8_t(-1));
                    std::fill(std::begin(r.placed_cells),std::end(r.placed_cells),std::uint16_t(65535));
                    for(int y=0;y<g;++y)after[y]=FULL^(1u<<hole);
                    for(int y=g;y<H;++y){after[y]=b[y-g];r.row_source[y]=std::int8_t(y-g);}
                    bool overflow=false;
                    for(int y=H-g;y<H;++y)overflow=overflow||b[y]!=0;
                    stats.overflow_events+=overflow;
                    r.value_target=continuation(after,r.next);copy_boards(r,b,after);
                    write_record(out,r,stats,true);++stats.parent_groups;b=after;
                    if(overflow){terminated=true;break;}
                }
                auto candidates=placements(b,queue[step]);
                if(candidates.empty()){terminated=true;break;}
                for(auto &c:candidates)c.target=continuation(c.child,queue[step+1]);
                std::vector<int> order(candidates.size());
                for(int i=0;i<int(order.size());++i)order[i]=i;
                std::stable_sort(order.begin(),order.end(),[&](int a,int z){
                    return candidates[a].reward+candidates[a].target>candidates[z].reward+candidates[z].target;
                });
                const int draw=rng.below(1000);
                int chosen=order[0];
                if(draw<20)chosen=rng.below(int(candidates.size()));
                else if(draw<120)chosen=order[rng.below(std::min(5,int(order.size())))];
                for(int ci=0;ci<int(candidates.size());++ci) {
                    const auto &c=candidates[ci];
                    Record r{};r.game_id=game;r.parent_id=parent;r.step=step;r.piece=queue[step];
                    r.next=queue[step+1];r.clears=c.clears;r.selected=ci==chosen;
                    r.clear_mask=c.clear_mask;r.reward=c.reward;r.value_target=c.target;
                    copy_boards(r,b,c.child);
                    std::copy(c.source.begin(),c.source.end(),r.row_source);
                    std::copy(c.cells.begin(),c.cells.end(),r.placed_cells);
                    if(occupied(c.child)!=occupied(b)+4-10*c.clears)
                        throw std::logic_error("lock mass conservation failed");
                    write_record(out,r,stats,garbage_condition);
                }
                ++parent;++stats.parent_groups;b=candidates[chosen].child;
                if(max_height(b)>20){terminated=true;break;}
            }
            stats.terminated_games+=terminated;
        }
        out.close();write_metadata(cfg,stats);
        std::cout<<"records="<<stats.records<<" lock_candidates="<<stats.lock_candidates
                 <<" selected_locks="<<stats.selected_locks<<" garbage_records="<<stats.garbage_records
                 <<" clear_candidate_fraction="<<double(stats.lock_candidates-stats.clear_candidates[0])/stats.lock_candidates
                 <<" selected_clear_fraction="<<double(stats.selected_locks-stats.selected_clears[0])/stats.selected_locks
                 <<" out="<<cfg.out<<'\n';
    } catch(const std::exception &e) {
        std::cerr<<"error: "<<e.what()<<'\n';return 1;
    }
    return 0;
}
#endif
