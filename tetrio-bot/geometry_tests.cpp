#include "engine.hpp"
#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace tetrio;
using json=nlohmann::json;

// An isolated geometry target can be built before the rest of the environment.
#ifdef TETRIO_GEOMETRY_STANDALONE
bool tetrio::Board::occupied(int x,int y) const {
    return x<0 || x>=WIDTH || y<0 || y>=HEIGHT || ((rows[y]>>x)&1u);
}
#endif

namespace {
void require(bool condition,const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
bool same_pose(const Pose& a,const Pose& b) {
    return a.x==b.x && a.y==b.y && a.rotation==b.rotation;
}
std::string fingerprint(const Placement& p) {
    auto cells=piece_cells(p.piece,p.pose);
    std::array<int,4> ids{};
    for(int i=0;i<4;++i) ids[i]=cells[i].y*WIDTH+cells[i].x;
    std::sort(ids.begin(),ids.end());
    std::ostringstream out;
    out<<int(p.spin);
    for(int id:ids) out<<':'<<id;
    return out.str();
}
void check_path(const Board& b,Piece piece,const Rules& rules,const Placement& p,bool last_clear=false) {
    auto spawn=spawn_pose(b,piece,rules,last_clear);
    require(spawn.has_value(),"Returned placement without a legal spawn");
    Pose pose=*spawn;
    Spin spin=Spin::None;
    bool last_rotation=false,half=false;
    int kick=-1;
    for(Move move:p.path) {
        if(move==Move::Left || move==Move::Right || move==Move::Down) {
            if(move==Move::Left) --pose.x;
            else if(move==Move::Right) ++pose.x;
            else --pose.y;
            require(!collides(b,piece,pose),"Witness translation collides");
            spin=Spin::None; last_rotation=false; half=false; kick=-1;
        } else {
            auto result=rotate_pose(b,piece,pose,rules,move);
            require(result.has_value(),"Witness rotation is not legal");
            pose=result->pose; kick=result->kick_index;
            last_rotation=true; half=move==Move::Half;
            spin=classify_spin(b,piece,pose,rules,kick,half);
        }
    }
    require(same_pose(pose,p.pose),"Witness does not end at returned pose");
    require(grounded(b,piece,pose),"Returned placement is not grounded");
    require(spin==p.spin && kick==p.kick_index && half==p.rotated180 && last_rotation==p.last_rotation,
            "Rotation/spin provenance disagrees with the witness");
}

void regression_tests() {
    Board empty;
    Rules rules;
    const int empty_counts[7]={17,9,34,34,34,17,17};
    std::size_t replayed=0;
    for(int p=0;p<7;++p) {
        auto placements=generate_placements(empty,Piece(p),rules);
        require(int(placements.size())==empty_counts[p],"Incorrect empty-board placement count for piece "+std::to_string(p));
        for(const auto& placement:placements) { check_path(empty,Piece(p),rules,placement); ++replayed; }
    }

    Board clutch;
    clutch.rows[22]=1u<<4;
    require(!spawn_pose(clutch,Piece::O,rules,false),"Blocked spawn allowed without a preceding clear");
    auto rescued=spawn_pose(clutch,Piece::O,rules,true);
    require(rescued && rescued->y==24,"Clutch did not raise the blocked spawn correctly");
    rules.clutch=false;
    require(!spawn_pose(clutch,Piece::O,rules,true),"Disabled clutch still rescued a spawn");
    rules.clutch=true;
    for(const auto& p:generate_placements(clutch,Piece::O,rules,true)) {
        check_path(clutch,Piece::O,rules,p,true); ++replayed;
    }

    // Integer occupancy and phase-conservative generated actions are distinct
    // only at the topmost hidden row. Do not promise a frame-sensitive kick.
    require(!collides(empty,Piece::I,{3,40,0}),"Integer geometry lost its 40th board row");
    require(!rotate_pose(empty,Piece::I,{3,40,0},rules,Move::CW),"Ceiling phase was ignored");
    Board ceiling;
    for(int y=21;y<=37;++y) ceiling.rows[y]=(1u<<4)|(1u<<5);
    require(!collides(ceiling,Piece::O,{4,39,0}),"Ceiling fixture is not an integer-valid pose");
    require(!spawn_pose(ceiling,Piece::O,rules,true),"Clutch emitted a fraction-sensitive ceiling placement");

    Board mini;
    mini.rows[2]=1u<<3;
    mini.rows[0]=(1u<<3)|(1u<<5);
    const Pose tpose{3,2,0};
    rules.spin_mode=SpinMode::TSpins;
    require(classify_spin(mini,Piece::T,tpose,rules,-1,false)==Spin::Mini,"Three-corner mini failed");
    require(classify_spin(mini,Piece::T,tpose,rules,3,false)==Spin::Full,"Fourth kick did not promote a mini");
    require(classify_spin(mini,Piece::T,tpose,rules,3,true)==Spin::Full,"180 fourth kick promotion differs from official engine");
    Board half_promote;
    const uint16_t half_rows[9]={0,0,58,844,80,331,726,520,669};
    std::copy(std::begin(half_rows),std::end(half_rows),half_promote.rows.begin());
    const auto half_rotation=rotate_pose(half_promote,Piece::T,{4,6,3},rules,Move::Half);
    require(half_rotation && same_pose(half_rotation->pose,{4,8,1}) && half_rotation->kick_index==3,
            "Actual 180 fourth-kick fixture no longer rotates as the official client does");
    require(classify_spin(half_promote,Piece::T,half_rotation->pose,rules,-1,true)==Spin::Mini &&
            classify_spin(half_promote,Piece::T,half_rotation->pose,rules,half_rotation->kick_index,true)==Spin::Full,
            "Actual 180 fourth kick did not upgrade the corner mini");
    Board full=mini;
    full.rows[2]|=1u<<5;
    require(classify_spin(full,Piece::T,tpose,rules,-1,false)==Spin::Full,"Both front corners did not produce a full spin");
    rules.spin_mode=SpinMode::MiniOnly;
    require(classify_spin(full,Piece::T,tpose,rules,3,false)==Spin::Mini,"Mini-only failed to downgrade a full T-spin");

    Board trapped_t;
    trapped_t.rows[3]=1u<<4;
    trapped_t.rows[1]=(1u<<2)|(1u<<6);
    trapped_t.rows[0]=1u<<4;
    const Spin expected[10]={Spin::None,Spin::None,Spin::Mini,Spin::None,Spin::Mini,
                             Spin::None,Spin::Mini,Spin::Mini,Spin::None,Spin::Full};
    for(int mode=0;mode<10;++mode) {
        rules.spin_mode=SpinMode(mode);
        require(classify_spin(trapped_t,Piece::T,tpose,rules,-1,false)==expected[mode],
                "T immobility fallback differs between a plus and non-plus rule");
    }

    Board trapped_o;
    trapped_o.rows[0]=(1u<<2)|(1u<<5);
    trapped_o.rows[2]=1u<<3;
    const Pose opose{3,1,0};
    rules.spin_mode=SpinMode::All;
    require(classify_spin(trapped_o,Piece::O,opose,rules,-1,false)==Spin::Full,"O all-spin immobility failed");
    rules.spin_mode=SpinMode::AllMiniPlus;
    require(classify_spin(trapped_o,Piece::O,opose,rules,-1,false)==Spin::Mini,"Non-T all-mini+ classification failed");
    auto orotation=rotate_pose(trapped_o,Piece::O,opose,rules,Move::CW);
    require(orotation && orotation->kick_index==-1,"O rotation was incorrectly disabled or displaced");

    std::mt19937 rng(0x918);
    for(int sample=0;sample<24;++sample) {
        Board board;
        for(int x=0;x<WIDTH;++x) {
            const int height=int(rng()%15);
            for(int y=0;y<height;++y) if((rng()%8)!=0) board.rows[y]|=1u<<x;
        }
        rules.spin_mode=SpinMode(sample%10);
        rules.kicks=sample%2 ? KickSystem::SRS : KickSystem::SRSPlus;
        rules.spin_180=(sample%3)!=0;
        for(int p=0;p<7;++p) {
            std::set<std::string> unique;
            for(const auto& placement:generate_placements(board,Piece(p),rules)) {
                check_path(board,Piece(p),rules,placement); ++replayed;
                require(unique.insert(fingerprint(placement)).second,"Duplicate footprint/spin placement");
                if(!rules.spin_180)
                    require(std::find(placement.path.begin(),placement.path.end(),Move::Half)==placement.path.end(),
                            "Disabled 180 rotation appeared in a witness");
            }
        }
    }
    std::cout<<json{{"geometry_regressions","passed"},{"placement_paths_replayed",replayed}}.dump()<<'\n';
}

json probe(const json& j) {
    Board board;
    if(j.contains("rows")) board.rows=j.at("rows").get<std::array<uint16_t,HEIGHT>>();
    Rules rules;
    rules.spin_mode=SpinMode(j.value("mode",6));
    rules.kicks=KickSystem(j.value("kicks",0));
    rules.spin_180=j.value("spin_180",true);
    rules.clutch=j.value("clutch",true);
    rules.visible_height=j.value("visible_height",20);
    const Piece piece=Piece(j.at("piece").get<int>());
    const auto pose_json=j.value("pose",std::array<int,3>{3,22,0});
    const Pose pose{pose_json[0],pose_json[1],pose_json[2]};
    const std::string op=j.at("op");
    auto as_pose=[](const Pose& p) { return json::array({p.x,p.y,p.rotation}); };
    if(op=="collision") return collides(board,piece,pose);
    if(op=="cells") {
        json result=json::array();
        for(const auto& cell:piece_cells(piece,pose)) result.push_back({cell.x,cell.y});
        return result;
    }
    if(op=="spin") return int(classify_spin(board,piece,pose,rules,j.value("kick",-1),j.value("half",false)));
    if(op=="rotate") {
        auto result=rotate_pose(board,piece,pose,rules,Move(j.at("move").get<int>()));
        if(!result) return nullptr;
        return json{{"pose",as_pose(result->pose)},{"kick",result->kick_index}};
    }
    if(op=="spawn") {
        auto result=spawn_pose(board,piece,rules,j.value("last_clear",false));
        return result ? as_pose(*result) : json(nullptr);
    }
    if(op=="placements") {
        json result=json::array();
        for(const auto& p:generate_placements(board,piece,rules,j.value("last_clear",false))) {
            json path=json::array();
            for(Move m:p.path) path.push_back(int(m));
            result.push_back({{"pose",as_pose(p.pose)},{"spin",int(p.spin)},{"kick",p.kick_index},
                              {"half",p.rotated180},{"last_rotation",p.last_rotation},{"path",path}});
        }
        return result;
    }
    throw std::invalid_argument("Unknown geometry probe operation");
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc>1 && std::string(argv[1])=="--probe") {
            for(std::string line;std::getline(std::cin,line);) {
                if(line.empty()) continue;
                std::cout<<probe(json::parse(line)).dump()<<'\n';
            }
        } else regression_tests();
    } catch(const std::exception& e) {
        std::cerr<<"Geometry test failure: "<<e.what()<<'\n';
        return EXIT_FAILURE;
    }
}
