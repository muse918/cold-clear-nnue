#include "protocol.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace tetrio;
namespace {
int checks=0;
void check(bool yes,const char* description) {++checks;if(!yes)throw std::runtime_error(description);}
template<class F>void rejects(F&& f,const char* description) {
    bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected,description);
}
}
int main() {
    try {
        const Rules league=Rules::tetra_league();
        check(to_json_value(parse_rules(to_json_value(league)))==to_json_value(league),"all rules roundtrip");
        const auto custom=parse_rules({{"spin_mode","all+"},{"kicks","SRS"},{"combo_mode","modern"},{"garbage_rounding","random"}});
        check(custom.spin_mode==SpinMode::AllPlus&&custom.kicks==KickSystem::SRS&&custom.combo_mode==ComboMode::Modern&&
              custom.garbage_rounding==GarbageRounding::Random,"custom modes parsed");
        rejects([&]{parse_rules({{"spinmod","all"}});},"unknown rule rejected");
        rejects([&]{parse_rules({{"preview_count",5.5}});},"fractional integer rejected");
        rejects([&]{parse_rules({{"visible_height",30}});},"unsupported dimensions rejected");
        rejects([&]{parse_rules({{"messiness_change",1.1}});},"invalid probabilities rejected");

        Json snapshot={{"current","t"},{"hold","I"},{"queue",{"J","L","O","S","Z"}},
            {"board",{{"rows",{1021,513,1}},{"garbage",{1021,1}},{"solid",{0,512}}}},
            {"hold_used",true},{"last_clear",true},{"combo",3},{"b2b",8},{"pieces_placed",300},
            {"total_sent",108.5},{"frame",12000},{"last_tank",11995},{"current_multiplier",1.16},
            {"current_garbage_cap",8},{"rng_seed",12345},{"last_hole",4},{"hole_changed",true},
            {"enemies",2},{"garbage_bonus",1.5},{"incoming",{
                {{"amount",5},{"hole",nullptr},{"width",2},{"active",false},{"spawn_ready",true},{"hardened",true},{"shielded",false},{"id",7}},
                {{"amount",3},{"hole",9},{"active",true},{"spawn_ready",false},{"shielded",true},{"id",8}}}}};
        State state=parse_state(snapshot,league);
        check(state_key(state)==state_key(parse_state(to_json_value(state),league)),"complete authoritative state roundtrip");
        check(state.board.rows[0]==1021&&state.board.rows[39]==0&&state.current==Piece::T,"bottom-first masks and lowercase input");
        Json bad=snapshot;bad.erase("current_multiplier");
        rejects([&]{parse_state(bad,league);},"late snapshot must supply multiplier");
        bad=snapshot;bad["queue"].push_back("I");
        rejects([&]{parse_state(bad,league);},"hidden sixth preview rejected");
        bad=snapshot;bad["queue"][0]=nullptr;
        rejects([&]{parse_state(bad,league);},"unknown marker inside observed queue rejected");
        bad=snapshot;bad["board"]["garbage"][1]=2;
        rejects([&]{parse_state(bad,league);},"garbage provenance outside occupancy rejected");
        bad=snapshot;bad["combo"]=-1;
        rejects([&]{parse_state(bad,league);},"Triangle negative sentinel rejected");
        bad=snapshot;bad["last_hole"]=-1;
        rejects([&]{parse_state(bad,league);},"hole state consistency");
        bad=snapshot;bad["current_multiplier"]=-.5;
        rejects([&]{parse_state(bad,league);},"negative fractional multiplier rejected");
        bad=snapshot;bad["frame"]=std::numeric_limits<uint64_t>::max();
        rejects([&]{parse_state(bad,league);},"large unsigned integer cannot wrap");
        check(parse_state({{"current",nullptr}},league).current==Piece::None,"explicit unknown root accepted");
        rejects([&]{parse_search_options({{"max_cache_nodes",1}});},"invalid cache rejected before search");
        rejects([&]{parse_search_options({{"max_depth",0}});},"invalid depth rejected");

        Protocol session;
        auto info=session.handle({{"cmd","info"}});
        check(info["effective_evaluator"]=="heuristic"&&info["protocol_version"]==1,"model-free fallback reports itself");
        session.handle({{"cmd","configure"},{"search",{{"max_depth",2},{"beam_width",8},{"max_nodes",1000},{"time_budget_ms",0}}}});
        Json initial={{"current","T"},{"queue",{"I","O","S","Z","J"}}};
        session.handle({{"cmd","state"},{"state",initial}});
        auto result=session.handle({{"cmd","search"},{"id","move1"}});
        check(result["action"].is_object()&&result["id"]=="move1","search returns placement and correlation id");
        auto start=parse_state(session.handle({{"cmd","state"}})["state"],league);
        auto canonical=match_action(result["action"],start,league);
        auto expected=apply_action(start,canonical,league);
        Json forged=result["action"];forged["spin"]="full";
        rejects([&]{session.handle({{"cmd","apply"},{"action",forged}});},"fabricated spin rejected on the SAME original board");
        check(session.handle({{"cmd","state"}})["state"]==to_json_value(start),"failed apply leaves original state unchanged");
        auto applied=session.handle({{"cmd","apply"},{"action",result["action"]}});
        check(applied["state"]==to_json_value(expected.state)&&applied["events"]["lines"]==expected.lines,"protocol apply equals native core");
        const auto before=session.handle({{"cmd","state"}})["state"];
        rejects([&]{session.handle({{"cmd","actions"},{"state",initial},{"allow_hold",1}});},"invalid action listing flag rejected");
        check(session.handle({{"cmd","state"}})["state"]==before,"failed action listing leaves state unchanged");
        rejects([&]{session.handle({{"cmd","configure"},{"rules",{{"preview_count",0}}}});},"incompatible live preview config rejected");
        check(session.handle({{"cmd","info"}})["rules"]["preview_count"]==5,"configure failure is atomic");
        rejects([&]{session.handle({{"cmd","configure"},{"evaluation",{{"mode","nnue"}}}});},"NNUE without model rejected");
        check(session.handle({{"cmd","search"}})["action"].is_object(),"session usable after invalid configuration");
        session.handle({{"cmd","enqueue"},{"packet",{{"amount",2},{"hole",4},{"active",true},{"spawn_ready",false}}}});
        auto queued=session.handle({{"cmd","state"}})["state"];
        check(queued["incoming"][0]["spawn_ready"]==false,"caller supplied readiness preserved");
        auto advanced=session.handle({{"cmd","advance"},{"frames",30}})["state"];
        check(advanced["frame"]==queued["frame"].get<int64_t>()+30,"external elapsed frames advance independently");
        auto prior=session.handle({{"cmd","state"}})["state"];
        rejects([&]{session.handle({{"cmd","enqueue"},{"packet",{{"amount",0}}}});},"invalid packet rejected");
        check(session.handle({{"cmd","state"}})["state"]==prior,"failed enqueue is atomic");
        session.handle({{"cmd","state"},{"state",{{"current",nullptr},{"queue",Json::array()}}}});
        check(session.handle({{"cmd","search"}})["action"].is_null(),"unknown current does not sample a piece");
        std::cout<<"protocol_tests: "<<checks<<" checks passed\n";
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
