#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace tetrio;
namespace {
int checks=0;
void check(bool value,const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
GarbagePacket packet(int amount,int hole=-1,bool active=true,bool ready=true,int64_t id=1) {
    GarbagePacket p;p.amount=amount;p.hole=hole;p.active=active;p.spawn_ready=ready;p.id=id;return p;
}
Action first_action(const State& state,const Rules& rules,bool hold=false) {
    for (const auto& action:enumerate_actions(state,rules)) if (action.use_hold==hold) return action;
    throw std::runtime_error("fixture has no action");
}
Action clear_action(const State& state,const Rules& rules,int lines) {
    for (const auto& action:enumerate_actions(state,rules,false))
        if (apply_action(state,action,rules).lines==lines) return action;
    throw std::runtime_error("numeric fixture has no intended clear");
}
void rejected(const State& state,const Action& action,const Rules& rules,const std::string& reason) {
    const auto before=state_key(state);
    const auto result=apply_action(state,action,rules);
    check(!result.valid && result.error.find(reason)!=std::string::npos,"overflow must reject with a useful reason");
    check(state_key(result.state)==before,"rejected transition must return the unchanged parent state");
    check(result.lines==0 && result.attack==0 && result.sent==0 && result.cancelled==0 &&
          result.received==0 && result.outgoing.empty(),"rejected transition cannot expose partial battle output");
}
}

int main() {
    try {
        Rules league=Rules::tetra_league();
        league.validate();
        check(league.b2b_charge_base==0 && league.pc_garbage==5 && league.pc_b2b==1 &&
              league.allclear_b2b_dupes && league.spin_mode==SpinMode::AllMiniPlus,"League preset");
        {
            State state;state.pieces_placed=15;
            auto result=resolve_battle(state,{Piece::I,4,0,Spin::None,true},league);
            check(state.combo==1 && state.b2b==2,"PC quad gives two official B2B increments");
            check(result.outgoing==std::vector<int>({5,5}) && result.attack==10,"PC main and bonus are separate packets");
            result=resolve_battle(state,{Piece::I,4,0,Spin::None,true},league);
            check(state.combo==2 && state.b2b==4 && result.outgoing==std::vector<int>({6,5}),"consecutive PC quad");
        }
        {
            State state;state.pieces_placed=15;
            const auto result=resolve_battle(state,{Piece::I,4,0,Spin::Mini,false},league);
            check(result.attack==4,"official mini quad has base attack four");
            state.b2b=7;state.combo=4;
            const auto empty=resolve_battle(state,{Piece::T,0,0,Spin::Full,false},league);
            check(state.b2b==7 && state.combo==0 && empty.attack==0,"no clear preserves B2B and resets combo");
        }
        {
            State state;state.b2b=10;state.pieces_placed=15;
            const auto result=resolve_battle(state,{Piece::O,1,0,Spin::None,false},league);
            check(result.surge==6 && result.outgoing==std::vector<int>({2,2,2}),"surge splits into three ordered packets");
            check(state.b2b==0 && state.combo==1,"ordinary clear consumes B2B charge");
            state.b2b=7;state.combo=3;state.pieces_placed=1;state.total_sent=0;
            state.incoming={packet(3,-1,false,true,1),packet(10,4,true,true,2)};
            const auto cancel=resolve_battle(state,{Piece::O,1,0,Spin::None,false},league);
            check(cancel.surge==3 && cancel.attack==4 && cancel.cancelled==8 && cancel.sent==0,
                  "opener cancellation is applied per surge packet before main attack");
        }
        {
            State fourteen;fourteen.pieces_placed=14;fourteen.incoming={packet(8,4)};
            State fifteen=fourteen;fifteen.pieces_placed=15;
            const auto a=resolve_battle(fourteen,{Piece::I,4,0,Spin::None,false},league);
            const auto b=resolve_battle(fifteen,{Piece::I,4,0,Spin::None,false},league);
            check(a.cancelled==8 && fourteen.incoming.empty(),"opener phase includes placed piece fourteen");
            check(b.cancelled==4 && fifteen.incoming[0].amount==4,"piece fifteen no longer doubles cancellation");
            State state;state.pieces_placed=1;state.total_sent=3;state.incoming={packet(2,4)};
            const auto c=resolve_battle(state,{Piece::I,4,0,Spin::None,false},league);
            check(c.cancelled==2 && c.sent==2,"opener pending-versus-sent condition");
        }
        {
            State state;state.frame=123;state.incoming={packet(3,-1,false,true,1),packet(10,4,true,true,2)};
            const auto result=resolve_battle(state,{},league);
            check(result.received==8 && result.tank.size()==8 && state.last_tank==123,"League tanks at most eight rows");
            check(state.incoming.size()==2 && state.incoming[0].id==2 && state.incoming[0].amount==2 &&
                  state.incoming[1].id==1,"later ready packet bypasses waiting packet and queue is reordered");
            check(state.rng_seed==1 && state.last_hole==-1,"explicit partial packet holes do not consume RNG");
            state=State{};state.incoming={packet(2)};
            const auto random=resolve_battle(state,{},league);
            check(random.tank[0].hole==0 && random.tank[1].hole==0,"Park-Miller generated first packet holes");
            check(state.rng_seed==984943658 && state.last_hole==4 && state.hole_changed,
                  "zero inner probability still draws RNG and packet completion rerolls");
        }
        {
            Rules rules=league;rules.garbage_blocking=GarbageBlocking::Limited;
            State state;state.pieces_placed=15;state.incoming={packet(5,8)};
            const auto limited=resolve_battle(state,{Piece::O,1,0,Spin::None,false},rules);
            check(!limited.blocked && limited.received==5,"limited blocking tanks after a zero-attack single");
            rules.garbage_blocking=GarbageBlocking::Combo;
            state=State{};state.incoming={packet(5,8)};
            check(resolve_battle(state,{Piece::O,1,0,Spin::None,false},rules).received==0,
                  "combo blocking preserves garbage on a zero-attack single");
        }
        {
            Rules rules=league;rules.garbage_multiplier=.5;rules.attack_cap=2;
            State state;state.pieces_placed=15;
            const auto result=resolve_battle(state,{Piece::T,2,1,Spin::Full,false},rules);
            check(result.attack==2,"downstack bonus is after rounding and before attack cap");
            rules.attack_cap=0;state=State{};state.pieces_placed=15;
            check(resolve_battle(state,{Piece::T,2,1,Spin::Full,false},rules).attack==3,
                  "downstack bonus is one unscaled line");
        }
        {
            Rules rules=league;rules.receive_multiplier=.5;rules.garbage_absolute_cap=8;
            State state;state.incoming={packet(5,0)};
            check(enqueue_garbage(state,packet(10,1),rules)==3 && state.incoming.back().amount==3,
                  "receive multiplier then partial absolute cap");
            check(enqueue_garbage(state,packet(5,2),rules)==2 && state.incoming.back().shielded,
                  "already-full absolute cap retains a shielded packet until confirmation");
            State time;time.frame=10799;
            advance_frames(time,league,1);
            check(time.current_multiplier==1,"League multiplier does not increase on margin frame");
            advance_frames(time,league,60);
            check(std::abs(time.current_multiplier-1.008)<1e-12,"League multiplier increases strictly after margin");
        }
        {
            State state;state.current=Piece::O;state.queue={Piece::I,Piece::T,Piece::Z};
            const auto empty_hold=apply_action(state,first_action(state,league,true),league);
            check(empty_hold.valid && empty_hold.state.hold==Piece::O && empty_hold.state.current==Piece::T &&
                  empty_hold.state.queue==std::vector<Piece>({Piece::Z}),"empty hold consumes placement preview and next current");
            state.hold=Piece::I;state.queue={Piece::T,Piece::Z};
            const auto existing_hold=apply_action(state,first_action(state,league,true),league);
            check(existing_hold.valid && existing_hold.state.current==Piece::T && existing_hold.state.hold==Piece::O &&
                  existing_hold.state.queue==std::vector<Piece>({Piece::Z}),"existing hold swaps without extra preview consumption");
            state.hold_used=true;
            const auto actions=enumerate_actions(state,league);
            check(std::none_of(actions.begin(),actions.end(),[](const Action&a){return a.use_hold;}),"used hold is unavailable");
            state=State{};state.current=Piece::O;
            const auto horizon=apply_action(state,first_action(state,league),league);
            check(horizon.valid && !horizon.state.dead && horizon.state.current==Piece::None,
                  "end of observed preview is unknown horizon, not topout");
        }
        {
            State state;state.current=Piece::I;state.queue={Piece::O};
            for (int y=0;y<4;y++) state.board.rows[y]=FULL_ROW^1;
            bool found=false;
            for (const auto& action:enumerate_actions(state,league,false)) {
                auto result=apply_action(state,action,league);
                if (result.lines!=4) continue;
                found=true;
                check(result.valid && result.perfect_clear && result.state.board.empty(),"placement clears whole rows into a PC");
                check(result.state.b2b==2 && result.attack==10 && result.delta.clear_mask==15,
                      "board transition and official battle counters compose");
                break;
            }
            check(found,"quad fixture is reachable");
        }
        {
            State state;state.current=Piece::O;state.queue={Piece::I};state.board.rows[21]=1u<<3;
            const auto result=apply_action(state,first_action(state,league),league);
            check(result.valid && result.state.dead,"actual next spawn collision causes topout");
            state=State{};state.current=Piece::O;state.board.rows[20]=state.board.solid[20]=FULL_ROW;
            Rules rules=league;rules.lockout=true;
            auto locked=apply_action(state,first_action(state,rules),rules);
            check(locked.valid && locked.state.dead,"enabled lockout rejects an entirely above-ceiling lock with no clear");
            state.board.rows[21]=state.board.rows[22]=FULL_ROW ^ ((1u<<4)|(1u<<5));
            auto clutched=apply_action(state,first_action(state,rules),rules);
            check(clutched.valid && clutched.lines==2 && !clutched.state.dead,"clutch clear rescues above-ceiling lockout");
            rules.clutch=false;
            check(apply_action(state,first_action(state,rules),rules).state.dead,"disabled clutch cannot rescue lockout");
        }
        {
            Board board;board.rows[39]=1;BoardDelta delta;
            for (int y=0;y<HEIGHT;y++) delta.row_source[y]=int8_t(y);
            check(!insert_garbage(board,{{9,3,true,1}},delta),"partial top-row discard is not official garbage smash");
            check(board.rows[0]==127 && board.garbage[0]==127 && board.solid[0]==0,
                  "right-edge wide hole expands left; hardened is not solid material");
            board=Board{};board.rows[39]=FULL_ROW;
            check(insert_garbage(board,{{0,1,false,1}},delta) && board.rows[0]==0,
                  "full top row blocks official PushLine");
            board=Board{};board.rows[39]=board.solid[39]=1;delta=BoardDelta{};
            check(!insert_garbage(board,{{0,1,false,1}},delta) && board.rows[1]==(FULL_ROW^1),
                  "partial top solid follows official negative splice index");
        }
        {
            State single;single.current=Piece::I;single.queue={Piece::O};single.pieces_placed=20;
            single.board.rows[0]=FULL_ROW ^ (uint16_t(15)<<3);single.board.rows[7]=1u<<9;
            const auto one=clear_action(single,league,1);
            State quad;quad.current=Piece::I;quad.queue={Piece::O};quad.pieces_placed=20;
            for (int y=0;y<4;y++) quad.board.rows[y]=FULL_ROW^1;
            quad.board.rows[7]=1u<<9;
            const auto four=clear_action(quad,league,4);
            State pc=quad;pc.board.rows[7]=0;
            const auto perfect=clear_action(pc,league,4);

            // These combined values are individually accepted by the public
            // snapshot/configuration schema, but their result exceeds int32.
            State state=single;state.b2b=1000000;state.current_multiplier=1000000;
            rejected(state,one,league,"B2B surge");
            Rules rules=league;rules.pc_garbage=2147;rules.garbage_multiplier=1000000;
            rejected(pc,perfect,rules,"total attack");
            rules=league;rules.garbage_multiplier=10000;rules.cancel_multiplier=40000;
            rules.garbage_target_bonus=GarbageTargetBonus::Defensive;state=quad;state.enemies=6;
            rejected(state,four,rules,"additional cancellation");
            rules=league;rules.garbage_multiplier=600000;rules.cancel_multiplier=895;
            state=quad;state.pieces_placed=0;
            rejected(state,four,rules,"opener cancellation");
            rules=league;rules.garbage_multiplier=(double(std::numeric_limits<int>::max())+.25)/2151.0;
            state=quad;state.combo=2147;state.board.garbage[0]=1u<<1;
            rejected(state,four,rules,"downstack bonus");

            // Native callers may also reach the integer type boundaries.
            state=quad;state.combo=std::numeric_limits<int>::max();
            rejected(state,four,league,"combo counter");
            state=quad;state.b2b=std::numeric_limits<int>::max();
            rejected(state,four,league,"B2B counter");
            state=quad;state.pieces_placed=std::numeric_limits<int>::max();
            rejected(state,four,league,"placed-piece counter");
            rules=league;rules.pc_b2b=std::numeric_limits<int>::max();
            rejected(pc,perfect,rules,"B2B reward");
            rules=league;rules.allclear_charges=true;rules.b2b_charge_at=std::numeric_limits<int>::max();
            rejected(pc,perfect,rules,"B2B charge reward");

            // A wide intermediate that produces a representable final surge
            // remains valid; RNG rounding must not change the surge's floor.
            state=single;state.b2b=std::numeric_limits<int>::max();
            rules=league;rules.b2b_charge_base=std::numeric_limits<int>::max();rules.garbage_multiplier=.25;
            rules.garbage_rounding=GarbageRounding::Random;
            const auto wide=apply_action(state,one,rules);
            check(wide.valid && wide.surge==1073741822 && wide.sent==1073741822,
                  "wide B2B arithmetic preserves a representable floored surge");
            check(wide.state.rng_seed==state.rng_seed,"surge uses floor without consuming RNG in random round mode");
            state=single;state.enemies=std::numeric_limits<int>::max();
            rules=league;rules.garbage_target_bonus=GarbageTargetBonus::Offensive;
            const auto targets=apply_action(state,one,rules);
            check(targets.valid && targets.attack==9,"target count clamps before multiplying large native counters");

            state=State{};state.current=Piece::O;state.queue={Piece::I};
            const auto ordinary=first_action(state,league);
            state.frame=state.last_tank=std::numeric_limits<int64_t>::max();
            rules=league;rules.messiness_timeout=1;
            const auto timeout=apply_action(state,ordinary,rules);
            check(timeout.valid && timeout.state.rng_seed==1,"timeout comparison cannot overflow the frame sum");
            auto elapsed=ordinary;elapsed.elapsed_frames=1;
            rejected(state,elapsed,league,"frame counter overflow");
            state.frame=state.last_tank=0;state.current_multiplier=std::numeric_limits<double>::max();
            rules=league;rules.garbage_margin=0;rules.garbage_increase=std::numeric_limits<double>::max();
            rejected(state,elapsed,rules,"dynamic garbage setting overflow");
        }
        std::cout<<"battle_tests: "<<checks<<" checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"battle_tests failed after "<<checks<<" checks: "<<error.what()<<'\n';
        return 1;
    }
}
