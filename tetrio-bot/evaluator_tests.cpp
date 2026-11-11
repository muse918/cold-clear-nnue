#include "evaluator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tetrio;
namespace {
size_t checks=0;
void check(bool value,const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
bool close(double a,double b,double absolute=1e-9,double relative=1e-9) {
    return std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=absolute+relative*std::abs(b);
}

struct Coverage {
    size_t edges=0,ordinary=0,clear=0,pc=0,garbage=0,clear_garbage=0,hold=0,dead=0;
    size_t selected=0,segments=0,max_segment=0,accumulator_components=0;
    std::array<size_t,5> clear_sizes{};
    double event_acc_error=0,seam_acc_error=0,event_value_error=0,seam_value_error=0;
} coverage;

void validate_delta(const State& parent,const Transition& result) {
    std::array<uint16_t,HEIGHT> placed=parent.board.rows;
    for (const auto cell:result.delta.placed_cells) {
        check(cell<WIDTH*HEIGHT,"valid transition must identify four placed cells");
        placed[cell/WIDTH]|=uint16_t(1u<<(cell%WIDTH));
    }
    int previous=-1;
    for (int y=0;y<HEIGHT;y++) {
        const int source=result.delta.row_source[y];
        check(source>=-1 && source<HEIGHT,"row source range");
        if (source<0) continue;
        check(source>previous,"surviving parent rows must remain unique and ordered");
        check(!(result.delta.clear_mask&(uint64_t(1)<<source)),"cleared row cannot survive");
        check(result.state.board.rows[y]==placed[source],"row-source occupancy after placement/clear/tank");
        check(result.state.board.garbage[y]==parent.board.garbage[source],"row-source garbage provenance");
        check(result.state.board.solid[y]==parent.board.solid[source],"row-source permanent provenance");
        previous=source;
    }
    check(result.delta.clears==result.lines,"delta clear count matches battle transition");
    check(result.delta.garbage<=result.received,"inserted rows cannot exceed consumed rows");
    result.state.board.validate();
}

using Evaluations=std::array<Evaluation,3>;
struct Harness {
    BotEvaluator full,events,seams;
    explicit Harness(const std::string& model):
        full(model,{EvaluationMode::NNUE,UpdatePolicy::FullRefresh}),
        events(model,{EvaluationMode::NNUE,UpdatePolicy::RefreshEvents}),
        seams(model,{EvaluationMode::NNUE,UpdatePolicy::RowSeams}) {}
    Evaluations fresh(const State& state) {
        return {{full.evaluate(state,nullptr,nullptr,nullptr),events.evaluate(state,nullptr,nullptr,nullptr),
                 seams.evaluate(state,nullptr,nullptr,nullptr)}};
    }
    Evaluations edge(const State& parent,const Evaluations& cache,const Action& action,const Transition& transition) {
        check(transition.valid,"enumerated action must apply successfully: "+transition.error);
        validate_delta(parent,transition);
        ++coverage.edges;
        coverage.ordinary+=transition.lines==0 && transition.delta.garbage==0;
        coverage.clear+=transition.lines>0;
        coverage.pc+=transition.perfect_clear;
        coverage.garbage+=transition.delta.garbage>0;
        coverage.clear_garbage+=transition.lines>0 && transition.delta.garbage>0;
        coverage.hold+=action.use_hold;
        coverage.dead+=transition.state.dead;
        ++coverage.clear_sizes[transition.lines];
        Evaluations current{{
            full.evaluate(transition.state,&parent,&cache[0],&transition),
            events.evaluate(transition.state,&parent,&cache[1],&transition),
            seams.evaluate(transition.state,&parent,&cache[2],&transition)
        }};
        for (int method=1;method<3;method++) {
            check(current[method].valid_accumulator==current[0].valid_accumulator,"accumulator validity parity");
            const double score_error=std::abs(current[method].value-current[0].value);
            (method==1?coverage.event_value_error:coverage.seam_value_error)=std::max(
                method==1?coverage.event_value_error:coverage.seam_value_error,score_error);
            check(close(current[method].value,current[0].value,1e-3,1e-5),
                  "incremental value differs from full refresh");
            if (!current[0].valid_accumulator) continue;
            for (size_t i=0;i<current[0].accumulator.size();i++) {
                ++coverage.accumulator_components;
                const double a=current[method].accumulator[i],b=current[0].accumulator[i];
                const double error=std::abs(a-b);
                (method==1?coverage.event_acc_error:coverage.seam_acc_error)=std::max(
                    method==1?coverage.event_acc_error:coverage.seam_acc_error,error);
                if (!close(a,b,1e-4,3e-5)) throw std::runtime_error(
                    "incremental accumulator differs from full refresh at component "+std::to_string(i));
            }
        }
        return current;
    }
    void fixture(const State& state,const Rules& rules,int required_lines=-1,bool required_pc=false,
                 bool required_garbage=false,bool required_hold=false) {
        const auto cache=fresh(state);
        bool witness=false;
        const auto actions=enumerate_actions(state,rules);
        check(!actions.empty(),"test fixture must have legal actions");
        for (const auto& action:actions) {
            const auto transition=apply_action(state,action,rules);
            edge(state,cache,action,transition);
            witness|=(required_lines<0 || transition.lines==required_lines) &&
                (!required_pc || transition.perfect_clear) &&
                (!required_garbage || transition.delta.garbage>0) && (!required_hold || action.use_hold);
        }
        check(witness,"fixture must exercise its intended event");
    }
};

State base(Piece current=Piece::T) {
    State state;state.current=current;state.pieces_placed=20;
    state.queue={Piece::I,Piece::O,Piece::T,Piece::L,Piece::J,Piece::S,Piece::Z};
    return state;
}
GarbagePacket incoming(int amount,int hole=4,int width=1) {
    GarbagePacket packet;packet.amount=amount;packet.hole=hole;packet.width=width;return packet;
}

// A tiny valid cell-NNUE with a known answer verifies the public adapter's piece
// context without relying on learned weights coincidentally separating pieces.
struct ContextModel {
    std::filesystem::path path;
    explicit ContextModel(std::filesystem::path location):path(std::move(location)) {
        std::ofstream out(path,std::ios::binary);
        if (!out) throw std::runtime_error("cannot create temporary context model");
        const char magic[16]={'S','P','A','T','I','A','L','N','N','U','E','v','1',0,0,0};
        out.write(magic,sizeof(magic));
        const std::array<uint32_t,5> dimensions{{0,0,128,135,32}};
        out.write(reinterpret_cast<const char*>(dimensions.data()),sizeof(dimensions));
        auto scalar=[&](float x){out.write(reinterpret_cast<const char*>(&x),sizeof(x));};
        auto floats=[&](const std::vector<float>& x){out.write(reinterpret_cast<const char*>(x.data()),
            static_cast<std::streamsize>(x.size()*sizeof(float)));};
        scalar(0);scalar(1);
        floats(std::vector<float>(400*128));
        floats(std::vector<float>(128));
        std::vector<float> head(135*32);
        for (int piece=0;piece<7;piece++) head[(128+piece)*32]=float(piece+1);
        floats(head);floats(std::vector<float>(32));
        std::vector<float> output(32);output[0]=1;floats(output);scalar(0);
        out.close();
        if (!out) throw std::runtime_error("failed to write context model");
    }
    ~ContextModel(){std::error_code ignored;std::filesystem::remove(path,ignored);}
};

void context_tests(const std::filesystem::path& executable) {
    ContextModel fixture(executable.parent_path()/"evaluator_piece_context_test.snn");
    BotEvaluator network(fixture.path.string(),{EvaluationMode::NNUE,UpdatePolicy::FullRefresh});
    BotEvaluator heuristic({}, {EvaluationMode::Heuristic,UpdatePolicy::FullRefresh});
    State state=base();state.board.rows[0]=0b1011;state.queue={Piece::Z};
    for (int piece=0;piece<7;piece++) {
        state.current=Piece(piece);
        check(close(network.evaluate(state,nullptr,nullptr,nullptr).value,piece+1),"NNUE input is state.current in IOTLJSZ order");
    }
    state.current=Piece::T;
    const auto before=network.evaluate(state,nullptr,nullptr,nullptr);
    state.queue={Piece::I,Piece::O};
    check(close(network.evaluate(state,nullptr,nullptr,nullptr).value,before.value),"queue.front must not replace current NNUE input");
    state.current=Piece::None;
    const auto unknown=network.evaluate(state,nullptr,nullptr,nullptr);
    check(unknown.valid_accumulator,"unknown-horizon fallback still maintains board accumulator");
    check(close(unknown.value,heuristic.evaluate(state,nullptr,nullptr,nullptr).value),
          "unknown current uses explicit board heuristic rather than an untrained zero context");
    state.current=Piece::T;state.b2b=100;state.combo=30;state.board.rows[15]=1;
    state.incoming={incoming(3),incoming(5),incoming(7),incoming(100)};
    state.incoming[1].active=false;state.incoming[2].spawn_ready=false;state.incoming[3].shielded=true;
    // Context: capped B2B 8 + capped combo 1.5 - pending 3 - ready*danger 1.2 = 5.3.
    check(close(network.evaluate(state,nullptr,nullptr,nullptr).value,3+5.3),
          "battle context distinguishes pending, ready, shielded, and capped counters");
    EvaluatorOptions mix;mix.mode=EvaluationMode::Hybrid;mix.nnue_weight=.25;
    BotEvaluator hybrid(fixture.path.string(),mix);
    check(close(hybrid.evaluate(state,nullptr,nullptr,nullptr).value,
                .25*3+.75*BotEvaluator::board_heuristic(state.board)+5.3),"hybrid weight applies to board terms before context");
    Transition reward;reward.sent=2;reward.cancelled=3;reward.lines=4;reward.perfect_clear=true;reward.attack=999;reward.received=999;
    check(close(network.reward(reward),21),"reward uses sent/cancelled/clear/PC without raw-attack double counting");
    state.dead=true;
    const auto dead=network.evaluate(state,nullptr,nullptr,nullptr);
    check(dead.value==-1e12 && !dead.valid_accumulator,"dead state is terminal without a reusable NNUE accumulator");
}
}

int main(int argc,char** argv) {
    try {
        const std::string model=argc>1?argv[1]:"../experiments/spatial-nnue/results/final/seed918/attention.snn";
        context_tests(std::filesystem::path(argv[0]));
        Harness harness(model);
        Rules league=Rules::tetra_league();
        harness.fixture(base(),league,0,false,false,true);
        {
            State state=base(Piece::I);state.board.rows[0]=FULL_ROW ^ (uint16_t(15)<<3);state.board.rows[5]=1u<<9;
            harness.fixture(state,league,1);
            Rules limited=league;limited.garbage_blocking=GarbageBlocking::Limited;state.incoming={incoming(12)};
            harness.fixture(state,limited,1,false,true);
        }
        {
            State state=base(Piece::O);
            state.board.rows[0]=state.board.rows[1]=FULL_ROW ^ ((1u<<4)|(1u<<5));state.board.rows[5]=1;
            harness.fixture(state,league,2);
        }
        {
            State state=base(Piece::I);
            for (int y=0;y<3;y++) state.board.rows[y]=FULL_ROW^1;
            harness.fixture(state,league,3);
            state.board.rows[3]=FULL_ROW^1;
            harness.fixture(state,league,4,true);
            state.current=Piece::O;state.hold=Piece::I;
            harness.fixture(state,league,4,true,false,true);
            Rules independent=league;independent.garbage_blocking=GarbageBlocking::None;
            state.incoming={incoming(3)};
            harness.fixture(state,independent,4,true,true,true);
        }
        {
            State state=base(Piece::L);state.incoming={incoming(8,9,3)};
            harness.fixture(state,league,0,false,true);
            state.board.rows[3]=state.board.solid[3]=1;state.board.rows[0]=1u<<8;
            harness.fixture(state,league,0,false,true);
        }
        {
            // Exactly 256 inserted rows must remain an event, not wrap to zero
            // when exported to the native kernel's uint8_t record field.
            Rules tall=league;tall.garbage_cap=tall.garbage_cap_max=256;
            State state=base(Piece::O);state.board.rows[8]=1;state.incoming={incoming(256,0,10)};
            const auto cache=harness.fresh(state);
            bool found=false;
            for (const auto& action:enumerate_actions(state,tall,false)) {
                const auto cells=piece_cells(action.placement.piece,action.placement.pose);
                if (!std::all_of(cells.begin(),cells.end(),[](const Cell& c){return c.x>=7;})) continue;
                const auto transition=apply_action(state,action,tall);
                check(transition.delta.garbage==256 && transition.state.board.empty() && !transition.state.dead,
                      "256-row event fixture is alive with all prior cells discarded");
                harness.edge(state,cache,action,transition);found=true;break;
            }
            check(found,"256-row event has a placement witness");
        }

        uint32_t rng=9182026;
        auto random=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;};
        for (int segment=0;segment<4;segment++) {
            State state=base(Piece(segment));state.queue.clear();
            for (int bag=0;bag<40;bag++) {
                std::array<Piece,7> pieces{{Piece::I,Piece::O,Piece::T,Piece::L,Piece::J,Piece::S,Piece::Z}};
                for (int i=6;i>0;i--) std::swap(pieces[i],pieces[random()%unsigned(i+1)]);
                state.queue.insert(state.queue.end(),pieces.begin(),pieces.end());
            }
            Rules rules=league;
            if (segment==2) rules.garbage_blocking=GarbageBlocking::Limited;
            if (segment==3) rules.garbage_blocking=GarbageBlocking::None;
            auto cache=harness.fresh(state);
            size_t length=0;
            for (int step=0;step<64 && !state.dead;step++) {
                if (step && step%12==0) state.incoming.push_back(incoming(1+(segment%2),int(random()%10)));
                const auto actions=enumerate_actions(state,rules);
                if (actions.empty()) break;
                double best=-std::numeric_limits<double>::infinity();
                Transition selected;Evaluations selected_cache{};
                for (const auto& action:actions) {
                    const auto transition=apply_action(state,action,rules);
                    const auto candidate_cache=harness.edge(state,cache,action,transition);
                    const double value=transition.state.dead?-1e12:BotEvaluator::board_heuristic(transition.state.board)+
                        4*transition.sent+3*transition.cancelled+2*transition.lines+4*transition.perfect_clear;
                    if (value>best) {best=value;selected=transition;selected_cache=candidate_cache;}
                }
                state=std::move(selected.state);cache=std::move(selected_cache);++length;++coverage.selected;
            }
            ++coverage.segments;coverage.max_segment=std::max(coverage.max_segment,length);
        }
        check(coverage.ordinary && coverage.clear && coverage.pc && coverage.garbage && coverage.clear_garbage && coverage.hold,
              "all requested transition families were exercised");
        for (int lines=1;lines<=4;lines++) check(coverage.clear_sizes[lines]>0,"each line-clear count was exercised");
        check(coverage.max_segment>=32,"a meaningful chained accumulator trajectory survived");
        {
            State state=base();const auto parent=harness.fresh(state);
            const auto action=enumerate_actions(state,league,false).front();const auto transition=apply_action(state,action,league);
            Evaluation invalid;const auto actual=harness.events.evaluate(transition.state,&state,&invalid,&transition);
            const auto expected=harness.full.evaluate(transition.state,nullptr,nullptr,nullptr);
            check(actual.accumulator==expected.accumulator && actual.value==expected.value,
                  "missing/invalid parent accumulator forces an exact full refresh");
            (void)parent;
        }
        std::cout<<std::setprecision(12)<<"{\n"
            <<"  \"suite\": \"evaluator_tests\",\n  \"passed\": true,\n  \"checks\": "<<checks
            <<",\n  \"transition_edges\": "<<coverage.edges<<",\n  \"ordinary_edges\": "<<coverage.ordinary
            <<",\n  \"clear_edges\": "<<coverage.clear<<",\n  \"perfect_clear_edges\": "<<coverage.pc
            <<",\n  \"garbage_edges\": "<<coverage.garbage<<",\n  \"clear_and_garbage_edges\": "<<coverage.clear_garbage
            <<",\n  \"hold_edges\": "<<coverage.hold<<",\n  \"dead_edges\": "<<coverage.dead
            <<",\n  \"selected_transitions\": "<<coverage.selected<<",\n  \"segments\": "<<coverage.segments
            <<",\n  \"max_segment_length\": "<<coverage.max_segment<<",\n  \"accumulator_components\": "<<coverage.accumulator_components
            <<",\n  \"refresh_events_max_acc_abs\": "<<coverage.event_acc_error
            <<",\n  \"row_seams_max_acc_abs\": "<<coverage.seam_acc_error
            <<",\n  \"refresh_events_max_value_abs\": "<<coverage.event_value_error
            <<",\n  \"row_seams_max_value_abs\": "<<coverage.seam_value_error
            <<",\n  \"tolerance\": {\"acc_abs\": 0.0001, \"acc_rel\": 0.00003, \"value_abs\": 0.001, \"value_rel\": 0.00001}\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"evaluator_tests failed after "<<checks<<" checks and "<<coverage.edges<<" edges: "<<error.what()<<'\n';
        return 1;
    }
}
