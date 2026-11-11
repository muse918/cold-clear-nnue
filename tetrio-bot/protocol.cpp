#include "protocol.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace tetrio {
namespace {
void require(bool yes,const std::string& message) {if(!yes)throw std::invalid_argument(message);}
void object(const Json& j,const char* name) {require(j.is_object(),std::string(name)+" must be an object");}
void keys(const Json& j,std::initializer_list<const char*> allowed,const char* name) {
    object(j,name);
    for(auto it=j.begin();it!=j.end();++it) {
        bool found=false;
        for(const auto* key:allowed) if(it.key()==key){found=true;break;}
        require(found,std::string(name)+": unknown field "+it.key());
    }
}
int64_t integer(const Json& j,const std::string& name,int64_t low,int64_t high) {
    require(j.is_number_integer(),name+" must be an integer");
    if(j.is_number_unsigned())require(j.get<uint64_t>()<=uint64_t(std::numeric_limits<int64_t>::max()),name+" is too large");
    const auto n=j.get<int64_t>();require(n>=low&&n<=high,name+" is outside its supported range");return n;
}
double number(const Json& j,const std::string& name,double low=0,double high=1e9) {
    require(j.is_number(),name+" must be a number");
    const double n=j.get<double>();require(std::isfinite(n)&&n>=low&&n<=high,name+" is outside its supported range");return n;
}
bool boolean(const Json& j,const std::string& name) {require(j.is_boolean(),name+" must be boolean");return j.get<bool>();}
std::string string(const Json& j,const std::string& name) {require(j.is_string(),name+" must be a string");return j.get<std::string>();}
template<class E>E enumeration(const Json& j,const char* name,std::initializer_list<std::pair<const char*,E>> entries) {
    const auto text=string(j,name);for(const auto& entry:entries)if(text==entry.first)return entry.second;
    throw std::invalid_argument(std::string("Unknown ")+name+": "+text);
}
Piece piece(const Json& j,bool nullable) {
    Piece p=j.is_null()?Piece::None:parse_piece(string(j,"piece"));
    require(nullable||p!=Piece::None,"observed queue/action pieces must be I/O/T/L/J/S/Z; use an empty queue for unknown previews");return p;
}
Json piece_json(Piece p) {return p==Piece::None?Json(nullptr):Json(piece_name(p));}
GarbagePacket parse_packet(const Json& j) {
    keys(j,{"amount","hole","width","active","spawn_ready","hardened","shielded","id"},"packet");
    require(j.contains("amount"),"packet.amount is required");
    GarbagePacket p;p.amount=int(integer(j.at("amount"),"amount",1,1000000));
    if(j.contains("hole"))p.hole=j.at("hole").is_null()?-1:int(integer(j.at("hole"),"hole",-1,9));
    if(j.contains("width"))p.width=int(integer(j.at("width"),"width",1,10));
    if(j.contains("id"))p.id=integer(j.at("id"),"id",0,9007199254740991LL);
#define PB(name) if(j.contains(#name))p.name=boolean(j.at(#name),#name)
    PB(active);PB(spawn_ready);PB(hardened);PB(shielded);
#undef PB
    return p;
}
Json packet_json(const GarbagePacket& p) {
    return {{"amount",p.amount},{"hole",p.hole<0?Json(nullptr):Json(p.hole)},{"width",p.width},
        {"active",p.active},{"spawn_ready",p.spawn_ready},{"hardened",p.hardened},{"shielded",p.shielded},{"id",p.id}};
}
void masks(const Json& j,std::array<uint16_t,HEIGHT>& out,const char* name) {
    require(j.is_array()&&j.size()<=HEIGHT,std::string(name)+" must be an array of at most 40 row masks");
    for(size_t i=0;i<j.size();++i)out[i]=uint16_t(integer(j[i],name,0,FULL_ROW));
}
Json options_json(const SearchOptions& o) {
    return {{"max_depth",o.max_depth},{"beam_width",o.beam_width},{"max_nodes",o.max_nodes},
        {"time_budget_ms",o.time_budget_ms},{"max_cache_nodes",o.max_cache_nodes},
        {"allow_hold",o.allow_hold},{"deduplicate",o.deduplicate},{"frames_per_piece",o.frames_per_piece}};
}
std::string mode_name(EvaluationMode m) {
    return m==EvaluationMode::Heuristic?"heuristic":m==EvaluationMode::NNUE?"nnue":"hybrid";
}
std::string update_name(UpdatePolicy m) {
    return m==UpdatePolicy::FullRefresh?"full":m==UpdatePolicy::RowSeams?"row_seams":"refresh_events";
}
Json options_json(const EvaluatorOptions& o) {
    return {{"mode",mode_name(o.mode)},{"update",update_name(o.update)},{"nnue_weight",o.nnue_weight},
        {"sent_reward",o.sent_reward},{"cancelled_reward",o.cancelled_reward},{"line_reward",o.line_reward},{"pc_reward",o.pc_reward}};
}
} // namespace

Json to_json_value(const Rules& r) {
    Json j;
#define PUT(name) j[#name]=r.name
    PUT(spin_180);PUT(clutch);PUT(lockout);PUT(visible_height);PUT(preview_count);
    PUT(b2b_charging);PUT(b2b_chaining);PUT(b2b_extras);PUT(b2b_charge_at);PUT(b2b_charge_base);
    PUT(pc_garbage);PUT(pc_b2b);PUT(all_clears);PUT(allclear_b2b_dupes);PUT(allclear_b2b_sends);PUT(allclear_charges);
    PUT(garbage_special_bonus);PUT(garbage_multiplier);PUT(garbage_increase);PUT(garbage_margin);
    PUT(garbage_cap);PUT(garbage_cap_increase);PUT(garbage_cap_margin);PUT(garbage_cap_max);PUT(garbage_absolute_cap);
    PUT(cancel_multiplier);PUT(receive_multiplier);PUT(opener_phase);PUT(attack_cap);PUT(has_garbage);
    PUT(garbage_hole_width);PUT(messiness_change);PUT(messiness_inner);PUT(messiness_no_same);PUT(messiness_center);PUT(messiness_timeout);
#undef PUT
    j["spin_mode"]=spin_mode_name(r.spin_mode);j["kicks"]=r.kicks==KickSystem::SRSPlus?"SRS+":"SRS";
    j["combo_mode"]=std::array<const char*,4>{{"multiplier","classic","none","modern"}}.at(size_t(r.combo_mode));
    j["garbage_blocking"]=std::array<const char*,3>{{"combo","limited","none"}}.at(size_t(r.garbage_blocking));
    j["garbage_rounding"]=r.garbage_rounding==GarbageRounding::Down?"down":"random";
    j["garbage_target_bonus"]=std::array<const char*,3>{{"none","offensive","defensive"}}.at(size_t(r.garbage_target_bonus));
    return j;
}

Rules parse_rules(const Json& j,Rules r) {
    object(j,"rules");const auto known=to_json_value(r);
    for(auto it=j.begin();it!=j.end();++it)require(known.contains(it.key())||it.key()=="preset","rules: unknown field "+it.key());
    if(j.contains("preset")){require(string(j.at("preset"),"preset")=="tetra_league","only tetra_league is a named preset");r=Rules::tetra_league();}
#define BOOL(name) if(j.contains(#name))r.name=boolean(j.at(#name),#name)
#define INT(name) if(j.contains(#name))r.name=int(integer(j.at(#name),#name,0,1000000))
#define NUM(name) if(j.contains(#name))r.name=number(j.at(#name),#name,0,1000000)
    BOOL(spin_180);BOOL(clutch);BOOL(lockout);INT(visible_height);INT(preview_count);
    BOOL(b2b_charging);BOOL(b2b_chaining);BOOL(b2b_extras);INT(b2b_charge_at);INT(b2b_charge_base);
    INT(pc_garbage);INT(pc_b2b);BOOL(all_clears);BOOL(allclear_b2b_dupes);BOOL(allclear_b2b_sends);BOOL(allclear_charges);
    BOOL(garbage_special_bonus);NUM(garbage_multiplier);NUM(garbage_increase);INT(garbage_margin);
    NUM(garbage_cap);NUM(garbage_cap_increase);INT(garbage_cap_margin);INT(garbage_cap_max);INT(garbage_absolute_cap);
    NUM(cancel_multiplier);NUM(receive_multiplier);INT(opener_phase);INT(attack_cap);BOOL(has_garbage);
    INT(garbage_hole_width);NUM(messiness_change);NUM(messiness_inner);BOOL(messiness_no_same);BOOL(messiness_center);INT(messiness_timeout);
#undef BOOL
#undef INT
#undef NUM
    if(j.contains("spin_mode"))r.spin_mode=parse_spin_mode(string(j.at("spin_mode"),"spin_mode"));
    if(j.contains("kicks"))r.kicks=enumeration<KickSystem>(j.at("kicks"),"kicks",{{"SRS+",KickSystem::SRSPlus},{"SRS",KickSystem::SRS}});
    if(j.contains("combo_mode"))r.combo_mode=enumeration<ComboMode>(j.at("combo_mode"),"combo_mode",{
        {"multiplier",ComboMode::Multiplier},{"classic",ComboMode::Classic},{"none",ComboMode::None},{"modern",ComboMode::Modern}});
    if(j.contains("garbage_blocking"))r.garbage_blocking=enumeration<GarbageBlocking>(j.at("garbage_blocking"),"garbage_blocking",{
        {"combo",GarbageBlocking::Combo},{"limited",GarbageBlocking::Limited},{"none",GarbageBlocking::None}});
    if(j.contains("garbage_rounding"))r.garbage_rounding=enumeration<GarbageRounding>(j.at("garbage_rounding"),"garbage_rounding",{
        {"down",GarbageRounding::Down},{"random",GarbageRounding::Random}});
    if(j.contains("garbage_target_bonus"))r.garbage_target_bonus=enumeration<GarbageTargetBonus>(j.at("garbage_target_bonus"),"garbage_target_bonus",{
        {"none",GarbageTargetBonus::None},{"offensive",GarbageTargetBonus::Offensive},{"defensive",GarbageTargetBonus::Defensive}});
    r.validate();return r;
}

Json to_json_value(const State& s) {
    Json j={{"board",{{"rows",s.board.rows},{"garbage",s.board.garbage},{"solid",s.board.solid}}},
        {"current",piece_json(s.current)},{"hold",piece_json(s.hold)},{"queue",Json::array()},{"incoming",Json::array()}};
    for(auto p:s.queue)j["queue"].push_back(piece_json(p));
    for(const auto& p:s.incoming)j["incoming"].push_back(packet_json(p));
#define PUT(name) j[#name]=s.name
    PUT(hold_used);PUT(last_clear);PUT(dead);PUT(combo);PUT(b2b);PUT(pieces_placed);PUT(total_sent);PUT(frame);PUT(last_tank);
    PUT(current_multiplier);PUT(current_garbage_cap);PUT(rng_seed);PUT(last_hole);PUT(hole_changed);PUT(enemies);PUT(garbage_bonus);
#undef PUT
    return j;
}

State parse_state(const Json& j,const Rules& r) {
    keys(j,{"board","current","hold","queue","hold_used","last_clear","dead","combo","b2b","pieces_placed","total_sent",
        "frame","last_tank","current_multiplier","current_garbage_cap","rng_seed","last_hole","hole_changed","enemies","garbage_bonus","incoming"},"state");
    State s;require(j.contains("current"),"state.current is required (null means unknown horizon)");s.current=piece(j.at("current"),true);
    if(j.contains("hold"))s.hold=piece(j.at("hold"),true);
    if(j.contains("board")) {
        const auto& b=j.at("board");keys(b,{"rows","garbage","solid"},"board");
        if(b.contains("rows"))masks(b.at("rows"),s.board.rows,"rows");
        if(b.contains("garbage"))masks(b.at("garbage"),s.board.garbage,"garbage");
        if(b.contains("solid"))masks(b.at("solid"),s.board.solid,"solid");
    }
    if(j.contains("queue")) {
        const auto& q=j.at("queue");require(q.is_array()&&q.size()<=size_t(r.preview_count),"queue exceeds configured visible preview_count");
        for(const auto& p:q)s.queue.push_back(piece(p,false));
    }
#define BOOL(name) if(j.contains(#name))s.name=boolean(j.at(#name),#name)
#define INT(name) if(j.contains(#name))s.name=int(integer(j.at(#name),#name,0,1000000))
    BOOL(hold_used);BOOL(last_clear);BOOL(dead);BOOL(hole_changed);INT(combo);INT(b2b);INT(pieces_placed);INT(enemies);
#undef BOOL
#undef INT
    if(j.contains("total_sent"))s.total_sent=number(j.at("total_sent"),"total_sent");
    if(j.contains("garbage_bonus"))s.garbage_bonus=number(j.at("garbage_bonus"),"garbage_bonus");
    if(j.contains("frame"))s.frame=integer(j.at("frame"),"frame",0,9007199254740991LL);
    if(j.contains("last_tank"))s.last_tank=integer(j.at("last_tank"),"last_tank",0,s.frame);
    if(j.contains("current_multiplier"))s.current_multiplier=number(j.at("current_multiplier"),"current_multiplier",-1,1000000);
    if(j.contains("current_garbage_cap"))s.current_garbage_cap=number(j.at("current_garbage_cap"),"current_garbage_cap",-1,1000000);
    require(s.current_multiplier>=0||s.current_multiplier==-1,"current_multiplier must be nonnegative or -1");
    require(s.current_garbage_cap>=0||s.current_garbage_cap==-1,"current_garbage_cap must be nonnegative or -1");
    require(s.current_multiplier>=0||r.garbage_increase==0||s.frame<=r.garbage_margin,
        "late snapshot requires authoritative current_multiplier; elapsed frame is not enough");
    require(s.current_garbage_cap>=0||r.garbage_cap_increase==0||s.frame<=r.garbage_cap_margin,
        "late snapshot requires authoritative current_garbage_cap; elapsed frame is not enough");
    if(s.current_multiplier<0)s.current_multiplier=r.garbage_multiplier;
    if(s.current_garbage_cap<0)s.current_garbage_cap=r.garbage_cap;
    if(j.contains("rng_seed"))s.rng_seed=uint32_t(integer(j.at("rng_seed"),"rng_seed",1,2147483646));
    if(j.contains("last_hole"))s.last_hole=j.at("last_hole").is_null()?-1:int(integer(j.at("last_hole"),"last_hole",-1,9));
    require(!s.hole_changed||s.last_hole>=0,"hole_changed requires a known last_hole");
    if(j.contains("incoming")) {
        const auto& incoming=j.at("incoming");require(incoming.is_array()&&incoming.size()<=4096,"incoming must be an array with at most 4096 packets");
        int64_t total=0;for(const auto& p:incoming){s.incoming.push_back(parse_packet(p));total+=s.incoming.back().amount;}
        require(total<=1000000,"pending garbage total exceeds 1000000");
    }
    s.board.validate();return s;
}

SearchOptions parse_search_options(const Json& j,SearchOptions o) {
    keys(j,{"max_depth","beam_width","max_nodes","time_budget_ms","max_cache_nodes","allow_hold","deduplicate","frames_per_piece"},"search");
    if(j.contains("max_depth"))o.max_depth=unsigned(integer(j.at("max_depth"),"max_depth",1,64));
    if(j.contains("beam_width"))o.beam_width=size_t(integer(j.at("beam_width"),"beam_width",1,1000000));
    if(j.contains("max_nodes"))o.max_nodes=size_t(integer(j.at("max_nodes"),"max_nodes",1,100000000));
    if(j.contains("max_cache_nodes"))o.max_cache_nodes=size_t(integer(j.at("max_cache_nodes"),"max_cache_nodes",2,10000000));
    if(j.contains("time_budget_ms"))o.time_budget_ms=number(j.at("time_budget_ms"),"time_budget_ms",0,3600000);
    if(j.contains("frames_per_piece"))o.frames_per_piece=int(integer(j.at("frames_per_piece"),"frames_per_piece",0,216000));
    if(j.contains("allow_hold"))o.allow_hold=boolean(j.at("allow_hold"),"allow_hold");
    if(j.contains("deduplicate"))o.deduplicate=boolean(j.at("deduplicate"),"deduplicate");
    return o;
}
EvaluatorOptions parse_evaluator_options(const Json& j,EvaluatorOptions o) {
    keys(j,{"mode","update","nnue_weight","sent_reward","cancelled_reward","line_reward","pc_reward"},"evaluation");
    if(j.contains("mode"))o.mode=enumeration<EvaluationMode>(j.at("mode"),"mode",{
        {"heuristic",EvaluationMode::Heuristic},{"nnue",EvaluationMode::NNUE},{"hybrid",EvaluationMode::Hybrid}});
    if(j.contains("update"))o.update=enumeration<UpdatePolicy>(j.at("update"),"update",{
        {"full",UpdatePolicy::FullRefresh},{"refresh_events",UpdatePolicy::RefreshEvents},{"row_seams",UpdatePolicy::RowSeams}});
    if(j.contains("nnue_weight"))o.nnue_weight=number(j.at("nnue_weight"),"nnue_weight",0,1);
#define WEIGHT(name) if(j.contains(#name))o.name=number(j.at(#name),#name,-1000000,1000000)
    WEIGHT(sent_reward);WEIGHT(cancelled_reward);WEIGHT(line_reward);WEIGHT(pc_reward);
#undef WEIGHT
    return o;
}

Json to_json_value(const Action& a) {
    const auto& p=a.placement;Json path=Json::array();for(auto m:p.path)path.push_back(move_name(m));
    Json cells=Json::array();for(const auto& c:piece_cells(p.piece,p.pose))cells.push_back({c.x,c.y});
    return {{"piece",piece_json(p.piece)},{"x",p.pose.x},{"y",p.pose.y},{"rotation",p.pose.rotation},
        {"spin",spin_name(p.spin)},{"use_hold",a.use_hold},{"elapsed_frames",a.elapsed_frames},
        {"last_rotation",p.last_rotation},{"rotated180",p.rotated180},{"kick_index",p.kick_index},{"path",path},{"cells",cells}};
}
Action match_action(const Json& j,const State& state,const Rules& r) {
    keys(j,{"piece","x","y","rotation","spin","use_hold","elapsed_frames","last_rotation","rotated180","kick_index","path","cells"},"action");
    for(const char* k:{"piece","x","y","rotation"})require(j.contains(k),std::string("action.")+k+" is required");
    const auto p=piece(j.at("piece"),false);
    const int x=int(integer(j.at("x"),"x",-4,10)),y=int(integer(j.at("y"),"y",0,44));
    const int rotation=int(integer(j.at("rotation"),"rotation",0,3));
    const bool hold=j.contains("use_hold")?boolean(j.at("use_hold"),"use_hold"):false;
    const int frames=j.contains("elapsed_frames")?int(integer(j.at("elapsed_frames"),"elapsed_frames",0,216000)):0;
    const Spin spin=j.contains("spin")?enumeration<Spin>(j.at("spin"),"spin",{
        {"none",Spin::None},{"mini",Spin::Mini},{"full",Spin::Full}}):Spin::None;
    for(auto action:enumerate_actions(state,r,true)) {
        const auto& q=action.placement;
        if(q.piece==p&&q.pose.x==x&&q.pose.y==y&&q.pose.rotation==rotation&&q.spin==spin&&action.use_hold==hold) {
            action.elapsed_frames=frames;return action;
        }
    }
    throw std::invalid_argument("action is not a generated reachable placement with this spin and hold state");
}

Json to_json_value(const Transition& t) {
    Json source=Json::array();for(auto y:t.delta.row_source)source.push_back(int(y));
    return {{"valid",t.valid},{"error",t.error},{"state",to_json_value(t.state)},
        {"events",{{"lines",t.lines},{"garbage_cleared",t.garbage_cleared},{"spin",spin_name(t.spin)},
            {"perfect_clear",t.perfect_clear},{"attack",t.attack},{"sent",t.sent},{"cancelled",t.cancelled},
            {"received",t.received},{"surge",t.surge},{"outgoing",t.outgoing}}},
        {"delta",{{"placed_cells",t.delta.placed_cells},{"row_source",source},{"clear_mask",t.delta.clear_mask},
            {"clears",t.delta.clears},{"garbage",t.delta.garbage}}}};
}
Json to_json_value(const SearchResult& r) {
    Json stats;
#define STAT(name) stats[#name]=r.stats.name
    STAT(nodes);STAT(evaluations);STAT(expanded);STAT(cache_hits);STAT(duplicate_paths);STAT(beam_pruned);STAT(unknown_cutoffs);
    STAT(terminal_cutoffs);STAT(reused_nodes);STAT(cache_nodes);STAT(completed_depth);STAT(reached_depth);
    STAT(node_budget_exhausted);STAT(deadline_exhausted);STAT(cache_budget_exhausted);STAT(elapsed_ms);
#undef STAT
    Json pv=Json::array(),candidates=Json::array();
    for(const auto& a:r.principal_variation)pv.push_back(to_json_value(a));
    for(const auto& c:r.candidates)candidates.push_back({{"action",to_json_value(c.action)},{"value",c.value},{"depth",c.depth},{"survives",c.survives}});
    return {{"action",r.action?to_json_value(*r.action):Json(nullptr)},{"value",r.value},{"pv",pv},
        {"candidates",candidates},{"stats",stats},{"termination",r.termination}};
}

Protocol::Protocol(std::string model,EvaluatorOptions o):evaluator_options_(o),model_path_(std::move(model)),
    evaluator_(new BotEvaluator(model_path_,o)),searcher_(new Searcher(rules_,*evaluator_)) {}

Json Protocol::handle(const Json& request) {
    object(request,"request");require(request.contains("cmd"),"cmd is required");
    const auto cmd=string(request.at("cmd"),"cmd");Json reply={{"ok",true},{"cmd",cmd}};
    if(request.contains("id"))reply["id"]=request.at("id");
    auto need_state=[&]() -> State& {require(state_.has_value(),"send a state snapshot first");return *state_;};
    if(cmd=="info") {
        keys(request,{"cmd","id"},"request");
        reply["name"]="cold-clear-nnue TETR.IO placement search";reply["protocol_version"]=1;
        reply["environment"]="TETR.IO Beta 1.7.8 / engine 19";
        reply["commands"]={"info","configure","state","actions","search","apply","enqueue","advance","clear_cache"};
        reply["rules"]=to_json_value(rules_);reply["search"]=options_json(search_options_);
        reply["evaluation"]=options_json(evaluator_options_);reply["effective_evaluator"]=evaluator_->name();
        reply["model_loaded"]=evaluator_->has_model();reply["coordinate_system"]="bottom-first 10x40, top-left matrix anchor";
        reply["path_contract"]="geometric witness only; no gravity or frame-accurate key execution";
    } else if(cmd=="configure") {
        keys(request,{"cmd","id","rules","search","evaluation","model"},"request");
        auto rules=request.contains("rules")?parse_rules(request.at("rules"),rules_):rules_;
        auto search=request.contains("search")?parse_search_options(request.at("search"),search_options_):search_options_;
        auto evaluation=request.contains("evaluation")?parse_evaluator_options(request.at("evaluation"),evaluator_options_):evaluator_options_;
        auto model=request.contains("model")?(request.at("model").is_null()?std::string{}:string(request.at("model"),"model")):model_path_;
        // Validate and load before changing anything in the live session.
        if(state_)parse_state(to_json_value(*state_),rules);
        auto evaluator=std::make_unique<BotEvaluator>(model,evaluation);
        auto searcher=std::make_unique<Searcher>(rules,*evaluator);
        searcher_.reset();evaluator_=std::move(evaluator);searcher_=std::move(searcher);
        rules_=rules;search_options_=search;evaluator_options_=evaluation;model_path_=model;
        reply["rules"]=to_json_value(rules_);reply["search"]=options_json(search_options_);
        reply["evaluation"]=options_json(evaluator_options_);reply["effective_evaluator"]=evaluator_->name();
    } else if(cmd=="state") {
        keys(request,{"cmd","id","state"},"request");
        if(request.contains("state"))state_=parse_state(request.at("state"),rules_);
        reply["state"]=to_json_value(need_state());
    } else if(cmd=="actions") {
        keys(request,{"cmd","id","state","allow_hold"},"request");
        const bool hold=request.contains("allow_hold")?boolean(request.at("allow_hold"),"allow_hold"):search_options_.allow_hold;
        if(request.contains("state"))state_=parse_state(request.at("state"),rules_);
        reply["actions"]=Json::array();for(const auto& a:enumerate_actions(need_state(),rules_,hold))reply["actions"].push_back(to_json_value(a));
    } else if(cmd=="search") {
        keys(request,{"cmd","id","state","search"},"request");
        const auto options=request.contains("search")?parse_search_options(request.at("search"),search_options_):search_options_;
        if(request.contains("state"))state_=parse_state(request.at("state"),rules_);
        reply.update(to_json_value(searcher_->search(need_state(),options)));
        reply["effective_evaluator"]=evaluator_->name();
    } else if(cmd=="apply") {
        keys(request,{"cmd","id","action"},"request");require(request.contains("action"),"action is required");
        const auto action=match_action(request.at("action"),need_state(),rules_);
        auto transition=apply_action(*state_,action,rules_);require(transition.valid,transition.error);
        state_=transition.state;reply.update(to_json_value(transition));reply["action"]=to_json_value(action);
    } else if(cmd=="enqueue") {
        keys(request,{"cmd","id","packet"},"request");require(request.contains("packet"),"packet is required");
        State next=need_state();const auto packet=parse_packet(request.at("packet"));
        reply["accepted"]=enqueue_garbage(next,packet,rules_);parse_state(to_json_value(next),rules_);
        state_=std::move(next);reply["state"]=to_json_value(*state_);
    } else if(cmd=="advance") {
        keys(request,{"cmd","id","frames"},"request");require(request.contains("frames"),"frames is required");
        State next=need_state();advance_frames(next,rules_,int(integer(request.at("frames"),"frames",0,216000)));
        state_=std::move(next);reply["state"]=to_json_value(*state_);
    } else if(cmd=="clear_cache") {
        keys(request,{"cmd","id"},"request");searcher_->clear();
    } else throw std::invalid_argument("Unknown cmd: "+cmd);
    return reply;
}
} // namespace tetrio
