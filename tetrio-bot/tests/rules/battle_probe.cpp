#include "../../engine.hpp"
#include "../../third_party/nlohmann/json.hpp"
#include <iostream>
#include <string>

using nlohmann::json;
using namespace tetrio;

namespace {
template<class T> void field(const json& j, const char* name, T& value) {
    if (j.contains(name)) value = j.at(name).get<T>();
}
void field(const json& j, const char* name, bool& value) {
    if (j.contains(name)) value = j.at(name).is_boolean() ? j.at(name).get<bool>() : j.at(name).get<int>() != 0;
}
json probe(const json& input) {
    const auto options = input.value("options", json::object());
    const auto initial = input.value("state", json::object());
    const auto clear = input.value("clear", json::object());
    Rules rules;
    field(options,"b2bcharging",rules.b2b_charging); field(options,"b2bchaining",rules.b2b_chaining);
    field(options,"b2bextras",rules.b2b_extras); field(options,"b2bcharge_at",rules.b2b_charge_at);
    field(options,"b2bcharge_base",rules.b2b_charge_base); field(options,"allclears",rules.all_clears);
    field(options,"allclear_garbage",rules.pc_garbage); field(options,"allclear_b2b",rules.pc_b2b);
    field(options,"allclear_b2b_dupes",rules.allclear_b2b_dupes);
    field(options,"allclear_b2b_sends",rules.allclear_b2b_sends);
    field(options,"allclear_charges",rules.allclear_charges);
    field(options,"garbagespecialbonus",rules.garbage_special_bonus);
    field(options,"garbagemultiplier",rules.garbage_multiplier);
    field(options,"cancelmultiplier",rules.cancel_multiplier);
    field(options,"receivemultiplier",rules.receive_multiplier);
    field(options,"garbagecap",rules.garbage_cap); field(options,"garbagecapmax",rules.garbage_cap_max);
    field(options,"garbageabsolutecap",rules.garbage_absolute_cap);
    field(options,"garbageattackcap",rules.attack_cap); field(options,"openerphase",rules.opener_phase);
    field(options,"hasgarbage",rules.has_garbage); field(options,"garbageholesize",rules.garbage_hole_width);
    field(options,"messiness_change",rules.messiness_change); field(options,"messiness_inner",rules.messiness_inner);
    field(options,"messiness_nosame",rules.messiness_no_same); field(options,"messiness_center",rules.messiness_center);
    field(options,"messiness_timeout",rules.messiness_timeout);
    const auto combo = options.value("combotable",std::string("multiplier"));
    rules.combo_mode = combo=="none" ? ComboMode::None : combo=="classic guideline" ? ComboMode::Classic :
                       combo=="modern guideline" ? ComboMode::Modern : ComboMode::Multiplier;
    const auto blocking = options.value("garbageblocking",std::string("combo blocking"));
    rules.garbage_blocking = blocking=="none" ? GarbageBlocking::None : blocking=="limited blocking" ?
                             GarbageBlocking::Limited : GarbageBlocking::Combo;
    rules.garbage_rounding = options.value("roundmode",std::string("down"))=="rng" ?
                            GarbageRounding::Random : GarbageRounding::Down;
    const auto target = options.value("garbagetargetbonus",std::string("none"));
    rules.garbage_target_bonus = target=="offensive" ? GarbageTargetBonus::Offensive : target=="defensive" ?
                                GarbageTargetBonus::Defensive : GarbageTargetBonus::None;
    if (options.value("spinbonuses",std::string("all-mini+"))=="handheld") rules.spin_mode=SpinMode::Handheld;
    State state;
    state.pieces_placed=1;
    field(initial,"combo",state.combo); field(initial,"b2b",state.b2b);
    field(initial,"pieces_placed",state.pieces_placed); field(initial,"total_sent",state.total_sent);
    field(initial,"rng_seed",state.rng_seed); field(initial,"frame",state.frame);
    field(initial,"last_tank",state.last_tank); field(initial,"hole_changed",state.hole_changed);
    field(initial,"enemies",state.enemies); field(initial,"garbage_bonus",state.garbage_bonus);
    if (initial.contains("last_hole") && !initial["last_hole"].is_null()) field(initial,"last_hole",state.last_hole);
    int64_t id=0;
    for (const auto& value : initial.value("incoming",json::array())) {
        GarbagePacket packet;
        packet.id=++id;
        field(value,"id",packet.id); field(value,"amount",packet.amount); field(value,"width",packet.width);
        if (value.contains("hole") && !value["hole"].is_null()) field(value,"hole",packet.hole);
        field(value,"active",packet.active); field(value,"ready",packet.spawn_ready);
        field(value,"hardened",packet.hardened); field(value,"shielded",packet.shielded);
        state.incoming.push_back(packet);
    }
    BattleClear event;
    field(clear,"lines",event.lines); field(clear,"garbage_rows",event.garbage_rows);
    field(clear,"perfect_clear",event.perfect_clear);
    const auto spin = clear.value("spin",std::string("none"));
    event.spin = spin=="full" ? Spin::Full : spin=="mini" ? Spin::Mini : Spin::None;
    const auto piece = clear.value("piece",std::string("t"));
    event.piece = piece=="t" ? Piece::T : piece=="i" ? Piece::I : piece=="o" ? Piece::O :
                  piece=="l" ? Piece::L : piece=="j" ? Piece::J : piece=="s" ? Piece::S : Piece::Z;
    BattleResult result;
    if (input.contains("enqueue")) {
        const auto& value=input["enqueue"];
        GarbagePacket packet;packet.active=false;packet.id=id+1;
        field(value,"id",packet.id);field(value,"amount",packet.amount);field(value,"width",packet.width);
        if (value.contains("hole") && !value["hole"].is_null()) field(value,"hole",packet.hole);
        field(value,"active",packet.active);field(value,"ready",packet.spawn_ready);
        field(value,"hardened",packet.hardened);field(value,"shielded",packet.shielded);
        enqueue_garbage(state,packet,rules);
    } else result=resolve_battle(state,event,rules,input.value("tank",true));
    json outgoing=result.outgoing, tank=json::array(), incoming=json::array();
    for (const auto& row:result.tank) tank.push_back({{"hole",row.hole},{"width",row.width}});
    for (const auto& packet:state.incoming) incoming.push_back({
        {"id",packet.id},{"amount",packet.amount},{"hole",packet.hole<0?json(nullptr):json(packet.hole)},
        {"width",packet.width},{"active",packet.active},{"ready",packet.spawn_ready},
        {"hardened",packet.hardened},{"shielded",packet.shielded}});
    return {{"combo",state.combo},{"b2b",state.b2b},{"attack",result.attack},{"sent",result.sent},
            {"cancelled",result.cancelled},{"received",result.received},{"surge",result.surge},
            {"outgoing",outgoing},{"tank",tank},{"blocked",result.blocked},{"total_sent",state.total_sent},
            {"rng_seed",state.rng_seed},{"last_hole",state.last_hole<0?json(nullptr):json(state.last_hole)},
            {"hole_changed",state.hole_changed},{"last_tank",state.last_tank},{"incoming",incoming}};
}
}

int main() {
    std::string line;
    while (std::getline(std::cin,line)) if (!line.empty()) {
        try { std::cout<<probe(json::parse(line)).dump()<<'\n'; }
        catch (const std::exception& e) { std::cout<<json({{"error",e.what()}}).dump()<<'\n'; }
    }
}
