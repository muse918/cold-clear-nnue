#pragma once
#include "evaluator.hpp"
#include "third_party/nlohmann/json.hpp"
#include <memory>

namespace tetrio {
using Json = nlohmann::json;
Json to_json_value(const State&);
Json to_json_value(const Rules&);
Json to_json_value(const Action&);
Json to_json_value(const Transition&);
Json to_json_value(const SearchResult&);
State parse_state(const Json&, const Rules&);
Rules parse_rules(const Json&, Rules base=Rules::tetra_league());
SearchOptions parse_search_options(const Json&, SearchOptions base={});
EvaluatorOptions parse_evaluator_options(const Json&, EvaluatorOptions base={});
Action match_action(const Json&, const State&, const Rules&);

// One session owns its rules, evaluator, memoized search DAG and optional state.
// Callers transport already reconciled battle snapshots, never raw server IGE.
class Protocol {
public:
    explicit Protocol(std::string model={}, EvaluatorOptions evaluation={});
    Json handle(const Json& request);
private:
    Rules rules_=Rules::tetra_league();
    SearchOptions search_options_{};
    EvaluatorOptions evaluator_options_{};
    std::string model_path_;
    std::unique_ptr<BotEvaluator> evaluator_;
    std::unique_ptr<Searcher> searcher_;
    std::optional<State> state_;
};
} // namespace tetrio
