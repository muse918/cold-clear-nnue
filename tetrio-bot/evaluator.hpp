#pragma once
#include "search.hpp"
#include <memory>
#include <string>

namespace tetrio {
enum class EvaluationMode { Heuristic, NNUE, Hybrid };
enum class UpdatePolicy { FullRefresh, RefreshEvents, RowSeams };
struct EvaluatorOptions {
    EvaluationMode mode=EvaluationMode::Hybrid;
    UpdatePolicy update=UpdatePolicy::RefreshEvents;
    double nnue_weight=.25;
    double sent_reward=4;
    double cancelled_reward=3;
    double line_reward=.5;
    double pc_reward=2;
};

// The included weights were trained on the previous surrogate, not battle wins.
// Battle context and reward are explicit, configurable terms; no new training
// or calibrated strength claim is implied by this integration.
class BotEvaluator final : public SearchEvaluator {
public:
    explicit BotEvaluator(std::string model_path={}, EvaluatorOptions options={});
    ~BotEvaluator();
    BotEvaluator(BotEvaluator&&) noexcept;
    BotEvaluator& operator=(BotEvaluator&&) noexcept;
    Evaluation evaluate(const State&,const State*,const Evaluation*,const Transition*) const override;
    double reward(const Transition&) const override;
    bool has_model() const;
    std::string name() const;
    static double board_heuristic(const Board&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tetrio
