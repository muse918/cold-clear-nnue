#include "evaluator.hpp"
#define SPATIAL_KERNEL_LIBRARY
#include "../experiments/spatial-nnue/kernels.cpp"

#include <cmath>
#include <optional>

namespace tetrio {
struct BotEvaluator::Impl {
    EvaluatorOptions options;
    std::optional<spatial::Model> model;
    explicit Impl(const std::string& path,EvaluatorOptions o):options(o) {
        if (!std::isfinite(o.nnue_weight) || o.nnue_weight<0 || o.nnue_weight>1 ||
            !std::isfinite(o.sent_reward) || !std::isfinite(o.cancelled_reward) ||
            !std::isfinite(o.line_reward) || !std::isfinite(o.pc_reward))
            throw std::invalid_argument("Invalid evaluator weights");
        if (!path.empty()) model=spatial::Model::load(path);
        if (!model && o.mode==EvaluationMode::NNUE)
            throw std::invalid_argument("NNUE mode requires a .snn model");
    }
};

BotEvaluator::BotEvaluator(std::string path,EvaluatorOptions o):impl_(new Impl(path,o)) {}
BotEvaluator::~BotEvaluator()=default;
BotEvaluator::BotEvaluator(BotEvaluator&&) noexcept=default;
BotEvaluator& BotEvaluator::operator=(BotEvaluator&&) noexcept=default;
bool BotEvaluator::has_model()const{return bool(impl_->model);}
std::string BotEvaluator::name()const {
    if (!impl_->model || impl_->options.mode==EvaluationMode::Heuristic) return "heuristic";
    return impl_->options.mode==EvaluationMode::NNUE?"nnue+context":"hybrid-nnue+heuristic+context";
}

double BotEvaluator::board_heuristic(const Board& board) {
    std::array<int,WIDTH> heights{};
    int total=0,highest=0,holes=0,covered=0,bump=0,wells=0;
    for (int x=0;x<WIDTH;++x) {
        for (int y=HEIGHT-1;y>=0;--y) if(board.rows[y]&(1u<<x)){heights[x]=y+1;break;}
        highest=std::max(highest,heights[x]);total+=heights[x];
        int blocks=0;
        for(int y=heights[x]-1;y>=0;--y) {
            if(board.rows[y]&(1u<<x))++blocks;
            else {++holes;covered+=blocks;}
        }
        if(x)bump+=std::abs(heights[x]-heights[x-1]);
    }
    for(int x=0;x<WIDTH;++x) {
        const int left=x?heights[x-1]:HEIGHT;
        const int right=x+1<WIDTH?heights[x+1]:HEIGHT;
        const int depth=std::max(0,std::min(left,right)-heights[x]);
        wells+=depth*(depth+1)/2;
    }
    const int danger=std::max(0,highest-12);
    return -.45*total-.65*highest-4.0*holes-.24*bump-.18*wells-.08*covered-2.0*danger*danger;
}

Evaluation BotEvaluator::evaluate(const State& state,const State* parent,
                                  const Evaluation* parent_evaluation,
                                  const Transition* transition) const {
    Evaluation result;
    if(state.dead){result.value=-1e12;return result;}
    const double heuristic=board_heuristic(state.board);
    double board_value=heuristic;
    const auto& options=impl_->options;
    if(impl_->model && options.mode!=EvaluationMode::Heuristic) {
        const auto& model=*impl_->model;
        spatial::Accumulator accumulator;
        if(parent && parent_evaluation && parent_evaluation->valid_accumulator && transition &&
           options.update!=UpdatePolicy::FullRefresh) {
            spatial::Accumulator previous;
            std::copy_n(parent_evaluation->accumulator.data(),spatial::MAX_POS,previous.pos.data());
            std::copy_n(parent_evaluation->accumulator.data()+spatial::MAX_POS,spatial::MAX_LOCAL,previous.local.data());
            spatial::Record record{};
            std::copy(parent->board.rows.begin(),parent->board.rows.end(),record.parent_rows);
            std::copy(state.board.rows.begin(),state.board.rows.end(),record.child_rows);
            std::copy(transition->delta.row_source.begin(),transition->delta.row_source.end(),record.row_source);
            std::copy(transition->delta.placed_cells.begin(),transition->delta.placed_cells.end(),record.placed_cells);
            record.clears=uint8_t(transition->delta.clears);
            // Kernel event dispatch needs zero vs nonzero. Custom room caps can
            // exceed 255; wrapping 256 to zero would skip a required refresh.
            record.garbage=uint8_t(std::min(transition->delta.garbage,255));
            record.clear_mask=transition->delta.clear_mask;
            const auto method=options.update==UpdatePolicy::RowSeams?
                spatial::Method::Seams:spatial::Method::RefreshEvents;
            spatial::transition(model,record,previous,accumulator,method);
        } else spatial::refresh(model,state.board.rows.data(),accumulator);
        std::copy(accumulator.pos.begin(),accumulator.pos.end(),result.accumulator.begin());
        std::copy(accumulator.local.begin(),accumulator.local.end(),result.accumulator.begin()+spatial::MAX_POS);
        result.valid_accumulator=true;
        // The trained network's next input is the NEXT PIECE TO PLACE. For a
        // post-lock State that is current, not queue.front(). At an unknown
        // visible horizon use the explicit board heuristic, not a hidden draw
        // or an untrained all-zero context vector.
        if(state.current!=Piece::None) {
            const double nnue=spatial::forward(model,accumulator,unsigned(state.current));
            board_value=options.mode==EvaluationMode::NNUE?nnue:
                options.nnue_weight*nnue+(1-options.nnue_weight)*heuristic;
        }
    }
    double pending=0,ready=0;
    for(const auto& packet:state.incoming) if(!packet.shielded) {
        pending+=packet.amount;
        if(packet.active && packet.spawn_ready)ready+=packet.amount;
    }
    const int danger=std::max(0,state.board.height()-12);
    // These battle-context terms are hand-set, not part of the old NNUE label.
    const double context=.4*std::min(state.b2b,20)+.15*std::min(state.combo,10)
        -.20*pending-.10*ready*danger;
    result.value=board_value+context;
    if(!std::isfinite(result.value))throw std::runtime_error("Non-finite evaluator output");
    return result;
}
double BotEvaluator::reward(const Transition& result) const {
    const auto& o=impl_->options;
    return o.sent_reward*result.sent+o.cancelled_reward*result.cancelled+
        o.line_reward*result.lines+o.pc_reward*int(result.perfect_clear);
}
} // namespace tetrio
