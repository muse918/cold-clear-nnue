// First-layer-only post-training quantization experiment.
// The small head deliberately stays float32, allowing a controlled comparison.
// Build: g++ -O3 -std=c++17 -march=native quantized.cpp -o quantized
// --repeats 0 validates/predicts without collecting timing measurements.
#define SPATIAL_KERNEL_LIBRARY
#include "kernels.cpp"
#include <type_traits>

namespace quantized {
using spatial::Model;
using spatial::Record;
using spatial::Method;
constexpr int W=spatial::W,H=spatial::H,PX=spatial::PX,PY=spatial::PY;

struct Options {
    std::string model,data,output,predictions;
    int bits=8,repeats=0;
    size_t limit=0;
};
Options parse(int argc,char**argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string k=argv[i];
        if(k=="--help") {
            std::cout<<"--model model.snn --data records.bin --output quantized.json "
                       "[--predictions quantized.f32] [--bits 8|16] [--limit N] [--repeats 0]\n";
            std::exit(0);
        }
        if(i+1==argc)throw std::runtime_error("missing argument for "+k);
        std::string v=argv[++i];
        if(k=="--model")o.model=v;else if(k=="--data")o.data=v;
        else if(k=="--output")o.output=v;else if(k=="--predictions")o.predictions=v;
        else if(k=="--bits")o.bits=std::stoi(v);else if(k=="--limit")o.limit=std::stoull(v);
        else if(k=="--repeats")o.repeats=std::stoi(v);else throw std::runtime_error("unknown option "+k);
    }
    if(o.model.empty()||o.data.empty()||o.output.empty()||(o.bits!=8&&o.bits!=16)||o.repeats<0)
        throw std::runtime_error("model, data, output and bits=8|16 are required");
    return o;
}

struct Bounds {
    std::vector<int64_t> pos_min,pos_max,local_min,local_max;
    int64_t global_min=0,global_max=0;
    double minimum_attention_denominator=std::numeric_limits<double>::infinity();
    bool fits16=true,fits32=true,attention_safe=true;
    void observe(int64_t lo,int64_t hi) {
        global_min=std::min(global_min,lo);global_max=std::max(global_max,hi);
        fits16=fits16 && lo>=-32768 && hi<=32767;
        fits32=fits32 && lo>=std::numeric_limits<int32_t>::min() && hi<=std::numeric_limits<int32_t>::max();
    }
};

int64_t rounded_bias(double v) {
    if(!std::isfinite(v)||std::abs(v)>double(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("quantized bias does not fit int32");
    return std::llround(v);
}

template<class Weight> struct QuantModel {
    const Model &m;
    std::vector<Weight> posw,lut;
    std::vector<int32_t> posb,localb;
    std::vector<float> pos_scale,local_scale;
    Bounds bounds;

    explicit QuantModel(const Model &model):m(model) {
        const int qmax=sizeof(Weight)==1?127:32767;
        posw.resize(m.posw.size());lut.resize(m.lut.size());
        posb.resize(m.posdim);pos_scale.resize(m.posdim);
        localb.resize(m.localdim);local_scale.resize(m.localdim);
        bounds.pos_min.resize(m.posdim);bounds.pos_max.resize(m.posdim);
        bounds.local_min.resize(m.localdim);bounds.local_max.resize(m.localdim);
        for(unsigned j=0;j<m.posdim;++j) {
            float maxabs=0;
            for(int i=0;i<W*H;++i)maxabs=std::max(maxabs,std::abs(m.posw[size_t(i)*m.posdim+j]));
            float scale=maxabs>0?maxabs/float(qmax):1.0f;
            pos_scale[j]=scale;
            posb[j]=int32_t(rounded_bias(double(m.posb[j])/scale));
            int64_t lo=posb[j],hi=posb[j];
            for(int i=0;i<W*H;++i) {
                auto k=size_t(i)*m.posdim+j;
                int q=int(std::llround(double(m.posw[k])/scale));
                q=std::max(-qmax,std::min(qmax,q));posw[k]=Weight(q);
                lo+=std::min(q,0);hi+=std::max(q,0);
            }
            bounds.pos_min[j]=lo;bounds.pos_max[j]=hi;bounds.observe(lo,hi);
        }
        for(unsigned j=0;j<m.localdim;++j) {
            float maxabs=std::abs(m.local_zero[j]);
            for(int p=0;p<4096;++p)maxabs=std::max(maxabs,std::abs(m.lut[size_t(p)*m.localdim+j]));
            float scale=maxabs>0?maxabs/float(qmax):1.0f;
            local_scale[j]=scale;
            // Quantize the complete empty-token baseline once. Empty residual is
            // exactly zero; each x-column pool always represents exactly37 tokens.
            localb[j]=int32_t(rounded_bias(double(PY)*m.local_zero[j]/scale));
            int mn=0,mx=0;
            for(int p=0;p<4096;++p) {
                auto k=size_t(p)*m.localdim+j;
                int q=int(std::llround(double(m.lut[k])/scale));
                q=std::max(-qmax,std::min(qmax,q));lut[k]=Weight(q);
                mn=std::min(mn,q);mx=std::max(mx,q);
            }
            int64_t lo=int64_t(localb[j])+int64_t(PY)*mn;
            int64_t hi=int64_t(localb[j])+int64_t(PY)*mx;
            bounds.local_min[j]=lo;bounds.local_max[j]=hi;bounds.observe(lo,hi);
            if(m.kind==2 && j%9==8) {
                double minimum=double(lo)*scale;
                bounds.minimum_attention_denominator=std::min(bounds.minimum_attention_denominator,minimum);
                if(lo<=0||!std::isfinite(minimum))bounds.attention_safe=false;
            }
        }
    }
};

template<class Acc> struct alignas(64) Accumulator {
    std::array<Acc,spatial::MAX_POS> pos{};
    std::array<Acc,spatial::MAX_LOCAL> local{};
};

template<class Weight,class Acc>
inline void add_cell(const QuantModel<Weight>&q,Accumulator<Acc>&a,unsigned cell,int sign) {
    const Weight*src=q.posw.data()+size_t(cell)*q.m.posdim;
    for(unsigned j=0;j<q.m.posdim;++j)a.pos[j]=Acc(int32_t(a.pos[j])+sign*int32_t(src[j]));
}
template<class Weight,class Acc>
inline void add_local(const QuantModel<Weight>&q,Accumulator<Acc>&a,int x,unsigned p,int sign) {
    if(!p)return;
    Acc*dst=a.local.data()+size_t(x)*q.m.localdim;
    const Weight*src=q.lut.data()+size_t(p)*q.m.localdim;
    for(unsigned j=0;j<q.m.localdim;++j)dst[j]=Acc(int32_t(dst[j])+sign*int32_t(src[j]));
}
template<class Weight,class Acc>
void refresh(const QuantModel<Weight>&q,const uint16_t*rows,Accumulator<Acc>&a) {
    for(unsigned j=0;j<q.m.posdim;++j)a.pos[j]=Acc(q.posb[j]);
    for(int y=0;y<H;++y) {
        unsigned bits=rows[y];
        while(bits){unsigned x=__builtin_ctz(bits);bits&=bits-1;add_cell(q,a,y*W+x,1);}
    }
    for(int x=0;x<PX;++x)
        for(unsigned j=0;j<q.m.localdim;++j)a.local[size_t(x)*q.m.localdim+j]=Acc(q.localb[j]);
    if(q.m.kind)for(int y=0;y<PY;++y)for(int x=0;x<PX;++x)
        add_local(q,a,x,spatial::pattern(rows,x,y),1);
}
template<class Weight,class Acc>
void transition(const QuantModel<Weight>&q,const Record&r,const Accumulator<Acc>&parent,
                Accumulator<Acc>&a,Method method) {
    if(method==Method::Full||(method==Method::RefreshEvents&&!spatial::ordinary_lock(r))) {
        refresh(q,r.child_rows,a);return;
    }
    std::copy_n(parent.pos.data(),q.m.posdim,a.pos.data());
    std::copy_n(parent.local.data(),PX*q.m.localdim,a.local.data());
    for(int y=0;y<H;++y) {
        unsigned removed=r.parent_rows[y]&~r.child_rows[y]&1023u;
        unsigned added=r.child_rows[y]&~r.parent_rows[y]&1023u;
        while(removed){unsigned x=__builtin_ctz(removed);removed&=removed-1;add_cell(q,a,y*W+x,-1);}
        while(added){unsigned x=__builtin_ctz(added);added&=added-1;add_cell(q,a,y*W+x,1);}
    }
    if(!q.m.kind)return;
    if(spatial::ordinary_lock(r)) {
        std::array<uint64_t,PX> dirty;spatial::dirty_lock_anchors(r,dirty);
        for(int x=0;x<PX;++x) {
            uint64_t ys=dirty[x];
            while(ys) {
                int y=__builtin_ctzll(ys);ys&=ys-1;
                unsigned op=spatial::pattern(r.parent_rows,x,y),np=spatial::pattern(r.child_rows,x,y);
                if(op==np)continue;
                Acc*dst=a.local.data()+size_t(x)*q.m.localdim;
                const Weight*oldv=q.lut.data()+size_t(op)*q.m.localdim;
                const Weight*newv=q.lut.data()+size_t(np)*q.m.localdim;
                // Remove first so both int32 intermediates are inside the
                // conservative subset bounds, then store the proven-safe sum.
                for(unsigned j=0;j<q.m.localdim;++j)
                    dst[j]=Acc(int32_t(dst[j])-int32_t(oldv[j])+int32_t(newv[j]));
            }
        }
    } else {
        uint64_t old_keep,new_keep;spatial::reusable_strips(r,old_keep,new_keep);
        constexpr uint64_t valid=(uint64_t(1)<<PY)-1;
        uint64_t old_dirty=valid&~old_keep,new_dirty=valid&~new_keep;
        while(old_dirty) {
            int y=__builtin_ctzll(old_dirty);old_dirty&=old_dirty-1;
            for(int x=0;x<PX;++x)add_local(q,a,x,spatial::pattern(r.parent_rows,x,y),-1);
        }
        while(new_dirty) {
            int y=__builtin_ctzll(new_dirty);new_dirty&=new_dirty-1;
            for(int x=0;x<PX;++x)add_local(q,a,x,spatial::pattern(r.child_rows,x,y),1);
        }
    }
}

template<class Weight,class Acc>
float forward(const QuantModel<Weight>&q,const Accumulator<Acc>&a,unsigned next) {
    spatial::Accumulator dequantized;
    for(unsigned j=0;j<q.m.posdim;++j)dequantized.pos[j]=float(a.pos[j])*q.pos_scale[j];
    for(int x=0;x<PX;++x)for(unsigned j=0;j<q.m.localdim;++j)
        dequantized.local[size_t(x)*q.m.localdim+j]=float(a.local[size_t(x)*q.m.localdim+j])*q.local_scale[j];
    return spatial::forward(q.m,dequantized,next);
}

template<class Acc>
size_t mismatches(const Model&m,const Accumulator<Acc>&a,const Accumulator<Acc>&b) {
    size_t n=0;
    for(unsigned j=0;j<m.posdim;++j)n+=a.pos[j]!=b.pos[j];
    for(unsigned j=0;j<PX*m.localdim;++j)n+=a.local[j]!=b.local[j];
    return n;
}

struct ChainParity {
    size_t transitions=0,segments=0,discontinuities=0,max_segment_length=0;
    std::array<size_t,3> failed_components{},failed_scores{};
};
template<class Weight,class Acc>
ChainParity verify_chains(const QuantModel<Weight>&q,const std::vector<Record>&records) {
    ChainParity report;
    bool have_previous=false;
    uint32_t previous_game=0;
    size_t length=0;
    std::array<uint16_t,H> previous_rows{};
    std::array<Accumulator<Acc>,3> carried;
    Accumulator<Acc> reference,updated;
    for(const auto&r:records) {
        if(!r.selected&&r.event!=1)continue;
        const bool same_game=have_previous&&previous_game==r.game_id;
        const bool contiguous=same_game&&std::equal(previous_rows.begin(),previous_rows.end(),r.parent_rows);
        if(!contiguous) {
            report.discontinuities+=same_game;
            ++report.segments;length=0;
            for(int k=1;k<3;++k)refresh(q,r.parent_rows,carried[k]);
        }
        ++report.transitions;
        report.max_segment_length=std::max(report.max_segment_length,++length);
        refresh(q,r.child_rows,reference);
        const float score=forward(q,reference,r.next);
        for(int k=1;k<3;++k) {
            transition(q,r,carried[k],updated,static_cast<Method>(k));
            report.failed_components[k]+=mismatches(q.m,reference,updated);
            report.failed_scores[k]+=score!=forward(q,updated,r.next);
            std::copy_n(updated.pos.data(),q.m.posdim,carried[k].pos.data());
            std::copy_n(updated.local.data(),PX*q.m.localdim,carried[k].local.data());
        }
        std::copy_n(r.child_rows,H,previous_rows.begin());
        previous_game=r.game_id;have_previous=true;
    }
    return report;
}

struct Rank {
    size_t groups=0,identical_top1=0,tolerant_top1=0,pairs=0,inverted_pairs=0,quantized_ties=0;
    double regret_sum=0,regret_max=0;
};
Rank rank_metrics(const std::vector<Record>&records,const spatial::Parents&parents,
                  const std::vector<float>&ref,const std::vector<float>&got) {
    Rank out;
    for(const auto &g:parents.groups) {
        if(records[g.representative].event!=0||g.children.size()<2)continue;
        ++out.groups;size_t best_ref=g.children[0],best_got=best_ref;
        auto fq=[&](size_t i){return double(records[i].reward)+ref[i];};
        auto qq=[&](size_t i){return double(records[i].reward)+got[i];};
        for(size_t i:g.children) {
            if(fq(i)>fq(best_ref))best_ref=i;
            if(qq(i)>qq(best_got))best_got=i;
        }
        out.identical_top1+=best_ref==best_got;
        const double regret=fq(best_ref)-fq(best_got);
        out.tolerant_top1+=regret<=1e-4;out.regret_sum+=regret;out.regret_max=std::max(out.regret_max,regret);
        for(size_t j=0;j<g.children.size();++j)for(size_t k=0;k<j;++k) {
            double rd=fq(g.children[j])-fq(g.children[k]);
            if(std::abs(rd)<=1e-6)continue;
            double qd=qq(g.children[j])-qq(g.children[k]);
            ++out.pairs;out.inverted_pairs+=rd*qd<0;out.quantized_ties+=std::abs(qd)<=1e-6;
        }
    }
    return out;
}

template<class T>void array_json(std::ostream&o,const std::vector<T>&v) {
    o<<'[';for(size_t i=0;i<v.size();++i){if(i)o<<',';o<<v[i];}o<<']';
}
std::string ratio_json(double numerator,size_t denominator) {
    if(!denominator)return "null";
    std::ostringstream out;out<<std::setprecision(12)<<numerator/denominator;return out.str();
}
template<class Weight>
void bounds_json(std::ostream&o,const QuantModel<Weight>&q) {
    const auto &b=q.bounds;
    o<<"{\"global_min\":"<<b.global_min<<",\"global_max\":"<<b.global_max
     <<",\"fits_i16\":"<<(b.fits16?"true":"false")<<",\"fits_i32\":"<<(b.fits32?"true":"false")
     <<",\"position_min\":";array_json(o,b.pos_min);o<<",\"position_max\":";array_json(o,b.pos_max);
    o<<",\"local_min\":";array_json(o,b.local_min);o<<",\"local_max\":";array_json(o,b.local_max);
    o<<",\"attention_denominator_lower_bound\":";
    if(q.m.kind==2)o<<b.minimum_attention_denominator;else o<<"null";
    o<<"}";
}

template<class Weight,class Acc>
spatial::Timing benchmark(const QuantModel<Weight>&q,const std::vector<Record>&records,
                         const spatial::Parents&parents,const std::vector<Accumulator<Acc>>&cache,
                         Method method,int repeats) {
    spatial::Timing result;result.count=records.size();
    std::vector<double> samples;
    Accumulator<Acc> child;
    for(int rep=0;rep<repeats;++rep) {
        double sum=0;
        auto start=spatial::Clock::now();
        for(size_t i=0;i<records.size();++i) {
            transition(q,records[i],cache[parents.group_of_record[i]],child,method);
            sum+=forward(q,child,records[i].next);
        }
        auto end=spatial::Clock::now();
        samples.push_back(std::chrono::duration<double,std::nano>(end-start).count()/records.size());
        spatial::benchmark_sink=spatial::benchmark_sink+sum;result.checksum=sum;
    }
    if(!samples.empty()) {
        result.median_ns=spatial::median(samples);
        result.min_ns=*std::min_element(samples.begin(),samples.end());
        result.max_ns=*std::max_element(samples.begin(),samples.end());
    }
    return result;
}

template<class Weight,class Acc>
int evaluate(const Options&o,const QuantModel<Weight>&q) {
    auto records=spatial::load_records(o.data,o.limit);
    if(records.empty())throw std::runtime_error("no records");
    const auto parents=spatial::prepare_parents(q.m,records);
    std::vector<Accumulator<Acc>> cache(parents.groups.size());
    for(size_t i=0;i<parents.groups.size();++i)
        refresh(q,records[parents.groups[i].representative].parent_rows,cache[i]);
    std::array<size_t,3> bad_components{},bad_scores{};
    std::vector<float> scores(records.size()),reference(records.size());
    std::vector<double> errors;errors.reserve(records.size());
    double sumabs=0,sumsq=0,maxerr=0;
    spatial::Accumulator fa;
    Accumulator<Acc> full,changed;
    for(size_t i=0;i<records.size();++i) {
        const auto &r=records[i];
        spatial::refresh(q.m,r.child_rows,fa);reference[i]=spatial::forward(q.m,fa,r.next);
        refresh(q,r.child_rows,full);scores[i]=forward(q,full,r.next);
        if(!std::isfinite(scores[i]))throw std::runtime_error("non-finite quantized prediction");
        for(int k=1;k<=2;++k) {
            transition(q,r,cache[parents.group_of_record[i]],changed,static_cast<Method>(k));
            bad_components[k]+=mismatches(q.m,full,changed);
            bad_scores[k]+=scores[i]!=forward(q,changed,r.next);
        }
        double e=std::abs(double(scores[i])-reference[i]);
        errors.push_back(e);sumabs+=e;sumsq+=e*e;maxerr=std::max(maxerr,e);
    }
    std::sort(errors.begin(),errors.end());
    const double p95=errors[std::min(errors.size()-1,size_t(.95*errors.size()))];
    const auto rank=rank_metrics(records,parents,reference,scores);
    const auto chain=verify_chains<Weight,Acc>(q,records);
    if(!o.predictions.empty()) {
        std::ofstream p(o.predictions,std::ios::binary);
        if(!p)throw std::runtime_error("cannot write predictions");
        p.write(reinterpret_cast<const char*>(scores.data()),std::streamsize(scores.size()*sizeof(float)));
    }
    std::ofstream out(o.output);
    if(!out)throw std::runtime_error("cannot write output");
    out<<std::setprecision(12)<<"{\n\"format\":\"spatial-first-layer-quant-v1\",\n"
       <<"\"accepted\":true,\"model_kind\":"<<q.m.kind<<",\"weight_bits\":"<<sizeof(Weight)*8
       <<",\"accumulator_bits\":"<<sizeof(Acc)*8<<",\n"
       <<"\"scope\":\"first-layer-only per-channel symmetric quantization; float32 activation/head and output scaling unchanged; no saturation\",\n"
       <<"\"rounding\":\"nearest with halfway away from zero; symmetric range excludes most-negative integer\",\n"
       <<"\"bounds_proof\":\"position bias plus each of400 weights min(0,w)/max(0,w); local37-empty-token baseline plus37*min/max residual. Bounds also cover remove-before-add subsets; replacement arithmetic uses int32.\",\n"
       <<"\"bounds\":";bounds_json(out,q);
    const size_t rawweights=(q.posw.size()+q.lut.size())*sizeof(Weight);
    const size_t biases=(q.posb.size()+q.localb.size())*sizeof(int32_t);
    const size_t scales=(q.pos_scale.size()+q.local_scale.size())*sizeof(float);
    const size_t active=(q.m.posdim+PX*q.m.localdim)*sizeof(Acc);
    const size_t head=(q.m.headw.size()+q.m.headb.size()+q.m.outw.size()+1)*sizeof(float);
    out<<",\n\"memory\":{\"first_layer_weights_bytes\":"<<rawweights
       <<",\"bias_bytes\":"<<biases<<",\"scale_bytes\":"<<scales
       <<",\"first_layer_total_bytes\":"<<rawweights+biases+scales
       <<",\"unchanged_float_head_bytes\":"<<head
       <<",\"total_parameter_payload_bytes\":"<<rawweights+biases+scales+head
       <<",\"active_accumulator_bytes\":"<<active<<",\"allocated_accumulator_bytes\":"<<sizeof(Accumulator<Acc>)
       <<",\"float_first_layer_weights_bytes\":"<<(q.m.posw.size()+q.m.lut.size())*sizeof(float)
       <<",\"float_model_serialized_bytes\":"<<q.m.serialized_bytes
       <<",\"storage_scope\":\"parameter payload, not resident process memory; validation harness retains the float master model\"},\n"
       <<"\"records\":"<<records.size()<<",\"parent_groups\":"<<parents.groups.size()<<",\n"
       <<"\"bit_exact_parity\":{\"refresh_events\":{\"failed_components\":"<<bad_components[1]
       <<",\"failed_scores\":"<<bad_scores[1]<<"},\"row_seams\":{\"failed_components\":"<<bad_components[2]
       <<",\"failed_scores\":"<<bad_scores[2]<<"}},\n"
       <<"\"chained_selected_parity\":{\"transitions\":"<<chain.transitions<<",\"segments\":"<<chain.segments
       <<",\"discontinuities\":"<<chain.discontinuities<<",\"max_segment_length\":"<<chain.max_segment_length
       <<",\"refresh_events\":{\"failed_components\":"<<chain.failed_components[1]<<",\"failed_scores\":"<<chain.failed_scores[1]
       <<"},\"row_seams\":{\"failed_components\":"<<chain.failed_components[2]<<",\"failed_scores\":"<<chain.failed_scores[2]<<"}},\n"
       <<"\"float_comparison\":{\"score_mae\":"<<sumabs/records.size()<<",\"score_rmse\":"<<std::sqrt(sumsq/records.size())
       <<",\"score_p95_abs\":"<<p95<<",\"score_max_abs\":"<<maxerr
       <<",\"rank_groups\":"<<rank.groups<<",\"top1_identical_fraction\":"<<ratio_json(rank.identical_top1,rank.groups)
       <<",\"top1_float_regret_le_1e_4_fraction\":"<<ratio_json(rank.tolerant_top1,rank.groups)
       <<",\"mean_float_choice_regret\":"<<ratio_json(rank.regret_sum,rank.groups)
       <<",\"max_float_choice_regret\":"<<ratio_json(rank.regret_max,rank.groups?1:0)
       <<",\"non_tied_float_pairs\":"<<rank.pairs<<",\"inverted_pair_fraction\":"<<ratio_json(rank.inverted_pairs,rank.pairs)
       <<",\"quantized_tie_fraction\":"<<ratio_json(rank.quantized_ties,rank.pairs)<<"},\n"
       <<"\"timing_scope\":\"cached parent to child including accumulator copy, dirty extraction, dequantization and unchanged float head; parent build excluded\",\n"
       <<"\"timings\":";
    if(o.repeats==0)out<<"null";
    else {
        out<<"{";
        for(int k=0;k<3;++k) {
            if(k)out<<',';
            auto method=static_cast<Method>(k);
            auto t=benchmark(q,records,parents,cache,method,o.repeats);
            out<<'"'<<spatial::method_name(method)<<"\":{\"median_ns\":"<<t.median_ns
               <<",\"min_ns\":"<<t.min_ns<<",\"max_ns\":"<<t.max_ns<<",\"checksum\":"<<t.checksum<<"}";
        }
        out<<"}";
    }
    out<<"\n}\n";
    std::cout<<"bits="<<sizeof(Weight)*8<<" acc="<<sizeof(Acc)*8<<" records="<<records.size()
             <<" mae="<<sumabs/records.size()<<" top1="<<ratio_json(rank.identical_top1,rank.groups)
             <<" bad_components="<<bad_components[1]+bad_components[2]<<" output="<<o.output<<'\n';
    return bad_components[1]||bad_components[2]||bad_scores[1]||bad_scores[2]||
        chain.failed_components[1]||chain.failed_components[2]||chain.failed_scores[1]||chain.failed_scores[2]?1:0;
}

template<class Weight>
int run(const Options&o,const Model&m) {
    QuantModel<Weight> q(m);
    const bool range_safe=sizeof(Weight)==1?q.bounds.fits16:q.bounds.fits32;
    if(!range_safe||!q.bounds.attention_safe) {
        std::ofstream out(o.output);
        if(!out)throw std::runtime_error("cannot write output");
        out<<std::setprecision(12)<<"{\"format\":\"spatial-first-layer-quant-v1\",\"accepted\":false,\"weight_bits\":"
           <<sizeof(Weight)*8<<",\"model_kind\":"<<m.kind<<",\"reason\":\""
           <<(!range_safe?(sizeof(Weight)==1?"int16 conservative range exceeded; retry the q16/int32 baseline":"int32 conservative range exceeded"):
                "attention denominator conservative lower bound is not positive")
           <<"\",\"bounds\":";bounds_json(out,q);out<<"}\n";
        std::cout<<"REJECTED unsafe quantization; output="<<o.output<<'\n';
        return 0;
    }
    if(sizeof(Weight)==1&&q.bounds.fits16)return evaluate<Weight,int16_t>(o,q);
    return evaluate<Weight,int32_t>(o,q);
}
} // namespace quantized

int main(int argc,char**argv) {
    try {
        const auto o=quantized::parse(argc,argv);
        const auto model=spatial::Model::load(o.model);
        return o.bits==8?quantized::run<int8_t>(o,model):quantized::run<int16_t>(o,model);
    } catch(const std::exception&e) {std::cerr<<"error: "<<e.what()<<'\n';return 1;}
}
