#include "protocol.hpp"
#include <iostream>

int main(int argc,char** argv) {
    try {
        std::string model;tetrio::EvaluatorOptions options;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") {
                std::cout<<"Usage: tetrio-bot [--model path.snn] [--mode heuristic|hybrid|nnue]\n"
                    "JSONL placement engine: one request and response per line.\n"
                    "Send {\"cmd\":\"info\"} for rules and supported commands.\n";return 0;
            }
            if((arg=="--model"||arg=="--mode")&&i+1<argc) {
                const std::string value=argv[++i];
                if(arg=="--model")model=value;
                else options=tetrio::parse_evaluator_options({{"mode",value}},options);
            } else throw std::invalid_argument("Unknown or incomplete command-line argument: "+arg);
        }
        tetrio::Protocol protocol(model,options);
        for(std::string line;std::getline(std::cin,line);) {
            if(line.empty())continue;
            tetrio::Json request,reply;
            try {
                if(line.size()>4*1024*1024)throw std::invalid_argument("request exceeds 4 MiB");
                request=tetrio::Json::parse(line);reply=protocol.handle(request);
            } catch(const std::exception& error) {
                reply={{"ok",false},{"error",error.what()}};
                if(request.is_object()&&request.contains("id"))reply["id"]=request.at("id");
            }
            std::cout<<reply.dump()<<'\n'<<std::flush;
        }
    } catch(const std::exception& error) {std::cerr<<"tetrio-bot: "<<error.what()<<'\n';return 1;}
    return 0;
}
