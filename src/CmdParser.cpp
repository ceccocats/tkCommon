#include "tkCommon/CmdParser.h"
#include "tkCommon/argh.h"

namespace tk { namespace common {

CmdParser::CmdParser(char **argv, std::string info) {
    argv_ptr = argv;
    argv_mode = argh::parser::Mode::PREFER_PARAM_FOR_UNREG_OPTION;
    generalInfo = info;

    // add help opt
    addBoolOpt("-h", "print help");
}

void CmdParser::setGeneralInfo(std::string info) {
    generalInfo = info;
}

std::string CmdParser::addArg(std::string name, std::string default_val, std::string info) {
    argh::parser cmdl(argv_ptr, argv_mode);
    Arg a;
    a.name = name;
    a.info = info;
    a.default_val = cmdl[args.size() + 1].empty() ? default_val : cmdl[args.size() + 1];
    a.optional = default_val.empty();
    args.push_back(a);
    return a.default_val;
}

bool CmdParser::addBoolOpt(std::string opt, std::string info) {
    if(opt.size() <= 1 || opt[0] != '-') {
        std::cout<<"option must start with '-', this is not valid: "<<opt<<"\n";
        exit(1);
    }

    argh::parser cmdl(argv_ptr, argv_mode);

    if(!cmdl(opt).str().empty()) {
        std::cout<<"ERROR: you cant append values to boolean flag: "<<opt<<"\n";
        exit(-1);
    }

    Opt o;
    o.name = opt;
    o.info = info;
    o.default_val_bool = cmdl[opt];
    o.isbool = true;
    opts.push_back(o);
    return o.default_val_bool;
}

std::string CmdParser::addOpt(std::string opt, std::string default_val, std::string info) {
    if(opt.size() <= 1 || opt[0] != '-') {
        std::cout<<"option must start with '-', this is not valid: "<<opt<<"\n";
        exit(1);
    }

    argh::parser cmdl(argv_ptr, argv_mode);
    Opt o;
    o.name = opt;
    o.info = info;
    o.default_val_str = cmdl(opt).str().empty() ? default_val : cmdl(opt).str();
    o.isbool = false;
    opts.push_back(o);
    return o.default_val_str;
}

void CmdParser::printUsage(std::string name) {
    std::cout << "usage: " << name;
    for (int i = 0; i < args.size(); i++)
        std::cout << " " << (args[i].optional ? "[" : "<") << args[i].name << (args[i].optional ? "]" : ">");
    std::cout << "\n";

    std::cout << "Args:\n";
    for (int i = 0; i < args.size(); i++) {
        std::string def_str = args[i].default_val;
        if(def_str.size() > DEFAULTW)
            def_str = def_str.substr(def_str.size()-DEFAULTW,def_str.size());
        def_str = "[ " + def_str + " ]";
        std::cout << "  " << std::left << std::setw(ARGSW) << args[i].name;
        std::cout << std::left << std::setw(DEFAULTW+5) << def_str;
        std::cout << args[i].info << "\n";
    }
    std::cout << "Opts:\n";
    for (int i = 0; i < opts.size(); i++) {
        std::string def_str = std::string(opts[i].isbool ? std::to_string(opts[i].default_val_bool) :
                                          opts[i].default_val_str);
        if(def_str.size() > DEFAULTW)
            def_str = def_str.substr(def_str.size()-DEFAULTW,def_str.size());
        def_str = "[ " + def_str + " ]";
        std::cout << "  " << std::left << std::setw(ARGSW) << opts[i].name;
        std::cout << std::left << std::setw(DEFAULTW+5)
                  << def_str;
        std::cout << opts[i].info << "\n";
    }
}

void CmdParser::print() {
    argh::parser cmdl(argv_ptr, argv_mode);

    // check unknown flags
    std::vector<std::string> flags;
    for (auto& f : cmdl.flags())
        flags.push_back(f);
    for (auto& f : cmdl.params())
        flags.push_back(f.first);
    for (std::string flag : flags) {
        bool found = false;
        for(int i=0; i<opts.size(); i++) {
            if(opts[i].name.substr(1,opts[i].name.size()) == flag)
                found = true;
        }
        if(!found) {
            std::cout<<"ERROR: flag not valid: -"<<flag<<"\n";
            exit(1);
        }
    }

    if (!generalInfo.empty()) {
        std::cout << generalInfo << "\n";
    }
    if (cmdl["-h"]) {
        printUsage(cmdl[0]);
        exit(1);
    }
}


}}