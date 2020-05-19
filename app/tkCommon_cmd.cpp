#include "tkCommon/CmdParser.h"
#include <thread>
#include <signal.h>


int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "test cmd parser");
    std::string arg0 = cmd.addArg("mappath", "map/", "path of the map to load");
    std::string arg1 = cmd.addArg("datapath", "", "path of the odom data to replay");
    std::vector<std::string> arg2 = cmd.addArgList("list", {"1", "2"}, "list of somethig");
    std::string opt0 = cmd.addOpt("-conf", "conf.yaml", "configuration yaml file");
    bool        opt1 = cmd.addBoolOpt("-v", "enable verbose");
    int         opt2 = cmd.addIntOpt("-num0", 23, "a number");
    float       opt3 = cmd.addFloatOpt("-num1", 12.34, "another number");
    cmd.parse();

    std::cout<<"inserted params:\n";
    std::cout<<arg0<<"\n";
    std::cout<<arg1<<"\n";
    for(int i=0; i<arg2.size(); i++)
        std::cout<<arg2[i]<<" ";
    std::cout<<"\n";
    std::cout<<opt0<<"\n";
    std::cout<<opt1<<"\n";
    std::cout<<opt2<<"\n";
    std::cout<<opt3<<"\n";
    return 0;
}