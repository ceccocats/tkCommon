#include "tkCommon/CmdParser.h"
#include "tkCommon/rt/Task.h"

bool gRun = true;
void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}


int main( int argc, char** argv){
    signal(SIGINT, sig_handler);
    tk::common::CmdParser cmd(argv, "RT task test");
    int period = cmd.addIntOpt("-p", 100, "task period (us)");
    cmd.parse();

    tk::rt::Task t;
    t.init(period);
    while (gRun) {
        t.wait();
        std::cout<<"tick "<<t.last_ts<<"\n";        
    }
    t.printStats();

    return 0;
}