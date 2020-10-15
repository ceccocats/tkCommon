#include "tkCommon/CmdParser.h"
#include "tkCommon/rt/Task.h"
#include "tkCommon/rt/Thread.h"

bool gRun = true;
void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

void *task(void *data) {
    tk::rt::Task t;
    t.init(1000);
    while (gRun) {
        t.wait();
    }
    t.printStats();
}

int main( int argc, char** argv){
    signal(SIGINT, sig_handler);
    tk::common::CmdParser cmd(argv, "RT task test");
    cmd.parse();

    tk::rt::Thread th;
    th.init(task, NULL);
    sleep(1);
    gRun = false;
    th.join();
    return 0;
}