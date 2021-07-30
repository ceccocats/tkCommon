#include "tkCommon/CmdParser.h"
#include "tkCommon/rt/Task.h"
#include "tkCommon/rt/Thread.h"
#include "tkCommon/rt/Profiler.h"

bool gRun = true;
void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

void *task(void *data) {

    tk::rt::Task t;
    t.init(1000);
    while (gRun) {
        tkPROF_tic(task);
        volatile int fake = 0;
        for(int i=0; i<10000; i++)
            fake = i*2;
        tkPROF_toc(task);
        
        t.wait();
    }
    t.printStats();
    pthread_exit(NULL);
}

int main( int argc, char** argv){
    signal(SIGINT, sig_handler);
    tk::common::CmdParser cmd(argv, "RT task test");
    cmd.parse();

    tk::rt::Thread th("task", 0);
    th.init(task, NULL);
    th.printInfo();
    sleep(1);
    gRun = false;
    th.join();

    tkPROF_print;
    return 0;
}