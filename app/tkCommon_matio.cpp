#define TIMER_ENABLE
#include "tkCommon/CmdParser.h"
#include "tkCommon/math/MatIO.h"
#include <thread>
#include <signal.h>

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "test matio");
    std::string matfile = cmd.addArg("matfile", "matlab.mat", "path of the mat file");
    cmd.print();
    
    tk::math::MatIO mat;
    mat.open(matfile);
    mat.stats();
    mat.readVar("key_45");

    return 0;
}