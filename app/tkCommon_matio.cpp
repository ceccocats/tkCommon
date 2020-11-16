#include "tkCommon/CmdParser.h"
#include "tkCommon/math/MatIO.h"
#include <thread>
#include <signal.h>

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "test matio");
    std::string matfile = cmd.addArg("matfile", "matlab.mat", "path of the mat file");
    std::string varname = cmd.addOpt("-var", "", "var to read");
    int recursive_limit = cmd.addIntOpt("-r", -1, "number of recursive print (-1 infinite)");
    int list_limit = cmd.addIntOpt("-l", -1, "max number of var to print (-1 infinite)");
    cmd.parse();

    tk::math::MatIO mat;
    mat.open(matfile);
    mat.stats();

    if(varname != "") {
        std::vector<std::string> path = splitString(varname, '.');
        tkASSERT(path.size() >= 1);

        tk::math::MatIO::var_t initial_var, var;
        for(int i=0; i< path.size(); i++) {
            if(i==0) {
                if(!mat.read(path[i], initial_var)) {
                    tkFATAL("var not found")
                }
                var = initial_var;
            } else {
                var = var[path[i]];
            }
        }
        var.print(0, recursive_limit, list_limit);
        initial_var.release();
    } else {
        for(int i=0; i<mat.size(); i++){
            tk::math::MatIO::var_t var;
            mat.read(mat[i], var);
            var.print(0, recursive_limit, list_limit);
            var.release();
        }
    }
    mat.close();


    return 0;
}