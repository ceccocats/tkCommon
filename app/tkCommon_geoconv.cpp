#include "tkCommon/CmdParser.h"
#include <thread>
#include <signal.h>


int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "Convert GEO data to XYZ, first line is the reference");
    std::string in_file = cmd.addArg("in_file", "", "path the file to convert");
    std::string out_file = cmd.addArg("out_file", in_file + "_converted", "path of the result file");
    cmd.parse();

    if(in_file == "") {
        std::cout<<"you must provide a txt file\n";
        return 1;
    }

    tk::common::GeodeticConverter geoConv;


    std::ifstream input(in_file);
    std::ofstream output(out_file);
    for(;;) {
        double a,b,c;
        input>>a>>b>>c;

        if(!input)
            break;

        if(!geoConv.isInitialised()) {
            geoConv.initialiseReference(a, b, c);
            output<<std::setprecision(20)<<"Ref: "<<a<<" "<<b<<" "<<c<<"\n";
            continue;
        }

        double x, y, z;
        geoConv.geodetic2Enu(a, b, c, &x, &y, &z);

        output<<std::setprecision(20)<<x<<" "<<y<<" "<<z<<"\n";
    }

    return 0;
}