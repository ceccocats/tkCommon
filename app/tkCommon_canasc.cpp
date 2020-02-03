#include "tkCommon/CmdParser.h"
#include "tkCommon/communication/CanInterface.h"
#include <thread>
#include <signal.h>


int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "Convert canlog to asc");
    std::string in_file = cmd.addArg("in_file", "can0.log", "path the file to convert");
    std::string out_file = cmd.addArg("out_file", in_file + ".asc", "path of the result file");
    int interface = cmd.addIntOpt("-i", 0, "Can interface");
    cmd.print();

    FILE *canOut = fopen(out_file.c_str(), "w");
    if(canOut == NULL) {
        perror("Open can file");
        return 1;
    }


    tk::communication::CanInterface canIn;
    canIn.initFile(in_file);

    timeStamp_t initialTs;
    tk::data::CanData_t data;
    int i;
    for(i=0; canIn.read(&data); i++) {
        if(i==0) {
            time_t tv_sec = data.stamp / 1e6;
            initialTs = tv_sec * 1e6;
            std::cout<<"Initial timestamp: "<<initialTs<<"\n";

            fprintf(canOut, "date %s", ctime(&tv_sec));
            fprintf(canOut, "base hex  timestamps absolute\n");
            fprintf(canOut, "no internal events logged\n");
        }

        double time = double(data.stamp - initialTs) / 1e6;
        int intPart = int(time);
        int decPart = int( (time - intPart)*1e6 );
        fprintf(canOut, "%4d.%06d %d  %-15X Rx   d %d", intPart, decPart, interface, data.frame.can_id, data.frame.can_dlc);
        for(int j=0; j<data.frame.can_dlc; j++)
            fprintf(canOut, " %02X", data.frame.data[j]);
        fprintf(canOut, "\n");
    }
    std::cout<<"saved "<<i<<" Entries to "<<out_file<<"\n";
    return 0;
}