#include "tkCommon/CmdParser.h"
#include "tkCommon/communication/CanInterface.h"
#include "tkCommon/communication/can/VehicleCanParser.h"

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "Can utils");
    std::string dbc_file = cmd.addArg("dbc", "", "DBC can to parse");
    std::string soc_file = cmd.addOpt("-read", "can0", "socket to read, accepts also record files");
    cmd.parse();

    tkASSERT(dbc_file != "", "you must provide a dbc file");

    tk::communication::VehicleCanParser vehparser;
    vehparser.init(dbc_file);

    tk::communication::CanInterface canSoc;
    if(soc_file.compare(soc_file.size()-3, 3, "pcap") == 0 || 
       soc_file.compare(soc_file.size()-3, 3, "log")  == 0) {
        canSoc.initFile(soc_file);
    } else {
        canSoc.initSocket(soc_file);
    }

    tk::data::VehicleData veh;
    tk::data::CanData_t data;
    while(canSoc.read(&data)) {
        vehparser.parse(data, veh);
    }

    canSoc.close();
    return 0;
}