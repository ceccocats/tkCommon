#include <iostream>
#include <tkCommon/common.h>
#include <csignal>
#include <tkCommon/communication/EthInterface.h>
#include <tkCommon/exceptions.h>

bool recorder, gRun = true;
tk::communication::Ethinterface iface;

void recorderLoop(std::string file, std::string interface, std::string filter){

    iface.record(file,interface,filter);
}

void replayLoop(std::string file, std::string filter){

    tkASSERT(iface.initPcapReplay(file,filter));
    uint8_t buffer[2000];

    while(gRun){

        timeStamp_t time;
        int n = iface.read(buffer,time);

        if(n == -1){

            gRun = false;
        }else{

            tk::common::hex_dump(std::cout, buffer, n);

            std::cout<<"Timestamp: "<<(long int)time<<"\n\n";
        }
    }

    iface.close();
}


void signal_handler(int signal)
{
    std::cout<<"\nRequest closing..\n";
    if(gRun == false){
        exit(-1);
    }
    gRun = false;

    if(recorder == true){
        std::cout<<iface.recordStat()<<std::endl;
        iface.close();
    }
}

int main(int argc, char* argv[]){
    
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    tk::exceptions::handleSegfault();

    std::string             file,filter,interface;

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    recorder                = cmd.addBoolOpt("-rec", "true if you want to rec, false replay");
    file                    = cmd.addOpt("-file", "", "reciver, recorder, sender");
    filter                  = cmd.addOpt("-filter", "", "pcap filter");
    interface               = cmd.addOpt("-interface", "eth0", "iface");
    cmd.parse();

    if( recorder == true){

        std::cout<<"Recording..\n";
        recorderLoop(file,interface,filter);
    }else{

        std::cout<<"Replay...\n";
        replayLoop(file,filter);
    }

    return 0;
}