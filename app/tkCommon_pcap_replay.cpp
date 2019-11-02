#include <iostream>
#include <tkCommon/common.h>
#include <csignal>
#include <tkCommon/communication/EthInterface.h>
#include <tkCommon/exceptions.h>

bool                            gRun = true;
tk::communication::Ethinterface iface;
tk::communication::UDPSocket    sender;

void replayLoop(std::string file, int port){

    tkASSERT(iface.initPcapReplay(file));
    sender.initSender(port,"127.0.0.1");
    uint8_t buffer[2000];
    int count = 0;
    timeStamp_t prec = 0, now, startComputation;

    while(gRun){      

        startComputation = getTimeStamp();

        int n = iface.read(buffer,now);

        sender.send(buffer,n);

        if(prec != 0){

            timeStamp_t sleep_for = (now-prec); - (getTimeStamp() - startComputation);
            usleep(sleep_for);
        }

        prec = now;

        count++;
        if(count%500 == 0){

            tk::tcolor::printMsg("pcapReplay", std::string{"send "}+std::to_string(count)+" packets\n");
        }


    }

    iface.close();
    sender.close();
}

int main(int argc, char* argv[]){

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    std::string file        = cmd.addOpt("-file", "", "pcap replay file");
    int port                = cmd.addIntOpt("-port", 2386, "pcap replay file");
    cmd.print();

    tk::exceptions::handleSegfault();

    replayLoop(file,port);

    return 0;
}