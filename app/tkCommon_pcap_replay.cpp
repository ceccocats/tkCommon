#include <iostream>
#include <tkCommon/common.h>
#include <csignal>
#include <tkCommon/communication/EthInterface.h>

bool                            gRun = true;
tk::communication::Ethinterface iface;
tk::communication::UDPSocket    sender;

void replayLoop(std::string file, int port, std::string filter,std::string ip){

    tkASSERT(iface.initPcapReplay(file,filter));
    sender.initSender(port,ip);
    uint8_t buffer[2000];
    int count = 0;
    timeStamp_t prec = 0, now;

    while(gRun){      

        int n = iface.read(buffer,now);

        if(n == -1){
            gRun = false;
            continue;
        }

        sender.send(buffer,n);

        if(prec != 0){

            if(now > prec){
                
                timeStamp_t sleep_for = (now-prec);
                usleep(sleep_for);
            }
        }

        prec = now;

        count++;
        if(count%500 == 0){

            tk::tformat::printMsg("pcapReplay", std::string{"send "}+std::to_string(count)+" packets\n");
        }
    }

    iface.close();
    sender.close();
}

int main(int argc, char* argv[]){

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    std::string file        = cmd.addArg("file", "", "pcap replay file");
    std::string filter      = cmd.addOpt("-filter", "", "pcap filter");
    std::string ip          = cmd.addOpt("-ip", "127.0.0.1", "pcap filter");
    int port                = cmd.addIntOpt("-port", 2368, "pcap replay file");
    cmd.parse();

    tk::exceptions::handleSegfault();

    replayLoop(file,port,filter,ip);

    return 0;
}
