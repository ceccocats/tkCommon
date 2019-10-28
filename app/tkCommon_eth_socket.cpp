#include <iostream>
#include <tkCommon/common.h>
#include <csignal>
#include <tkCommon/communication/ethernet/UDPSocket.h>

bool gRun = true;

void reciverLoop(int port, std::string ip){

    tk::communication::UDPSocket socket;

    socket.initReceiver(port,ip);
    uint8_t buffer[1000];

     while(gRun){

        int num = socket.receive(buffer,1000);
        if (num == -1){

            std::cout<<"Socket Error\n";
            return;
        }

        tk::common::hex_dump(std::cout, buffer, num);
    }

    socket.close();

}

void senderLoop(int port, std::string ip){

    tk::communication::UDPSocket socket;

    socket.initSender(port,ip);
    uint8_t buffer[1000];

    while(gRun){

        std::cin>>buffer;

        socket.send(buffer,(int)strlen((const char *)buffer));
    }

    socket.close();
}


void signal_handler(int signal)
{
    std::cout<<"\nRequest closing..\n";
    if(gRun == false){
        exit(-1);
    }
    gRun = false;
}

int main(int argc, char* argv[]){
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::string             type, ip;
    int                     port;

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    type                    = cmd.addOpt("-type", "reciver" ,"reciver, sender");
    ip                      = cmd.addOpt("-ip", "", "ip multicast only in case of multicast type");
    port                    = atoi(cmd.addArg("-port", "40000", "port").c_str());
    cmd.print();



    if( type == "reciver"){
        std::cout<<"Reciver...\n";
        reciverLoop(port,ip);
    }

    if( type == "sender"){
        std::cout<<"Sender...\n";
        senderLoop(port,ip);
    }

    return 0;
}