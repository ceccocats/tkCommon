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
    if(gRun == false){
        std::cout<<"\nForced closing.\n";
        exit(-1);
    }
    std::cout<<"\nRequest closing..\n";
    gRun = false;
}

int main(int argc, char* argv[]){
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    std::string  type       = cmd.addOpt("-type", "reciver" ,"reciver, sender");
    std::string  ip         = cmd.addOpt("-ip", "", "ip");
    int port                = cmd.addIntOpt("-port", 40000, "port");
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