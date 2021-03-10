#include "tkCommon/communication/ethernet/TCPSocket.h"

bool 
tk::communication::TCPSocket::initClient(const int port, const std::string ip, bool readTimeout){

    //ret client
    this->sock_client = 1;

    //Set socket struct
    memset(&this->sock_addr, 0, sizeof(this->sock_addr));
    this->sock_addr.sin_family          = AF_INET; // IPv4
    this->sock_addr.sin_port            = htons(port);

    this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());

    // open socket
    this->sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (this->sock_fd < 0){
        tkERR("error while opening socket.\n");
        perror("TCP error");
        return false;
    }

    //Set 1 sec of timeout
    if(readTimeout == true){
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        setsockopt(this->sock_fd, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeout));
    }
    
    // listen socket
    int r = connect(this->sock_fd, (const struct sockaddr *)&this->sock_addr,  sizeof(this->sock_addr));
        if (r < 0) {
        tkERR("error while connecting to the socket.\n");
        perror("TCP error");
        return false;
    }

    //tkMSG(std::string{"Connected to "}+ip+" on port "+std::to_string(port)+"\n")

    return true;
}

bool 
tk::communication::TCPSocket::initServer(const int port){
    
    std::cout<<"i'm sorry\n";
    return false;
    //TODO: check and implemet

    /*this->sock_client = 1;
    bool ret = initClient(port,"");
    if(!ret){
        return false;
    }

    struct sockaddr client;
    memset(&client, 0, sizeof(client));

    clsWrn("Server attending connection..\n")

    this->sock_client = ::accept(this->sock_fd, (struct sockaddr *)&client, (socklen_t*)sizeof(client));
    if(this->sock_client < 0){
        clsErr("error while connecting to the socket.\n");
        perror("TCP error");
        return false;
    }

    clsWrn(std::string{"Server accept connection from "}+client.sa_data+"\n")
    return true;    */
}

int 
tk::communication::TCPSocket::receive(uint8_t* buffer, int length){

    if(this->sock_client == 1)
        return ::read(this->sock_fd, buffer, length);
    
    return 0;

}

bool 
tk::communication::TCPSocket::send(uint8_t* buffer, int length){

    if(this->sock_client == 1)
        return ::send(this->sock_fd, buffer, length, 0) > 0;

    return false;

}

bool 
tk::communication::TCPSocket::close(){

    if(this->sock_client == 1)
        return ::close(this->sock_fd) > 0;
    
    return false;

}