#include "tkCommon/communication/ethernet/TCPSocket.h"

bool 
tk::communication::TCPSocket::initClient(const int port, const std::string ip){

    //Set socket struct
    memset(&this->sock_addr, 0, sizeof(this->sock_addr));
    this->sock_addr.sin_family          = AF_INET; // IPv4
    this->sock_addr.sin_port            = htons(port);

    if(this->sock_client){
        this->sock_addr.sin_addr.s_addr    = htonl(INADDR_ANY);
    }else{
        this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());
    }

    // open socket
    this->sock_fd = socket(AF_INET, SOCK_STREAM, AF_INET);
    if (this->sock_fd < 0){
        clsErr("error while opening socket.\n");
        perror("TCP error");
        return false;
    }

    // bind socket
    int r = bind(this->sock_fd, (const struct sockaddr *)&this->sock_addr,  sizeof(this->sock_addr));
    if (r < 0) {
        clsErr("error while binding the socket.\n");
        perror("UDP error");
        return false;
    }
    
    // listen socket
    r = connect(this->sock_fd, (const struct sockaddr *)&this->sock_addr,  sizeof(this->sock_addr));
        if (r < 0) {
        clsErr("error while connecting to the socket.\n");
        perror("TCP error");
        return false;
    }

    if(this->sock_client != 1){
        clsSuc(std::string{"client TCP connected to server "}+ip+"\n")
    }

    return true;
}

bool 
tk::communication::TCPSocket::initServer(const int port){

    this->sock_client = 1;
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
    return true;    
}

int 
tk::communication::TCPSocket::receive(uint8_t* buffer, int length){

    if(this->sock_client == -1)
        return ::read(this->sock_fd, buffer, length); 
    else
        return ::read(this->sock_client, buffer, length);     
}

bool 
tk::communication::TCPSocket::send(uint8_t* buffer, int length){

    if(this->sock_client == -1)
        return ::send(this->sock_fd, buffer, length, 0) > 0; 
    else
        return ::send(this->sock_client, buffer, length, 0) > 0; 
}

bool 
tk::communication::TCPSocket::close(){

    ::close(this->sock_client);
    return ::close(this->sock_fd) > 0;
}