#include "tkCommon/communication/ethernet/TCPSocket.h"

bool 
tk::communication::TCPSocket::initClient(const int port, const std::string ip, bool readTimeout){

    //ret client
    this->tcp_type = 1;

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
    
    int len;
  
    // socket create and verification
    server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == -1) {
        tkERR("socket creation failed...\n");
        return false;
    }
    else
        printf("Socket successfully created..\n");
    bzero(&server_addr, sizeof(server_addr));
  
    // assign IP, PORT
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(port);
  
    // Binding newly created socket to given IP and verification
    if ((::bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr))) != 0) {
        tkERR("socket bind failed...\n");
        return false;    
    }
    else
        tkDBG("Socket successfully binded..\n");
  
    // Now server is ready to listen and verification
    if ((::listen(server_fd, 5)) != 0) {
        tkERR("Listen failed...\n");
        return false;
    }
    else
        tkDBG("Server listening..\n");
    len = sizeof(sock_addr);
  
    // Accept the data packet from client and verification
    sock_fd = ::accept(server_fd, (struct sockaddr*)&sock_addr, (socklen_t*) &len);
    if (sock_fd < 0) {
        tkERR("server acccept failed...\n");
        return false;
    }
    else
        tkMSG("server acccept the client...\n");
  
    this->tcp_type = 2;
    return true;   
}

int 
tk::communication::TCPSocket::receive(uint8_t* buffer, int length){

    if(this->tcp_type == 1 || this->tcp_type == 2)
        return ::read(this->sock_fd, buffer, length);
    
    return 0;

}

bool 
tk::communication::TCPSocket::send(uint8_t* buffer, int length){

    if(this->tcp_type == 1 || this->tcp_type == 2)
        return ::send(this->sock_fd, buffer, length, 0) > 0;

    return false;

}

bool 
tk::communication::TCPSocket::close(){

    if(this->tcp_type == 1 || this->tcp_type == 2)
        return ::close(this->sock_fd) > 0;
    
    if(this->tcp_type == 2)
        return ::close(this->server_fd) > 0;
    return false;

}