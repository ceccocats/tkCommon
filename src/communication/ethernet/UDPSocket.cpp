#include "tkCommon/communication/ethernet/UDPSocket.h"

namespace tk { namespace communication {
    UDPSocket::UDPSocket() = default;


    UDPSocket::~UDPSocket() = default;


    bool
    UDPSocket::initReceiver(const int port, const std::string ip) {

        memset(&this->sock_addr, 0, sizeof(this->sock_addr));
        
        bool status = initSender(port, ip);

        if(status == false){
            return false;
        }

        // Bind the socket with the sender address
        int r = bind(this->sock_fd, (const struct sockaddr *)&this->sock_addr,  sizeof(this->sock_addr));
        if (r < 0) {
            clsErr("error while binding the socket.\n");
            return false;
        }

        reciver = true;
        return true;
    }

    bool
    UDPSocket::initSender(const int port, const std::string ip) {

        this->sock_addr.sin_family          = AF_INET; // IPv4
        this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());
        this->sock_addr.sin_port            = htons(port);

        if(ip.empty()){

             this->sock_addr.sin_addr.s_addr    = INADDR_ANY;
        }else{

            this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());
        }

        // open socket
        this->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (this->sock_fd < 0){
            clsErr("error while opening socket.\n");
            return false;
        }

        // allow multiple sockets
        if (!ip.empty()) {
            if (isMulticast(ip)) {

                int optname = 1;
                int r = setsockopt(this->sock_fd, SOL_SOCKET, SO_REUSEADDR, (char*) &optname, sizeof(optname));
                if (r < 0){
                    clsErr("error while allowing multiple sockets.\n");
                    return false;
                }
                // join multicast group
                struct ip_mreq mreq;
                mreq.imr_multiaddr.s_addr = inet_addr(ip.c_str());
                mreq.imr_interface.s_addr = htonl(INADDR_ANY);
                r = setsockopt(this->sock_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *) &mreq, sizeof(mreq));
                if (r < 0) {   
                    clsErr("error while joining multicast group.")
                    return false;
                }

                clsSuc("multicast socket created.\n");
            } else

                clsSuc("classic socket created.\n");
        }else{

            clsSuc("socket port created.\n");
        }

        reciver = false;
        return true;
    }


    int
    UDPSocket::receive(uint8_t *buffer, int length) {
        if (reciver == true)
            return recvfrom(this->sock_fd, buffer, (size_t) length, MSG_WAITALL, nullptr, nullptr);
        else
            return -1;
    }

    bool
    UDPSocket::send(uint8_t *buffer, int length) {
        if (reciver == false) {
            int r = sendto(this->sock_fd, buffer, (size_t) length, 0, (sockaddr *) &this->sock_addr,
                           sizeof(this->sock_addr));
            return r != -1;
        } else
            return false;
    }


    bool
    UDPSocket::close() {
        return ::close(this->sock_fd);
    }

    bool
    UDPSocket::isMulticast (const std::string ip) {
        return (ntohl(inet_addr(ip.c_str())) & 0xF0000000) == 0xE0000000;
    }
}}