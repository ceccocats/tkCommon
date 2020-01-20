#include "tkCommon/communication/ethernet/UDPSocket.h"

namespace tk { namespace communication {
    UDPSocket::UDPSocket() = default;


    UDPSocket::~UDPSocket() = default;


    bool
    UDPSocket::initReceiver(const int port, const std::string ip) {
    
        memset(&this->sock_addr, 0, sizeof(this->sock_addr));

        this->sock_addr.sin_family          = AF_INET; // IPv4
        this->sock_addr.sin_port            = htons(port);

        if(ip.empty()){

             this->sock_addr.sin_addr.s_addr    = htonl(INADDR_ANY);
        }else{

            this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());
        }

        // open socket
        this->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (this->sock_fd < 0){
            clsErr("error while opening socket.\n");
            perror("UDP error");
            return false;
        }

        if (!ip.empty()) {
            if (isMulticast(ip)) {
                this->isMulti = true;
                this->imr.imr_multiaddr.s_addr = inet_addr(ip.c_str());
                this->imr.imr_interface.s_addr = htonl(INADDR_ANY);
                int r = setsockopt(this->sock_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &imr, sizeof(ip_mreq));
                int val = 1;
                r = setsockopt(this->sock_fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
                if (r < 0){
                    clsErr("error while allowing multiple sockets.\n");
                    perror("UDP error");
                    return false;
                }
                clsSuc("multicast socket created.\n");
            } else{

                this->isMulti = false;
                clsSuc("classic socket created.\n");
            }
        }else{

            clsSuc("socket port created.\n");
        }


        // Bind the socket with the sender address
        int r = bind(this->sock_fd, (const struct sockaddr *)&this->sock_addr,  sizeof(this->sock_addr));
        if (r < 0) {
            clsErr("error while binding the socket.\n");
            perror("UDP error");
            return false;
        }

        reciver = true;
        return true;
    }

    bool
    UDPSocket::initSender(const int port, const std::string ip, const int srcport) {

        memset(&this->sock_addr, 0, sizeof(this->sock_addr));

        this->sock_addr.sin_family          = AF_INET; // IPv4
        this->sock_addr.sin_port            = htons(port);

        if(ip.empty()){

             this->sock_addr.sin_addr.s_addr    = htonl(INADDR_ANY);
        }else{

            this->sock_addr.sin_addr.s_addr     = inet_addr(ip.c_str());
        }

        // open socket
        this->sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (this->sock_fd < 0){
            clsErr("error while opening socket.\n");
            perror("socket error");
            return false;
        }

        // set source port
        if(srcport != 0) {
            int val = 1;
            int r = setsockopt(this->sock_fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
            if (r < 0){
                clsErr("error while allowing multiple sockets.\n");
                perror("UDP error");
                return false;
            }

            std::cout<<"BIND ON SRCPORT: "<<srcport<<"\n";
            struct sockaddr_in srcaddr;
            memset(&srcaddr, 0, sizeof(srcaddr));
            srcaddr.sin_family = AF_INET;
            srcaddr.sin_addr.s_addr = htonl(INADDR_ANY);
            srcaddr.sin_port = htons(srcport);
            if (bind(sock_fd, (struct sockaddr *) &srcaddr, sizeof(srcaddr)) < 0) {
                clsErr("error while setting socket.\n");
                perror("bind");
            }
        }

        // allow multiple sockets
        if (!ip.empty()) {
            if (isMulticast(ip)) {
                this->isMulti = true;
                unsigned char ttl = 1;
                int r = setsockopt(this->sock_fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
                if (r < 0){
                    clsErr("error while allowing multiple sockets.\n");
                    return false;
                }
                clsSuc("multicast socket created.\n");

            } if(isBroadcast(ip)) {
                int broadcast= 1;
                int r = setsockopt(this->sock_fd, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
                if (r < 0){
                    clsErr("error while setting broadcast\n");
                    return false;
                }
                clsSuc("broadcast socket created.\n");

            } else{
                this->isMulti = false;
                clsSuc("classic socket created.\n");
            }
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

        if(this->isMulti && reciver){
            setsockopt(this->sock_fd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &this->imr, sizeof(this->imr));
        }
        return ::close(this->sock_fd);
    }

    bool
    UDPSocket::isMulticast (const std::string ip) {
        return (ntohl(inet_addr(ip.c_str())) & 0xF0000000) == 0xE0000000;
    }

    bool
    UDPSocket::isBroadcast (const std::string ip) {
        return ip.compare(ip.size()-3, 3, "255") == 0;
    }
}}