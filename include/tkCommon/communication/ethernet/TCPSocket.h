#pragma once

#include <iostream>
#include <cstring>

//socket library
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include <tkCommon/terminalFormat.h>

namespace tk{ namespace communication {

    class TCPSocket{
    public:

        /**
         * Method that create a TCP server socket
         *
         * @param port          UDP port
         * @return
         */
        bool initServer(const int port);

        /**
         * Method that create a TCP client socket
         *
         * @param port          UDP port
         * @param ip            IP
         * @return
         */
        bool initClient(const int port, const std::string ip);

        /**
         * Method that read a packet
         *
         * @param buffer    Data packet (return)
         * @param length    Length of packet that we want to read
         * @return          Number of byte received
         */
        int receive(uint8_t* buffer, int length);

        /**
         * Method that send a packet with specific length
         *
         * @param buffer    Data packet
         * @param length    Length of packet that we want to send
         * @return
         */
        bool send(uint8_t* buffer, int length);

        /**
         * Method that close the socket
         *
         */
        bool close();

    private:
        int             sock_fd = -1;    
        int             sock_client = -1;
        sockaddr_in     sock_addr;
        ip_mreq         imr;
    };

}}
