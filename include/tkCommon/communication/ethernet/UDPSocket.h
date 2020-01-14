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

    class UDPSocket{
    public:
         UDPSocket();
        ~UDPSocket();

        /**
         * Method that create a socket as receiver
         *
         * @param port          UDP port
         * @param ip            IP [optional]
         * @return
         */
        bool initReceiver(const int port, const std::string ip = "");

        /**
         * Method that create a socket as sender
         * @param port          UDP port
         * @param ip            IP
         * @return
         */
        bool initSender(const int port, const std::string ip, const int srcport = 0);

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

        /**
         * Method that return if an ip is multicast
         *
         * @param ip        IP
         * @return          true if ip is multicast
         */
        static bool isMulticast(const std::string ip);

        /**
         * Method that return if an ip is broadcast
         *
         * @param ip        IP
         * @return          true if ip is multicast
         */
        static bool isBroadcast(const std::string ip);

    private:
        int             sock_fd;
        bool            reciver;
        sockaddr_in     sock_addr;
        ip_mreq         imr;

        bool            isMulti;
    };

}}
