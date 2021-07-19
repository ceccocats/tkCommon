#pragma once

#include "tkCommon/communication/ethernet/PCAPHandler.h"
#include "tkCommon/communication/ethernet/UDPSocket.h"
#include "tkCommon/communication/ethernet/TCPSocket.h"

namespace tk { namespace communication {

    static const int BUFFER_LENGTH = 30000;

    class Ethinterface {
    public:
        Ethinterface();
        ~Ethinterface();

        //Init
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that create a recive socket
         *
         * @param port      UDP port
         * @param ip        IP [optional]
         * @return          Success
         */
        bool initUDP(const int port, const std::string ip = "", time_t timeout_us = -1);

        /**
         * Method that create a recive socket
         *
         * @param port      TCP port
         * @param ip        IP
         * @return          Success
         */
        bool initTCP(const int port, const std::string ip, bool timeout = false);

        /**
         * Method that init pcap ethernet
         * 
         * @param fileName  Saving file name
         * @param filter    Filter on recorder, default empty
         * @return          Success
         */
        bool initPcapReplay(const std::string fileName, const std::string filter="");




        //Reading
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that return a packet
         * @param buffer    Packet data
         * @param stamp     Packet timestamp
         * @return          Packet lenght
         */
        int read(uint8_t* buffer, timeStamp_t& stamp);



        //Writing
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that send a packet (only TCP)
         * @param buffer    Packet data
         * @param length    Packet lenght
         * @return          status
         */
        bool send(uint8_t* buffer, int length);





        //Recording
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that record the ethernet messages (blocking)
         * 
         * @param fileName  Saving file name
         * @param filter    Filter on recorder, default empty
         */
        void record(const std::string fileName, const std::string iface, const std::string filter);

        /**
         * Method that return the recording statistics
         * 
         * @return          String with stat
         */
        std::string recordStat();

         /**
         * Method that return the recive packets
         * 
         * @return          String with stat
         */
        u_int  ifaceRecive();

         /**
         * Method that return the drop packets
         * 
         * @return          String with stat
         */
        u_int  ifaceDrop();




        //Closing
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that stop the recorder or close the replay file or socket
         * @return          Success
         */
        bool close();



    private:
        bool        replayMode, isUdp;

        PCAPHandler pcap;
        UDPSocket   udpSocket;
        TCPSocket   tcpSocket;

        bool        recording = false;
        pcap_stat   stat;
    };
}}