#pragma once

#include "tkCommon/communication/ethernet/PCAPHandler.h"
#include "tkCommon/communication/ethernet/UDPSocket.h"

namespace tk { namespace communication {
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
        bool initUDP(const int port, const std::string ip = "");

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




        //Closing
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that stop the recorder or close the replay file or socket
         * @return          Success
         */
        bool close();



    private:
        bool            replayMode;
        PCAPHandler     pcap;
        UDPSocket       socket;

        pcap_stat       stat;
    };
}}