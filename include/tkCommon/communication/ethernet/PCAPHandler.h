#pragma once
#include <pcap.h>

#include "tkCommon/common.h"
#include "tkCommon/utils.h"
#include "tkCommon/terminalFormat.h"
#include "tkCommon/communication/ethernet/PacketParser.h"

namespace tk { namespace communication {

    class PCAPHandler {
    public:
        PCAPHandler();
        ~PCAPHandler();

        //Replay
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that init the pcap replay from a file
         * @param fileName  File you want to read
         * @param filter    Filter on a file, default empty
         * @return          Success
         */
        bool initReplay(const std::string fileName, const std::string filter="");
        /**
         * Method that return a packet
         * @param buffer    Packet data
         * @param stamp     Packet timestamp
         * @param HEADER_LEN header to skip (42 for UDP)
         * @return          Packet lenght
         */
        int getPacket(uint8_t* buffer, timeStamp_t& stamp);

        //Recording
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that init pcap ethernet record
         * @param fileName  Saving file name
         * @param filter    Filter on recorder, default empty
         * @return          Success
         */
        bool initRecord(const std::string fileName, const std::string iface, const std::string filter="");
        /**
         * Method that record the ethernet messages (blocking)
         */
        void record();
        /**
         * Method that return the recording statistics
         * @param stat      Statistic pcap struct
         */
        void recordStat(struct pcap_stat& stat);

        //Closing
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that stop the recorder or close the replay file
         * @return          Success
         */
        bool close();
    private:
        bool                replayMode;
        pcap_dumper_t       *pcapDumper;
        pcap_t      	    *pcapFile;
        PacketParser        parser;

    };
}}