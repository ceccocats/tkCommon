#pragma once

#include <tkCommon/communication/ethernet/PCAPHandler.h>
#include "tkCommon/common.h"
#include "tkCommon/data/CanData.h"

namespace tk { namespace communication {
    class CanInterface {
    public:
        CanInterface();
        ~CanInterface();

        //Init
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that create a recive socket
         *
         * @param port      SocketCan port
         * @return          Success
         */
        bool initSocket(const std::string port = "can0");

        /**
         * Method that init read from log file
         *
         * @param fileName  Saving file name
         * @param filter    Filter on recorder, default empty
         * @return          Success
         */
        bool initFile(const std::string fileName);


        //Reading
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that return a packet
         * @param can_frame Packet data
         * @return          Packet lenght
         */
        int read(tk::data::CanData_t *data);

        /**
         * Write frem on socket, works only in online mode
         * @param frame
         * @return
         */
        bool write (tk::data::CanData_t *data);

        //Recording
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that record the ethernet messages (blocking)
         *
         * @param fileName  Saving file name
         * @param interface SocketCan port to record
         */
        bool startRecord(const std::string fileName, const std::string iface);


        //Closing
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * Method that stop the recorder or close the replay file or socket
         * @return          Success
         */
        bool close();


    private:
        bool    offlineMode;
        int     soc = -1;

        bool                            pcapmode;
        tk::communication::PCAPHandler  pcap;
        std::ifstream                   logstream;

        //Record static thread
        static void* record (void* object){
            CanInterface* obj = reinterpret_cast<CanInterface*>(object);
            obj->pcap.record();
        }
    };
}}