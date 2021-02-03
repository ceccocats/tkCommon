#pragma once

#include <libserial/SerialPort.h>
#include "tkCommon/time.h"
#include "tkCommon/log.h"


namespace tk{ namespace communication{

    class SerialPort {
    private:
        LibSerial::SerialPort serialPort;

    public:
        SerialPort() {};
        ~SerialPort() {};

        /**
         * @brief 
         * 
         * @param port 
         * @return true 
         * @return false 
         */
        bool init(const std::string& port = "/dev/ttyACM0");

        /**
         * @brief 
         * 
         * @param msg 
         * @param terminator 
         * @param timeout 
         * @return true 
         * @return false 
         */
        bool readLine(std::string &msg, char terminator = '\n', timeStamp_t timeout = 250);

        /**
         * @brief 
         * 
         * @param msg 
         * @param size 
         * @param timeout 
         * @return true 
         * @return false 
         */
        bool read(std::string& msg, int size = 1024, timeStamp_t timeout = 250);
        
        /**
         * Close the socket
         *
         * @return false if fail
         */    
        bool close();

        /**
         * Status of the socket
         *
         * @return true if open and active
         */  
        bool status() { return serialPort.IsOpen(); }
    };
    
}}
