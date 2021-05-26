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
        bool init(const std::string& port = "/dev/ttyACM0", int baud = 9600);

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

        bool readByte(uint8_t& byte, timeStamp_t timeout = 250);
        
        /**
         * @brief 
         * 
         * @param msg 
         * @return true 
         * @return false 
         */
        bool write(std::string &msg);

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
        bool isOpen() { return serialPort.IsOpen(); }

        bool isDataAvailable() { return serialPort.IsDataAvailable(); }
    };
    
}}
