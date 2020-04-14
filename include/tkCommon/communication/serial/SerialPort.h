#pragma once
#include "tkCommon/utils.h"


namespace tk{ namespace sensors{

    /**
     * Class for interfacing the Serial port
     * 
     * @author Francesco Gatti
     * 
     * @version 1.0
     * 
     */
    class SerialPort {

    private:
        std::string port;
        int soc = -1;

    public:
        SerialPort() {};
        ~SerialPort() {};

        /**
         * Initialize object on the desidered CAN interface
         * 
         * @param port string of the interface to open 
         * @return false if fail
         */
        bool init(std::string port = "/dev/ttyACM0");

        
        /**
         * Read message on socket
         *
         * @param msg string msg to be filled
         * @return false if fail
         */
        bool readSoc(std::string &msg);
        
        /**
         * Close the socket
         *
         * @return false if fail
         */    
        bool closeSoc();

        /**
         * Status of the socket
         *
         * @return true if open and active
         */  
        bool status() { return soc >= 0; }
    };
    
}}
