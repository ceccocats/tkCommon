#include "tkCommon/communication/serial/SerialPort.h"

namespace tk { namespace communication {
    bool 
    SerialPort::init(const std::string& port)
    {
        // open port
        serialPort.Open(port);
        if(!serialPort.IsOpen())
            return false;

        // set baud
        serialPort.SetBaudRate(LibSerial::BaudRate::BAUD_115200);

        // Set the number of data bits.
        serialPort.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8) ;

        // Turn off hardware flow control.
        serialPort.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE) ;

        // Disable parity.
        serialPort.SetParity(LibSerial::Parity::PARITY_NONE) ;
        
        // Set the number of stop bits.
        serialPort.SetStopBits(LibSerial::StopBits::STOP_BITS_1) ;

        // Wait for data to be available at the serial port.
        while(!serialPort.IsDataAvailable()) 
        {
            usleep(1000) ;
        }

        return true;
    }

    bool 
    SerialPort::close()
    {
        serialPort.Close();

        return true;
    }

    bool
    SerialPort::readLine(std::string& msg, char terminator, timeStamp_t timeout)
    {
        try {
            serialPort.ReadLine(msg, terminator, timeout);
        } catch (const LibSerial::ReadTimeout&) {
            tkWRN("Timeout.\n");
            return false;
        }

        return true;
    }

    bool
    SerialPort::read(std::string& msg, int size, timeStamp_t timeout)
    {
        try {
            serialPort.Read(msg, size, timeout);
        } catch (const LibSerial::ReadTimeout&) {
            tkWRN("Timeout.\n");
            return false;
        }

        return true;
    }
}}

/*
bool tk::sensors::SerialPort::init(std::string port) {

    this->port = port;

    soc = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);

    bool tty_status = soc >= 0;

    if(!tty_status)
        std::cout<<"can't open Serial port: "<<port<<"\n";

    return tty_status;
}

bool tk::sensors::SerialPort::closeSoc() {
    int err = close(soc);
    if(err == -1) {
        printf("Serial close ERROR: %s\n", strerror(errno));
        return false;
    } else {
        printf("Serial closed\n");
        return true;
    }
}


bool tk::sensors::SerialPort::readSoc(std::string &msg) {

    // get line from device
    char line_buf[1024];
    char c = 0;
    int i;
    for(i=0; i<1024-1 && c != '\n'; i++) {    
        int rd = read(soc, &c, 1);
        if(rd < 0) {
            std::cout<<"error read serial\n";
            return false;
        }
        //std::cout<<c<<" "<<i<<"\n";
        line_buf[i] = c; 
    }
    line_buf[i] = '\0'; 
    //std::cout<<"LINE: "<<line_buf<<"  END\n";

    msg = std::string(line_buf);
    return true;
}
*/