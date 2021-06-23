#include "tkCommon/communication/serial/SerialPort.h"

using namespace tk::communication;

#ifdef SERIAL_ENABLED
bool 
SerialPort::init(const std::string& port, int baud)
{
    // open port
    try {
        serialPort.Open(port);
    } catch (LibSerial::OpenFailed e) {
        tkERR("Bad file descriptor.\n");
        return false;
    }
    //if(!serialPort.IsOpen())
    //    return false;

    // set baud
    switch (baud)
    {
    case 9600:
        serialPort.SetBaudRate(LibSerial::BaudRate::BAUD_9600);
        break;
    case 115200:
        serialPort.SetBaudRate(LibSerial::BaudRate::BAUD_115200);
        break;
    case 921600:
        serialPort.SetBaudRate(LibSerial::BaudRate::BAUD_921600);
        break;
    default:
        return false;
        break;
    }
    

    // Set the number of data bits.
    serialPort.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8) ;

    // Turn off hardware flow control.
    serialPort.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE) ;

    // Disable parity.
    serialPort.SetParity(LibSerial::Parity::PARITY_NONE) ;
    
    // Set the number of stop bits.
    serialPort.SetStopBits(LibSerial::StopBits::STOP_BITS_1) ;

    // Wait for data to be available at the serial port.
    //while(!serialPort.IsDataAvailable()) 
    //{
    //    usleep(1000) ;
    //}

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

bool 
SerialPort::readByte(uint8_t& byte, timeStamp_t timeout)
{
    try {
        serialPort.ReadByte(byte, timeout);
    } catch (const LibSerial::ReadTimeout&) {
        tkWRN("Timeout.\n");
        return false;
    }
    return true;
}

bool
SerialPort::write(std::string &msg)
{
    serialPort.Write(msg);
    serialPort.DrainWriteBuffer();

    return true;
}

bool 
SerialPort::isOpen() 
{ 
    return serialPort.IsOpen(); 
}

bool 
SerialPort::isDataAvailable() { 
    return serialPort.IsDataAvailable(); 
}
#else
bool 
SerialPort::init(const std::string& port, int baud)
{
    soc = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);

    bool tty_status = soc >= 0;

    if(!tty_status) {
        tkERR("Can't open Serial port: "<<port);
        return false;
    }

    return true;
}

bool 
SerialPort::close()
{
    int err = ::close(soc);
    if(err == -1) {
        tkERR(strerror(errno));
        return false;
    } 
    return true;
}

bool
SerialPort::readLine(std::string& msg, char terminator, timeStamp_t timeout)
{
    msg.clear();
    char c = 0;
    timeStamp_t initial_time = getTimeStamp();
    timeStamp_t dt = 0;
    for(int i = 0; c != terminator && dt < timeout ; i++) {    
        int rd = ::read(soc, &c, 1);
        if(rd < 0) {
            tkERR("Error while reading.");
            return false;
        }
        msg.append(std::string(c)); 
        dt = getTimeStamp() - initial_time;
    }
    //line_buf[i] = '\0'; 

    return true;
}

bool
SerialPort::read(std::string& msg, int size, timeStamp_t timeout)
{
    tkWRN("Not implemented for unix socket, please install libserial v1.0.0");
    return false;
}

bool 
SerialPort::readByte(uint8_t& byte, timeStamp_t timeout)
{
    tkWRN("Not implemented for unix socket, please install libserial v1.0.0");
    return false;
}

bool
SerialPort::write(std::string &msg)
{
    tkWRN("Not implemented for unix socket, please install libserial v1.0.0");
    return false;
}

bool 
SerialPort::isOpen() 
{ 
    tkWRN("Not implemented for unix socket, please install libserial v1.0.0");
    return false;
}

bool 
SerialPort::isDataAvailable() { 
    tkWRN("Not implemented for unix socket, please install libserial v1.0.0");
    return false;
}
#endif
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