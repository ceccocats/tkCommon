#include "tkCommon/communication/serial/SerialPort.h"
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <unistd.h>
#include <string.h>

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
