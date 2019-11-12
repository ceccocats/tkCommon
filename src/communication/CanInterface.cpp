#include "tkCommon/communication/CanInterface.h"
#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>

namespace tk { namespace communication {

    CanInterface::CanInterface() = default;
    CanInterface::~CanInterface() = default;


    bool
    CanInterface::initSocket(const std::string port){
        offlineMode = false;

        struct ifreq ifr;
        struct sockaddr_can addr;

        // open socket
        soc = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if(soc < 0) {
            clsErr("Error opening socket: " + port + "\n");
            return false;
        }

        addr.can_family = AF_CAN;
        strcpy(ifr.ifr_name, port.c_str());

        if (ioctl(soc, SIOCGIFINDEX, &ifr) < 0) {
            clsErr("Error setting socket: " + port + "\n");
            return false;
        }

        addr.can_ifindex = ifr.ifr_ifindex;

        //if(fcntl(soc, F_SETFL, O_NONBLOCK) < 0) {
        //    return false;
        //}

        if (bind(soc, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            clsErr("Error binding socket: " + port + "\n");
            return false;
        }
        return true;
    }

    bool
    CanInterface::initFile(const std::string fileName){
        offlineMode = true;

        logstream.open(fileName, std::ifstream::in);
        if(logstream)
            clsSuc("opened file: " + fileName + "\n")
        else
            clsErr("fail open file: " + fileName + "\n")
        return logstream.is_open();
    }

    int
    CanInterface::read(tk::data::CanData_t *data) {

        if(offlineMode){
            if(!logstream.is_open())
                return false;

            std::string line;
            if(!std::getline(logstream, line))
                return false;

            std::istringstream iss(line);

            std::string stamp;
            // get timestamp string
            iss >> stamp;
            // remove brakets
            stamp = stamp.substr(1, stamp.size()-2);
            // remove point
            stamp.erase(std::remove(stamp.begin(), stamp.end(), '.'), stamp.end());

            // fill timestamp value
            data->stamp = std::stoull(stamp);

            // trash interface name
            std::string name;
            iss >> name;

            // get data string
            std::string s;
            for(int i =0; i<2; i++) {
                std::getline(iss, s, '#');

                // get msg ID
                if(i==0) {
                    data->frame.can_id = std::stoul(s, nullptr, 16);
                    //std::cout<<std::hex<<data.frame.can_id<<" ";
                }

                // parse data
                if(i==1) {
                    data->frame.can_dlc = s.size()/2;
                    if(data->frame.can_dlc > CAN_MAX_DLEN) {
                        return false;
                    }

                    std::string val = "00";
                    for(int k=0; k<data->frame.can_dlc; k++) {
                        val[0] = s[k*2];
                        val[1] = s[k*2+1];
                        data->frame.data[k] = std::stoul(val, nullptr, 16);
                        //std::cout<<val;
                    }
                    //std::cout<<"\n";
                }
            }
            return true;

        }else{
            int retval = 0;

            retval = ::read(soc, &data->frame, sizeof(struct can_frame));
            bool ok = retval == sizeof(struct can_frame);
            if(ok) {
                struct timeval tv;
                ioctl(soc, SIOCGSTAMP, &tv);
                data->stamp = tv2TimeStamp(tv);
            }
            return ok;
        }
    }

    bool
    CanInterface::write (tk::data::CanData_t *data) {
        if(offlineMode)
            return false;

        int retval;
        retval = ::write(soc, &data->frame, sizeof(struct can_frame));

        if(retval == -1) {
            printf("CAN write ERROR: %s\n", strerror(errno));
        }
        return retval == sizeof(struct can_frame);
    }

    void
    CanInterface::record(const std::string fileName, const std::string iface){

    }

    bool
    CanInterface::close(){

        if(offlineMode){
            logstream.close();
            return true;
        }else{
            int err = ::close(soc);
            if(err == -1) {
                printf("CAN close ERROR: %s\n", strerror(errno));
                return false;
            } else {
                printf("CAN closed\n");
                return true;
            }
        }
    }


}}