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

        bool ok = pcap.initReplay(fileName);
        if(ok)
            clsSuc("opened file: " + fileName + "\n")
        else
            clsErr("fail open file: " + fileName + "\n")
        return ok;
    }

    int
    CanInterface::read(tk::data::CanData_t *data) {

        if(offlineMode){
            data->frame.can_dlc = pcap.getPacket(data->frame.data, data->stamp);

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
        pcap.initRecord(fileName + "/" + iface + ".pcap", iface);
        //system(std::string(std::string("candump -L ") + iface + " > " + fileName + "/" + iface + ".log &").c_str());
    }

    bool
    CanInterface::close(){

        if(offlineMode){
            pcap.close();
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