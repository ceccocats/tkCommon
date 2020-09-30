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

        bool ok;
        if(tk::common::endsWith(fileName,"pcap")) {
            ok = pcap.initReplay(fileName);
            pcapmode = true;
        } else {
            logstream.open(fileName);
            ok = logstream.is_open();
            pcapmode = false;
        }

        if(ok)
            clsSuc("opened file: " + fileName + "\n")
        else
            clsErr("fail open file: " + fileName + "\n")
        return ok;
    }

    int
    CanInterface::read(tk::data::CanData_t *data) {

        if(offlineMode){

            if(pcapmode) {
                uint8_t buffer[64];
                uint8_t *buf = buffer;
                int len = pcap.getPacket(buffer, data->stamp); // we get the header to read ID and DLC

                // FIXME: this is linux cooked capture header skip (luca shitty record)
                bool cooked = false;
                if(len > 16) {
                    buf = buf + 16;
                    len -= 16;
                    cooked = true;
                }
                tkASSERT(len <= 16);
                
                if (len > 8) { // header is 8 byte
                    uint16_t id;
                    if(!cooked) {
                        memcpy(&id, buf + 2, sizeof(uint16_t));
                        id = __bswap_16(id);
                    } else {
                        memcpy(&id, buf, sizeof(uint16_t));
                    }
                    data->frame.can_id = id;
                    memcpy(&data->frame.can_dlc, buf + 4, sizeof(uint8_t));

                    // dimensions doesn match
                    if (8 + data->frame.can_dlc > len)
                        return false;

                    memcpy(&data->frame.data, buf + 8, data->frame.can_dlc * sizeof(uint8_t));
                    //tk::common::hex_dump(std::cout, buf, len);
                    //std::cout << std::hex << data->frame.can_id << std::dec << " " << int(data->frame.can_dlc) << " "
                    //          << data->stamp << "\n";
                    return true;
                } else {
                    return false;
                }
            } else {
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
                    // get msg ID
                    if(i==0) {
                        std::getline(iss, s, '#');
                        data->frame.can_id = std::stoul(s, nullptr, 16);
                        //std::cout<<std::hex<<data.frame.can_id<<" ";
                    }

                    // parse data
                    if(i==1) {
                        std::getline(iss, s, ' ');

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
            }
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

    bool
    CanInterface::startRecord(const std::string fileName, const std::string iface){
        pcap.initRecord(fileName + "/" + iface + ".pcap", iface);
        pthread_t t1;
        int val = pthread_create(&t1, NULL, CanInterface::record, this);

        if(val == -1){
            clsErr("error creating recorder thread.\n");
            return false;
        }else{
            return true;
        }
        //system(std::string(std::string("candump -L ") + iface + " > " + fileName + "/" + iface + ".log &").c_str());
    }

    bool
    CanInterface::close(){

        if(offlineMode){
            if(pcapmode)
                pcap.close();
            else 
                logstream.close();
            return true;
        }else{
            int err = ::close(soc);
            if(err == -1) {
                clsErr(std::string("CAN close ERROR: ") + strerror(errno) + "\n");
                return false;
            } else {
                clsMsg("CAN closed\n");
                return true;
            }
        }
    }


}}