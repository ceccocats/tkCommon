#include "tkCommon/communication/EthInterface.h"

namespace tk { namespace communication {

    Ethinterface::Ethinterface() = default;
    Ethinterface::~Ethinterface() = default;


    bool
    Ethinterface::initUDP(const int port, const std::string ip){
        
        this->replayMode    = false;
        return this->socket.initReceiver(port,ip);
    }

    bool
    Ethinterface::initPcapReplay(const std::string fileName, const std::string filter){

        this->replayMode    = true;
        return this->pcap.initReplay(fileName,filter);
    }

    int
    Ethinterface::read(uint8_t* buffer, timeStamp_t& stamp){

        if(this->replayMode == true){

            //Pcap
            return this->pcap.getPacket(buffer,stamp);
        }else{

            //Socket
            int len = this->socket.receive(buffer,20000);
            stamp   = getTimeStamp();

            tkASSERT(len != 20000)
            return len;

        }
    }

    void
    Ethinterface::record(const std::string fileName, const std::string iface, const std::string filter){

        this->replayMode = true;
        pcap.initRecord(fileName, iface, filter);
        pcap.record();
    }

    std::string 
    Ethinterface::recordStat(){

        pcap.recordStat(stat);
        return std::string("##Recive ")+std::to_string(stat.ps_recv)+"##Drop: "+std::to_string(stat.ps_drop)+"##IfaceDrop: "+std::to_string(stat.ps_ifdrop);
    }

    bool
    Ethinterface::close(){

        if(this->replayMode == true){

            //Pcap
            return this->pcap.close();
        }else{

            //Socket
            return this->socket.close();
        }
    }


}}