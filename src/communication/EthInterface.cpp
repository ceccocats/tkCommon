#include "tkCommon/communication/EthInterface.h"

namespace tk { namespace communication {

    bool
    Ethinterface::initUDP(const int port, const std::string ip = ""){
        
        this->replayMode    = false;
        return this->socket.initReceiver(port,ip);
    }

    bool
    Ethinterface::initPcap(const std::string fileName, const std::string filter=""){

        this->replayMode    = true;
        this->fileName      = fileName;
        this->filter        = filter;
        return this->pcap.initReplay(fileName,filter);
    }

    int
    Ethinterface::read(u_int8_t& buffer, timeStamp_t& stamp){

        if(this->replayMode == true){

            //Pcap
            return this->pcap.getPacket(buffer,stamp);
        }else{

            //Socket
            int len;
            this->socket.receive(buffer,len);
            stamp           = getTimeStamp();
            return len;

        }
    }

    void
    Ethinterface::record(){

        pcap.initRecord(this->fileName,this->filter);
        pcap.record();
    }

    std::string 
    Ethinterface::recordStat(){

        pcap.recordStat(stat);
        return "##Recive "+stat.ps_recv+"##Drop: "+stat.ps_drop+"##IfaceDrop: "+stat.ps_ifdrop;
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