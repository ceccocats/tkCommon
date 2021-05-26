#include "tkCommon/communication/ethernet/PCAPHandler.h"

tk::communication::PCAPHandler::PCAPHandler() = default;
tk::communication::PCAPHandler::~PCAPHandler() = default;


bool tk::communication::PCAPHandler::initReplay(const std::string fileName, const std::string filter){

    replayMode = true;
    char errbuff[PCAP_ERRBUF_SIZE];
    struct bpf_program fp;

    tkMSG("opening: " + fileName + "\n");
    pcapFile = pcap_open_offline(fileName.c_str(), errbuff);

    if (!pcapFile){
        tkERR(std::string{"error from pcap_open_live(): "}+std::string{errbuff}+"\n");
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
            tkERR(std::string{"couldn't parse filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            tkERR(std::string{"couldn't install filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }
    }

    return true;
}

bool tk::communication::PCAPHandler::initRecord(const std::string fileName, const std::string iface, const std::string filter){

    replayMode = false;
    bpf_u_int32 net;
    struct bpf_program fp;
    char errbuff[PCAP_ERRBUF_SIZE];


    if ((pcapFile = pcap_open_live(iface.c_str(), BUFSIZ, 0, 1000, errbuff)) == NULL) {
        tkERR(std::string{"error from pcap_open_live(): "}+std::string{errbuff}+"\n");
        return false;
    }

    if ((pcapDumper = pcap_dump_open(pcapFile, fileName.c_str())) == NULL) {
        tkERR(std::string{"error from pcap_dump_open():: "}+pcap_geterr(pcapFile)+"\n");
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
            tkERR(std::string{"couldn't parse filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            tkERR(std::string{"couldn't install filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }
    }

    return true;
}

void tk::communication::PCAPHandler::record(){

    if(!replayMode){

        pcap_loop(pcapFile, 0, pcap_dump, (u_char *)pcapDumper);
    }
}

void tk::communication::PCAPHandler::recordStat(struct pcap_stat& stat){

    if(!replayMode){
        pcap_stats(pcapFile, &stat);
    }
}

int tk::communication::PCAPHandler::getPacket(uint8_t* buffer, timeStamp_t& stamp){

    if(!replayMode){
        tkERR("you are in recorder mode.\n");
        return -1;
    }

    if (!pcapFile){
        tkERR("pcap error.\n");
        return -1;
    }

    int len = parser.computeNextPacket(pcapFile,buffer,stamp);

    /*if(len < 0){
        tkWRN("Reading error.\n");
        return -1;
    }*/

    return len;
}

bool tk::communication::PCAPHandler::close(){

    if(!replayMode){

        pcap_breakloop(pcapFile);
        pcap_dump_close(pcapDumper);
    }

    pcap_close(pcapFile);
    return true;

}