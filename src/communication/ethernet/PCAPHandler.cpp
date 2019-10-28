#include "tkCommon/communication/ethernet/PCAPHandler.h"

tk::communication::PCAPHandler::PCAPHandler() = default;
tk::communication::PCAPHandler::~PCAPHandler() = default;


bool tk::communication::PCAPHandler::initReplay(const std::string fileName, const std::string filter){

    replayMode = true;
    char errbuff[PCAP_ERRBUF_SIZE];
    struct bpf_program fp;

    pcapFile = pcap_open_offline(fileName.c_str(), errbuff);
    if (!pcapFile){
        fprintf(stderr, "PCAPHandler: error from pcap_open_live(): %s\n", errbuff);
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {

            fprintf(stderr, "PCAPHandler: couldn't parse filter %s: %s\n", filter.c_str(), pcap_geterr(pcapFile));
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            fprintf(stderr, "PCAPHandler: couldn't install filter %s: %s\n", filter.c_str(), pcap_geterr(pcapFile));
            return false;
        }
    }

    return true;
}

bool tk::communication::PCAPHandler::initRecord(const std::string fileName, const std::string iface, const std::string filter){

    replayMode = false;
    bpf_u_int32 net;
    struct bpf_program fp;
    char errbuf[PCAP_ERRBUF_SIZE];


    if ((pcapFile = pcap_open_live(iface.c_str(), BUFSIZ, 0, 1000, errbuf)) == NULL) {
        fprintf(stderr, "PCAPHandler: error from pcap_open_live(): %s\n", errbuf);
        return false;
    }

    if ((pcapDumper = pcap_dump_open(pcapFile, fileName.c_str())) == NULL) {
        fprintf(stderr, "PCAPHandler: error from pcap_dump_open(): %s\n", pcap_geterr(pcapFile));
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {

            fprintf(stderr, "PCAPHandler: couldn't parse filter %s: %s\n", filter.c_str(), pcap_geterr(pcapFile));
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            fprintf(stderr, "PCAPHandler: couldn't install filter %s: %s\n", filter.c_str(), pcap_geterr(pcapFile));
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

int tk::communication::PCAPHandler::getPacket(uint8_t& buffer, timeStamp_t& stamp){

    if(!replayMode){
        fprintf(stderr, "PCAPHandler: you are in recorder mode.");
        return -1;
    }

    if (!pcapFile){
        fprintf(stderr, "PCAPHandler: error pcap.");
        return -1;
    }

    struct pcap_pkthdr *header;
    int returnValue = pcap_next_ex(this->pcapFile, &header, (const u_char**)&buffer);

    if (returnValue < 0){
        fprintf(stderr, "PCAPHandler: end of packet file.");
        return -1;
    }

    stamp = header->ts.tv_usec;
    return header->len;
}

bool tk::communication::PCAPHandler::close(){

    if(!replayMode){

        pcap_breakloop(pcapFile);
        pcap_dump_close(pcapDumper);
    }

    pcap_close(pcapFile);
    return true;

}