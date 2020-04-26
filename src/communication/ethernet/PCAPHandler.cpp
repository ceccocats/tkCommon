#include "tkCommon/communication/ethernet/PCAPHandler.h"
#include "tkCommon/communication/ethernet/packet_parser.h"

tk::communication::PCAPHandler::PCAPHandler() = default;
tk::communication::PCAPHandler::~PCAPHandler() = default;


bool tk::communication::PCAPHandler::initReplay(const std::string fileName, const std::string filter){

    replayMode = true;
    char errbuff[PCAP_ERRBUF_SIZE];
    struct bpf_program fp;

    clsMsg("opening: " + fileName + "\n");
    pcapFile = pcap_open_offline(fileName.c_str(), errbuff);

    if (!pcapFile){
        clsErr(std::string{"error from pcap_open_live(): "}+std::string{errbuff}+"\n");
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
            clsErr(std::string{"couldn't parse filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            clsErr(std::string{"couldn't install filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
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
        clsErr(std::string{"error from pcap_open_live(): "}+std::string{errbuff}+"\n");
        return false;
    }

    if ((pcapDumper = pcap_dump_open(pcapFile, fileName.c_str())) == NULL) {
        clsErr(std::string{"error from pcap_dump_open():: "}+pcap_geterr(pcapFile)+"\n");
        return false;
    }

    if(filter != "") {

        if (pcap_compile(pcapFile, &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
            clsErr(std::string{"couldn't parse filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
            return false;
        }

        if (pcap_setfilter(pcapFile, &fp) == -1) {
            clsErr(std::string{"couldn't install filter "}+filter+":"+pcap_geterr(pcapFile)+"\n");
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
        clsErr("you are in recorder mode.\n");
        return -1;
    }

    if (!pcapFile){
        clsErr("pcap error.\n");
        return -1;
    }

    /*struct pcap_pkthdr *header;
    const u_char *pkt_data;
    int returnValue = pcap_next_ex(this->pcapFile, &header, &pkt_data);

    struct sniff_ip* ip_header = (struct sniff_ip*)(pkt_data + SIZE_ETHERNET);

    std::cout<<"-----"<<std::endl;
    
    std::cout<<"******"<<std::endl;


    if (returnValue < 0){
        clsWrn("end of packet file.\n");
        return -1;
    }

    int HEADER_LEN = 0;
    PCAP_header pcap_header;
    if(parse_UDP_packet(pkt_data, header->caplen, pcap_header)) {
        //dump_PCAP_header(pcap_header);
        HEADER_LEN = pcap_header.headerLen;
    }
    std::memcpy(buffer,pkt_data + HEADER_LEN,header->len - HEADER_LEN);
    //stamp = header->ts.tv_sec;
    stamp = header->ts.tv_sec * 1e6 + header->ts.tv_usec;*/

    int n = skipHeaderAndUnfrag(pcapFile,buffer,stamp);

    //std::cout<<"stamp:"<<stamp<<std::endl;

    return n;
}

bool tk::communication::PCAPHandler::close(){

    if(!replayMode){

        pcap_breakloop(pcapFile);
        pcap_dump_close(pcapDumper);
    }

    pcap_close(pcapFile);
    return true;

}