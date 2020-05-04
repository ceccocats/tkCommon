#include "tkCommon/communication/ethernet/PacketParser.h"

int 
tk::communication::PacketParser::firstFrame(pcap_t* pcapFile, timeStamp_t& stamp){
                
    int status = pcap_next_ex(pcapFile, &header, &pkt_data);
    if(status < 0) return -1;

    //set timestamp
    stamp = header->ts.tv_sec * 1e6 + header->ts.tv_usec;

    //get ethernet header
    if(header->caplen < sizeof(sniff_ethernet)) return -1;
    ethernet = (struct sniff_ethernet*)(pkt_data);

    //get ip header
    if(header->caplen < sizeof(sniff_ip) + SIZE_ETHERNET) return -1;
    ip = (struct sniff_ip*)(pkt_data + SIZE_ETHERNET);
    size_ip = IP_HL(ip)*4;
    if (size_ip < 20) return -1;

    // if is fragmented ?
    if(ip->ip_off != FRAGMENT_MASK){

        if(ip->ip_off == 64){
            return header->len - size_ip - SIZE_ETHERNET;
        }

        if(ip->ip_off == 0){
            return header->len - size_ip - SIZE_ETHERNET;
        }

        if(detectFragment == false){
            clsMsg("IP Fragment detected, auto defragment active\n");
        }
        detectFragment = true;
        return firstFrame(pcapFile,stamp);
    }else{

        return header->len - size_ip - SIZE_ETHERNET;
    }
}

int
tk::communication::PacketParser::defragment(pcap_t* pcapFile,uint8_t* buffer, timeStamp_t& stamp, int len_buffer, int total_lenght){

    do{

        int status = pcap_next_ex(pcapFile, &header, &pkt_data);
        if(status < 0) return -1;

        //set timestamp
        stamp = header->ts.tv_sec * 1e6 + header->ts.tv_usec;

        //get ethernet header
        if(header->caplen < sizeof(sniff_ethernet)) return -2;
        ethernet = (struct sniff_ethernet*)(pkt_data);

        //get ip header
        if(header->caplen < sizeof(sniff_ip) + SIZE_ETHERNET) return -2;
        ip = (struct sniff_ip*)(pkt_data + SIZE_ETHERNET);
        size_ip = IP_HL(ip)*4;
        if (size_ip < 20) return -2;

        int header_lenght   = size_ip + SIZE_ETHERNET;
        int payload_lenght  = header->len - header_lenght;

        std::memcpy(buffer + len_buffer, pkt_data + header_lenght, payload_lenght);
        len_buffer += payload_lenght;

    }while(len_buffer < total_lenght);

    return len_buffer;
}

int 
tk::communication::PacketParser::computeNextPacket(pcap_t* pcapFile,uint8_t* buffer, timeStamp_t& stamp){

    int payload_lenght = this->firstFrame(pcapFile,stamp);

    int header_lenght = size_ip + SIZE_ETHERNET;

    //get if udp
    if(ip->ip_p == IPPROTO_UDP){

        udp = (struct sniff_udp*)(pkt_data + header_lenght);

        header_lenght   += SIZE_UDP;
        payload_lenght  -= SIZE_UDP;

        std::memcpy(buffer,pkt_data + header_lenght, payload_lenght);

        //Is Fragment?
        if(ntohs(ip->ip_len) != (ntohs(udp->uh_ulen)+size_ip)){
            return defragment(pcapFile,buffer,stamp,payload_lenght,ntohs(udp->uh_ulen)-SIZE_UDP);
        }

        return payload_lenght;
    }

    //get if tcp
    if(ip->ip_p == IPPROTO_TCP){

        tcp = (struct sniff_tcp*)(pkt_data + header_lenght);
        int size_tcp = TH_OFF(tcp)*4;
        if (size_tcp < 20) return -1;
        
        header_lenght   += size_tcp;
        payload_lenght  -= size_tcp;

        std::memcpy(buffer,pkt_data + header_lenght, payload_lenght);

        //Is Fragment?
        if(ntohs(ip->ip_len) != (ntohs(udp->uh_ulen)+size_ip)){
            return defragment(pcapFile,buffer,stamp,payload_lenght,ntohs(ip->ip_len)-size_tcp-size_ip); //TODO: need to test
        }

        return payload_lenght;
    }

    return -1;
}