#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <netinet/in.h>
#include <netinet/ip.h>
#include <net/if.h>
#include <netinet/if_ether.h>

#include <pcap.h>
#include <bitset>

/* ethernet headers are always exactly 14 bytes [1] */
#define SIZE_UDP 8

/* ethernet headers are always exactly 14 bytes [1] */
#define SIZE_ETHERNET 14

/* Ethernet header */
struct sniff_ethernet {
        u_char  ether_dhost[ETHER_ADDR_LEN];    /* destination host address */
        u_char  ether_shost[ETHER_ADDR_LEN];    /* source host address */
        u_short ether_type;                     /* IP? ARP? RARP? etc */
};

/* IP header */
struct sniff_ip {
        u_char  ip_vhl;                 /* version << 4 | header length >> 2 */
        u_char  ip_tos;                 /* type of service */
        u_short ip_len;                 /* total length */
        u_short ip_id;                  /* identification */
        u_short ip_off;                 /* fragment offset field */
        #define IP_RF 0x8000            /* reserved fragment flag */
        #define IP_DF 0x4000            /* dont fragment flag */
        #define IP_MF 0x2000            /* more fragments flag */
        #define IP_OFFMASK 0x1fff       /* mask for fragmenting bits */
        u_char  ip_ttl;                 /* time to live */
        u_char  ip_p;                   /* protocol */
        u_short ip_sum;                 /* checksum */
        struct  in_addr ip_src,ip_dst;  /* source and dest address */
};
#define IP_HL(ip)               (((ip)->ip_vhl) & 0x0f)
#define IP_V(ip)                (((ip)->ip_vhl) >> 4)

/* TCP header */
typedef u_int tcp_seq;

struct sniff_tcp {
        u_short th_sport;               /* source port */
        u_short th_dport;               /* destination port */
        tcp_seq th_seq;                 /* sequence number */
        tcp_seq th_ack;                 /* acknowledgement number */
        u_char  th_offx2;               /* data offset, rsvd */
#define TH_OFF(th)      (((th)->th_offx2 & 0xf0) >> 4)
        u_char  th_flags;
        #define TH_FIN  0x01
        #define TH_SYN  0x02
        #define TH_RST  0x04
        #define TH_PUSH 0x08
        #define TH_ACK  0x10
        #define TH_URG  0x20
        #define TH_ECE  0x40
        #define TH_CWR  0x80
        #define TH_FLAGS        (TH_FIN|TH_SYN|TH_RST|TH_ACK|TH_URG|TH_ECE|TH_CWR)
        u_short th_win;                 /* window */
        u_short th_sum;                 /* checksum */
        u_short th_urp;                 /* urgent pointer */
};

struct sniff_udp {
         u_short uh_sport;               /* source port */
         u_short uh_dport;               /* destination port */
         u_short uh_ulen;                /* udp length */
         u_short uh_sum;                 /* udp checksum */

};


int skipHeaderAndUnfrag(pcap_t* pcapFile,uint8_t* buffer, timeStamp_t& stamp){

	const struct sniff_ethernet *ethernet;	/* The ethernet header [1] */
	const struct sniff_ip 		*ip;		/* The IP header */
	const struct sniff_tcp 		*tcp;		/* The TCP header */
	const struct sniff_udp 		*udp;		/* The UDP header */


	struct pcap_pkthdr 	*header;			/* The pointer to header info */
    const u_char 		*pkt_data;			/* The pointer to data */

    int size_ip;

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

    //std::cout<<(ip->ip_off & IP_OFFMASK)<<std::endl;
    //std::cout<<ip->ip_len<<std::endl;

    std::bitset<16> y(ip->ip_off);
    //std::cout <<"...."<<y << '\n';

    }while(ip->ip_off != 32 &&  ip->ip_off != 64);//64 no fragment, 32 first fragment

    int header_lenght = SIZE_ETHERNET + size_ip;

    //get if udp
    if(ip->ip_p == IPPROTO_UDP){

        udp = (struct sniff_udp*)(pkt_data + header_lenght);
        header_lenght += SIZE_UDP;

        if(ntohs(ip->ip_len) != (ntohs(udp->uh_ulen)+size_ip)){
                int len = ntohs(udp->uh_ulen) - SIZE_UDP;

                int lunghezza_effettiva = header->len - header_lenght;

                std::memcpy(buffer,pkt_data + header_lenght,lunghezza_effettiva);

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

                    int lunghezza = size_ip + SIZE_ETHERNET;

                    int copia = header->len - lunghezza;

                    std::memcpy(buffer+lunghezza_effettiva,pkt_data + lunghezza,copia);

                    lunghezza_effettiva +=copia;

                    std::bitset<16> y(ip->ip_off);
    //std::cout <<"...."<<y << '\n';

                    //std::cout<<lunghezza_effettiva<<std::endl;

                    

                }while(lunghezza_effettiva != len);

                return len;



        }

    }else{

        //get if tcp
        if(ip->ip_p == IPPROTO_TCP){

            tcp = (struct sniff_tcp*)(pkt_data + SIZE_ETHERNET + size_ip);
            int size_tcp = TH_OFF(tcp)*4;
            if (size_tcp < 20) return -2;
            header_lenght += size_tcp;

        }else{

            return -3;
        }
    }

    int payload_lenght = header->len - header_lenght;

    std::bitset<16> y(ip->ip_off);
    //std::cout << payload_lenght<<"...."<<y << '\n';

    //std::cout<<payload_lenght<<std::endl;

    std::memcpy(buffer,pkt_data + header_lenght ,payload_lenght);

    return payload_lenght;
}


















//GATTO
///////////////////////////////////////////////////////////////////
/*
struct PCAP_header {
    int vlanID;
    int headerLen;
    int dataLen;

}__attribute__ ((__packed__));

struct UDP_hdr {
	u_short	uh_sport;		
	u_short	uh_dport;		
	u_short	uh_ulen;	
	u_short	uh_sum;			
};

bool parse_UDP_packet(const unsigned char *packet, unsigned int capture_len, PCAP_header &out) {
	struct ip *ip;
	struct ether_header *eth;
	struct UDP_hdr *udp;
	unsigned int IP_header_length;
	int vlanid = 0;
	int packetLen = capture_len;

	// For simplicity, we assume Ethernet encapsulation. 
	if (capture_len < sizeof(struct ether_header)) {
		return false;
    }

	// get ethernet type
    eth = (struct ether_header*) packet;
	uint8_t ethtype;
	memcpy(&ethtype, &eth->ether_type, sizeof(uint8_t));

	// Skip the Ethernet header. 
	packet += sizeof(struct ether_header);
	capture_len -= sizeof(struct ether_header);

	if (capture_len < sizeof(struct ip)) {
	     //Didn't capture a full IP header 
		return false;
    }

	// if VLAN read it
    if (ethtype == 0x81) {
        packet += sizeof(uint8_t);
        uint8_t vlanid_8 = *packet;
        vlanid = int(vlanid_8);
        packet += 3*sizeof(uint8_t);
        capture_len -= 4*sizeof(uint8_t);
    }

    ip = (struct ip*) packet;
	IP_header_length = ip->ip_hl * 4;	// ip_hl is in 4-byte words 
	if (capture_len < IP_header_length) {
	    std::cout<<"didn't capture the full IP header including options\n";
		return false;
    }

	if (ip->ip_p != IPPROTO_UDP) {
	    //std::cout<<"NON UDP packet: "<<int(ip->ip_p)<<"\n";
		return false;
    }

	// Skip over the IP header to get to the UDP header. 
	packet += IP_header_length;
	capture_len -= IP_header_length;

	if (capture_len < sizeof(struct UDP_hdr)) {
		// to short UDP header
		return false;
    }

	udp = (struct UDP_hdr*) packet;
	capture_len -= sizeof(UDP_hdr);

	// fill out struct
	out.dataLen = capture_len;
	out.headerLen = packetLen - capture_len;
	out.vlanID = vlanid;
	return true;
}

void dump_PCAP_header(PCAP_header &out) {
    std::cout<<"PCAP header:\n";
    std::cout<<"\theaderLen: "<<out.headerLen<<"\n";
    std::cout<<"\tdataLen: "<<out.dataLen<<"\n";
    std::cout<<"\tvlan: "<<out.vlanID<<"\n";
}*/