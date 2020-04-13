#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <netinet/in.h>
#include <netinet/ip.h>
#include <net/if.h>
#include <netinet/if_ether.h>

#include <pcap.h>

struct PCAP_header {
    int vlanID;
    int headerLen;
    int dataLen;

}__attribute__ ((__packed__));

struct UDP_hdr {
	u_short	uh_sport;		/* source port */
	u_short	uh_dport;		/* destination port */
	u_short	uh_ulen;		/* datagram length */
	u_short	uh_sum;			/* datagram checksum */
};

bool parse_UDP_packet(const unsigned char *packet, unsigned int capture_len, PCAP_header &out) {
	struct ip *ip;
	struct ether_header *eth;
	struct UDP_hdr *udp;
	unsigned int IP_header_length;
	int vlanid = 0;
	int packetLen = capture_len;

	/* For simplicity, we assume Ethernet encapsulation. */
	if (capture_len < sizeof(struct ether_header)) {
		return false;
    }

	// get ethernet type
    eth = (struct ether_header*) packet;
	uint8_t ethtype;
	memcpy(&ethtype, &eth->ether_type, sizeof(uint8_t));

	/* Skip the Ethernet header. */
	packet += sizeof(struct ether_header);
	capture_len -= sizeof(struct ether_header);

	if (capture_len < sizeof(struct ip)) {
	    /* Didn't capture a full IP header */
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
	IP_header_length = ip->ip_hl * 4;	/* ip_hl is in 4-byte words */
	if (capture_len < IP_header_length) {
	    std::cout<<"didn't capture the full IP header including options\n";
		return false;
    }

	if (ip->ip_p != IPPROTO_UDP) {
	    //std::cout<<"NON UDP packet: "<<int(ip->ip_p)<<"\n";
		return false;
    }

	/* Skip over the IP header to get to the UDP header. */
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
}