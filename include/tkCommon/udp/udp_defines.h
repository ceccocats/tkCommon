
/* ============================================================================
 UDP communication with limited packed size
 ============================================================================ */

#ifndef __UDP_DEFINES_HH
#define __UDP_DEFINES_HH

/*\

uncomment to remove multicast support

#define UDP_NO_MULTICAST_SUPPORT

\*/

#include "buffer_defines.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined (_DS1401)
  #include "ds1401_defines.h"
#endif

#ifndef UDP_MTU_MAX_MAX_BYTES
  #define UDP_MTU_MAX_MAX_BYTES 65536  /* Maximum packet bytes */
#endif

#ifndef UDP_MTU_MAX_BYTES
  #define UDP_MTU_MAX_BYTES 1472  /* Maximum packet bytes */
#endif

/* Times */
#define UDP_SLEEP_MS             1
#define UDP_APP_TIMEOUT_MS      50
#define UDP_RECV_SND_TIMEOUT_MS  5  /* Warning: windows has an undocumented minimum limit of about 500 ms */

/* If the timeout is less than 400 ms it creates a non-blocking socket */
#ifdef UDP_ON_WINDOWS
  #pragma warning (disable : 4820)
  #if UDP_RECV_SND_TIMEOUT_MS <= 400
    #define WIN_NONBLOCK
  #endif
#endif

#define UDP_DATAGRAM_PART_HEADER_SIZE 12
#define UDP_DATAGRAM_MESSAGE_SIZE     (UDP_MTU_MAX_BYTES-UDP_DATAGRAM_PART_HEADER_SIZE)

typedef struct {
  int32_t  datagram_id;          /* message ID */
  uint32_t total_message_size;   /* total length of the packet */
  uint16_t sub_message_size;     /* sub packet size */
  uint16_t sub_message_position; /* sub packet position in the message */
  uint8_t  message[UDP_DATAGRAM_MESSAGE_SIZE]; /* part of datagram message */
} datagram_part_t;

extern
void
datagram_part_to_buffer(
  datagram_part_t const * D,
  uint8_t                 buffer[]
);

extern
void
buffer_to_datagram_part(
  uint8_t const     buffer[],
  datagram_part_t * D
);

typedef struct {
  int32_t  server_run;
  int32_t  datagram_id;
  uint32_t total_message_size;
  uint32_t received_packets;
  uint32_t n_packets;
  uint64_t start_time_ms;
} packet_info_t;

extern
void
Packet_Init( packet_info_t * pi, uint64_t start_time_ms );

extern
void
Packet_Add_to_buffer(
  packet_info_t         * pi,
  datagram_part_t const * pk,
  uint8_t                 buffer[],
  uint32_t                buffer_max_size
);

extern
void
Packet_Build_from_buffer(
  uint8_t const     buffer[],
  uint32_t          packet_size,
  uint16_t          pos,
  int32_t           datagram_id,
  datagram_part_t * pk
);

extern
uint32_t
Packet_Number( uint32_t packet_size );

/* Get time function (milliseconds) */
extern uint64_t get_time_ms( void );

/* Sleep function (milliseconds) */
extern void sleep_ms( uint32_t time_sleep_ms );

#ifdef __cplusplus
}
#endif

#ifdef UDP_ON_WINDOWS
  #include <Winsock2.h>
  typedef int socklen_t;
#else
  #include <unistd.h>
  #include <arpa/inet.h>
  #include <netinet/in.h>
  /*
  DA INGARE SE SERVE
  #if ! ( defined(__APPLE__) || defined(__MACH__) )
    #include <linux/in.h>
  #endif
  */
  #include <sys/socket.h>
  #include <sys/types.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <time.h>
#endif

/* -------------------------------------------------- */

#endif
