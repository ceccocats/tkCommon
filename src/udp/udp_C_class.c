/* ============================================================================
  UDP communication with limited packed size
 ============================================================================ */

#include "tkCommon/udp/udp_C_class.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef UDP_ON_WINDOWS
  #include "tkCommon/udp/udp_C_win_code.h"
  #include <Ws2tcpip.h>
  typedef int ssize_t;
#else
  #include "tkCommon/udp/udp_C_unix_code.h"
#endif

/*\
 |   ____             _        _
 |  / ___|  ___   ___| | _____| |_
 |  \___ \ / _ \ / __| |/ / _ \ __|
 |   ___) | (_) | (__|   <  __/ |_
 |  |____/ \___/ \___|_|\_\___|\__|
\*/

void
Socket_new( SocketData * pS ) {
  pS->socket_id  = -1;
  pS->connected  = UDP_FALSE;
  pS->timeout_ms = UDP_APP_TIMEOUT_MS;
}

void
Socket_check( SocketData * pS ) {
  if ( pS->socket_id >= 0 ) {
    UDP_printf( "Opened socket id = %d\n", (int)pS->socket_id );
  } else {
    UDP_printf( "Socket not opened\n" );
    exit(-1);
  }
}

/*\
 | Send message function
\*/

int
Socket_send(
  SocketData * pS,
  int32_t      message_id,
  uint8_t      message[],
  uint32_t     message_size
) {

  uint32_t        n_packets;
  datagram_part_t packet;
  uint16_t        ipos;
  #ifdef UDP_ON_WINDOWS
  int             isend;
  #else
  ssize_t         isend;
  #endif

  #if defined(WIN_NONBLOCK)
    uint64_t socket_start_time;
    uint64_t socket_elapsed_time;
  #endif

  n_packets = Packet_Number( message_size );

  /* Send packets */
  for ( ipos = 0; ipos < n_packets; ++ipos ) {

    /* estrae pacchetto */
    Packet_Build_from_buffer(
      message, message_size, ipos, message_id, &packet
    );

    /* serializza pacchetto */
    uint8_t data_buffer[UDP_MTU_MAX_BYTES];
    datagram_part_to_buffer( &packet, data_buffer );

    #ifdef UDP_ON_WINDOWS
    int nbytes = (size_t) (UDP_DATAGRAM_PART_HEADER_SIZE+packet.sub_message_size);
    #else
    size_t nbytes = (size_t) (UDP_DATAGRAM_PART_HEADER_SIZE+packet.sub_message_size);
    #endif

    #if defined(WIN_NONBLOCK)
    socket_start_time = get_time_ms();
    while ( 1 ) {
      isend = sendto(
        pS->socket_id,
        data_buffer, nbytes,
        0,
        (struct sockaddr *) &pS->sock_addr, pS->sock_addr_len
      );
      if ( isend < 0 ) {
        socket_elapsed_time = get_time_ms() - socket_start_time;
        if ( WSAGetLastError() != WSAEWOULDBLOCK ||
             socket_elapsed_time >= UDP_RECV_SND_TIMEOUT_MS ) {
          UDP_CheckError( "sendto() failed" );
          return UDP_FALSE;
        }
      } else {
        break;
      }
    }
    #elif defined(UDP_ON_WINDOWS)
    isend = sendto(
      socket_id,
      data_buffer, nbytes,
      0,
      (struct sockaddr *) &pS->sock_addr, pS->sock_addr_len
    );
    if ( isend < 0 ) {
      UDP_CheckError( "sendto() failed" );
      return UDP_FALSE;
    }
    #elif defined(__MACH__) || defined(__linux__)
    isend = sendto(
      pS->socket_id,
      data_buffer, nbytes,
      0,
      (struct sockaddr *) &pS->sock_addr, pS->sock_addr_len
    );
    if ( isend < 0 ) {
      #ifdef UDP_ON_WINDOWS
      UDP_CheckError( "error sendto" );
      #else
      char error_str[1024];
      strerror_r( errno, error_str, 1024 );
      UDP_printf("error sendto: %s\n",error_str);
      #endif
      return UDP_FALSE;
    }
    #endif
  }

  #ifdef DEBUG_UDP
  UDP_printf(
    "Sent message of %d packets to %s:%d\n",
     n_packets, pS->sock_addr.sin_addr, pS->sock_addr.sin_port
  );
  #endif
  return UDP_TRUE;
}
/*\
 | Receive message function
\*/

/* return number of bytes received or -1 */
int
Socket_receive(
  SocketData * pS,
  int32_t    * p_message_id,
  int32_t    * p_message_len,
  uint8_t      message[],
  uint32_t     message_max_size,
  uint64_t     start_time_ms
) {

  datagram_part_t packet;
  ssize_t         recv_bytes      = 0;
  uint64_t        elapsed_time_ms = 0;

  #if defined(WIN_NONBLOCK)
  uint64_t socket_start_time;
  uint64_t socket_elapsed_time;
  #endif

  packet_info_t pi;
  Packet_Init( &pi, start_time_ms );

  /* Receive packets */
  elapsed_time_ms = start_time_ms == 0 ? 0 : get_time_ms() - start_time_ms;
  while ( elapsed_time_ms <= pS->timeout_ms ) {

    uint8_t data_buffer[UDP_MTU_MAX_BYTES];

    #if defined(WIN_NONBLOCK)
    socket_start_time = get_time_ms();
    while ( 1 ) {
      pS->sock_addr_len = sizeof(pS->sock_addr);
      recv_bytes = recvfrom(
        pS->socket_id, data_buffer, (size_t) UDP_MTU_MAX_BYTES,
        0, NULL, NULL
      );
      socket_elapsed_time = get_time_ms() - socket_start_time;

      if ( recv_bytes < 0 ) {
        if ( WSAGetLastError() != WSAEWOULDBLOCK ||
             socket_elapsed_time >= UDP_RECV_SND_TIMEOUT_MS ) break;
      } else {
        break;
      }
    }
    #elif defined(UDP_ON_WINDOWS)
    pS->sock_addr_len = sizeof(pS->sock_addr);
    recv_bytes = recvfrom(
      pS->socket_id, data_buffer, (size_t) UDP_MTU_MAX_BYTES,
      0, NULL, NULL
    );
    #elif defined(__MACH__) || defined(__linux__)
    pS->sock_addr_len = sizeof(pS->sock_addr);
    recv_bytes = recvfrom(
      pS->socket_id, data_buffer, (size_t) UDP_MTU_MAX_BYTES,
      0, NULL, NULL
    );
    #endif

    if ( recv_bytes > 0 ) {
      /* deserializza */
      buffer_to_datagram_part( data_buffer, &packet );
      Packet_Add_to_buffer( &pi, &packet, message, message_max_size );
    } else {
      sleep_ms(UDP_SLEEP_MS);
    }

    if ( pi.received_packets == pi.n_packets && pi.n_packets > 0 ) break;

    /* Calculate elapsed time */
    if ( start_time_ms != 0 )
      elapsed_time_ms = get_time_ms() - start_time_ms;

  }

  if ( pi.received_packets == pi.n_packets ) {
    *p_message_id  = pi.datagram_id;
    *p_message_len = (int32_t)pi.total_message_size;
    #ifdef DEBUG_UDP
    UDP_printf(
      "Received message of %d packets from %s:%d\n",
      pi.n_packets,
      pS->sock_addr.sin_addr,
      pS->sock_addr.sin_port
    );
    #endif
    return UDP_TRUE;
  } else if ( elapsed_time_ms >= pS->timeout_ms ) {
    UDP_printf(
      "Receive Warning: Time-out reached! Timeout is: %lu Time needed: %lu\n",
      pS->timeout_ms, elapsed_time_ms
    );
    return UDP_FALSE;
  } else {
    UDP_printf( "Receive Warning: Server not running'n" );
    return UDP_FALSE;
  }

}

#ifdef __cplusplus
}
#endif
