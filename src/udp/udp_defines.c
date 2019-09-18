#include "tkCommon/udp/udp_defines.h"

#ifndef UDP_TRUE
  #define UDP_TRUE 1
#endif

#ifndef UDP_FALSE
  #define UDP_FALSE -1
#endif

#ifdef __cplusplus
  extern "C" {
#endif

#ifdef UDP_ON_WINDOWS

  uint64_t
  get_time_ms() {
    uint64_t time_ms;
    LARGE_INTEGER time_query;
    LARGE_INTEGER frequency;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time_query);
    time_query.QuadPart *= 1000000;
    time_ms = (time_query.QuadPart / frequency.QuadPart) / 1000;
    return time_ms;
  }

  void
  sleep_ms( uint32_t time_sleep_ms )
  { Sleep(time_sleep_ms); }

  typedef int ssize_t;

#else

  uint64_t
  get_time_ms() {
    uint64_t time_ms;
    struct timeval time_get;
    gettimeofday(&time_get, NULL);
    time_ms = (uint64_t) (time_get.tv_sec*1000 + time_get.tv_usec/1000);
    return time_ms;
  }

  void
  sleep_ms( uint32_t time_sleep_ms )
  { usleep(time_sleep_ms*1000); }

#endif

void
datagram_part_to_buffer(
  datagram_part_t const * D,
  uint8_t                 buffer[]
) {
  uint8_t * ptr = buffer;
  ptr += int32_to_buffer( D->datagram_id, ptr );
  ptr += uint32_to_buffer( D->total_message_size, ptr );
  ptr += uint16_to_buffer( D->sub_message_size, ptr );
  ptr += uint16_to_buffer( D->sub_message_position, ptr );
  memcpy( ptr, D->message, D->sub_message_size );
}

extern
void
buffer_to_datagram_part(
  uint8_t const     buffer[],
  datagram_part_t * D
) {
  uint8_t const * ptr = buffer;
  ptr += buffer_to_int32( ptr, &D->datagram_id );
  ptr += buffer_to_uint32( ptr, &D->total_message_size );
  ptr += buffer_to_uint16( ptr, &D->sub_message_size );
  ptr += buffer_to_uint16( ptr, &D->sub_message_position );
  if ( D->sub_message_size > UDP_DATAGRAM_MESSAGE_SIZE ) {
    UDP_printf(
      "buffer_to_datagram_part, sub_message_size (%d) > UDP_DATAGRAM_MESSAGE_SIZE (%d)\n",
      D->sub_message_size, UDP_DATAGRAM_MESSAGE_SIZE
    );
    exit(1);
  }
  memcpy( D->message, ptr, D->sub_message_size );
}

void
Packet_Init( packet_info_t * pi, uint64_t start_time_ms ) {
  pi->server_run         = UDP_FALSE;
  pi->datagram_id        = -1;
  pi->total_message_size = 0;
  pi->received_packets   = 0;
  pi->n_packets          = 0;
  pi->start_time_ms      = start_time_ms;
}

void
Packet_Add_to_buffer(
  packet_info_t         * pi,
  datagram_part_t const * pk,
  uint8_t                 buffer[],
  uint32_t                buffer_max_size
) {
  /* sanity check */
  // UDP_printf("Packet_Add_to_buffer 1\n");
  int32_t offs = pk->sub_message_position * UDP_DATAGRAM_MESSAGE_SIZE;
  int32_t ends = offs+pk->sub_message_size;
  if ( offs < 0 || ends > (int32_t)buffer_max_size ) {
    UDP_printf(
      "Packet_Add_to_buffer\noffs = %d\nbuffer_max_size = %d\nends = %d\nwrong sub_message_position (%d)\n",
      offs, buffer_max_size, ends, pk->sub_message_position
    );
    exit(1);
  }

  /* copia segmento nel buffer */
  memcpy(
    buffer + pk->sub_message_position * UDP_DATAGRAM_MESSAGE_SIZE,
    pk->message,
    pk->sub_message_size
  );

  int32_t ok = UDP_TRUE;
  if ( pi->received_packets > 0 ) {
    /* check that datagram_id is not changed */
    if ( pi->datagram_id        != pk->datagram_id ||
         pi->total_message_size != pk->total_message_size ) {
      ok = UDP_FALSE;
      UDP_printf("------- Packet lost! --------\n");
      UDP_printf("------- message id %6d ---------\n",pi->datagram_id);
      pi->received_packets = 0;
    }
  }

  /* first packet, compute number of packets */
  /* pi->server_run         = ?????; DA INDAGARE */
  pi->datagram_id        = pk->datagram_id;
  pi->total_message_size = pk->total_message_size;
  ++pi->received_packets;

/*
  if ( buffer_max_size < pk->total_message_size ) {


  }
*/
  /* compute total number of packets */
  pi->n_packets = pk->total_message_size / UDP_DATAGRAM_MESSAGE_SIZE;
  if ( (pk->total_message_size % UDP_DATAGRAM_MESSAGE_SIZE) != 0 ) ++pi->n_packets;

  #ifdef DEBUG_UDP
  UDP_printf("Packet received!\n");
  UDP_printf("server_run            = %s\n", ( pi->server_run == UDP_TRUE ?"TRUE":"FALSE" ) );
  UDP_printf("datagram_id           = %d\n",  pi->datagram_id);
  UDP_printf("n_packets             = %d\n",  pi->n_packets);
  UDP_printf("sub_message_position  = %hd\n", pk->sub_message_position);
  UDP_printf("sub_message_size      = %hd\n", pk->sub_message_size);
  UDP_printf("size of data received = %d\n",  true_sub_packet_size);
  #endif
}

void
Packet_Build_from_buffer(
  uint8_t const     buffer[],
  uint32_t          packet_size,
  uint16_t          pos,
  int32_t           datagram_id,
  datagram_part_t * pk
) {
  uint32_t ppos = pos * UDP_DATAGRAM_MESSAGE_SIZE;
  pk->datagram_id          = datagram_id;
  pk->total_message_size   = packet_size;
  pk->sub_message_position = pos;
  pk->sub_message_size     = UDP_DATAGRAM_MESSAGE_SIZE;
  if ( pk->sub_message_size + ppos > packet_size ) pk->sub_message_size = (uint16_t)(packet_size - ppos);
  memcpy(
    pk->message,
    buffer + pos * UDP_DATAGRAM_MESSAGE_SIZE,
    pk->sub_message_size
  );
}

extern
uint32_t
Packet_Number( uint32_t message_size ) {
  uint32_t n_packets = message_size / UDP_DATAGRAM_MESSAGE_SIZE;
  if ( (message_size % UDP_DATAGRAM_MESSAGE_SIZE) != 0 ) ++n_packets;
  return n_packets;
}

#ifdef __cplusplus
}
#endif
