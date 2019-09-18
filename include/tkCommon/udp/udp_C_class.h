
/* ============================================================================
 UDP communication with limited packed size
 ============================================================================ */

#ifndef __UDP_C_CLASS_HH
#define __UDP_C_CLASS_HH

#include "udp_defines.h"

#ifndef UDP_TRUE
  #define UDP_TRUE 1
#endif

#ifndef UDP_FALSE
  #define UDP_FALSE -1
#endif

#ifdef __cplusplus
  #include <cstddef>
#else
  #define nullptr NULL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*\
 |   ____             _        _
 |  / ___|  ___   ___| | _____| |_
 |  \___ \ / _ \ / __| |/ / _ \ __|
 |   ___) | (_) | (__|   <  __/ |_
 |  |____/ \___/ \___|_|\_\___|\__|
\*/

typedef struct  {
#ifdef UDP_ON_WINDOWS
  SOCKET             socket_id;
#else
  int                socket_id;
#endif
  int                connected;
  struct sockaddr_in sock_addr;
  socklen_t          sock_addr_len;
  uint64_t           timeout_ms;
} SocketData;

extern
void
Socket_new( SocketData * pS );

extern
void
Socket_check( SocketData * pS );

extern
int
Socket_open_as_client(
  SocketData * pS,
  char const   addr[],
  int          port,
  int          connect
);

extern
int
Socket_open_as_server( SocketData * pS, int bind_port );

extern
int
Socket_close( SocketData * pS );

extern
int
Socket_send(
  SocketData * pS,
  int32_t      message_id,
  uint8_t      message[],
  uint32_t     message_size
);

extern
int
Socket_receive(
  SocketData * pS,
  int32_t    * message_id,
  int32_t    * message_len,
  uint8_t      message[],
  uint32_t     message_max_size,
  uint64_t     start_time
);

extern
int
Socket_send_raw(
  SocketData *  pS,
  uint8_t const message[],
  uint32_t      message_size
);

extern
int
Socket_receive_raw(
  SocketData * pS,
  uint8_t      message[],
  uint32_t     message_size
);

/*\
 |   __  __       _ _   _               _
 |  |  \/  |_   _| | |_(_) ___ __ _ ___| |_
 |  | |\/| | | | | | __| |/ __/ _` / __| __|
 |  | |  | | |_| | | |_| | (_| (_| \__ \ |_
 |  |_|  |_|\__,_|_|\__|_|\___\__,_|___/\__|
\*/

#ifndef UDP_NO_MULTICAST_SUPPORT
extern
int
MultiCast_open_as_sender(
  SocketData * pS,
  char const   group_address[],
  int          group_port
);

extern
int
MultiCast_open_as_listener(
  SocketData * pS,
  char const   group_address[],
  int          group_port
);
#endif

#ifdef __cplusplus
}
#endif

#endif
