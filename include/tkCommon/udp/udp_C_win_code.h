#ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include <Winsock.h>
#include <Winsock2.h>
#include <Ws2tcpip.h>

#pragma comment (lib, "Ws2_32.lib")

static
void
UDP_CheckError( char const msg[] ) {
  /* Get the error message, if any. */
  DWORD errorMessageID = WSAGetLastError();
  if ( errorMessageID == 0 ) return ; /* No error message has been recorded */
  LPSTR messageBuffer = NULL;
  size_t size = FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
    FORMAT_MESSAGE_FROM_SYSTEM |
    FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL,
    errorMessageID,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    (LPSTR)&messageBuffer,
    0,
    NULL
  );
  UDP_printf("[%s]\n%s", msg, messageBuffer);
  LocalFree(messageBuffer);
}

/*\
 |   ____             _        _
 |  / ___|  ___   ___| | _____| |_
 |  \___ \ / _ \ / __| |/ / _ \ __|
 |   ___) | (_) | (__|   <  __/ |_
 |  |____/ \___/ \___|_|\_\___|\__|
\*/

int
Socket_open_as_client(
  SocketData * pS,
  char const   addr[],
  int          port,
  int          conn
) {
  int      ret;
  unsigned opt_buflen;
  struct timeval timeout;
  char ipAddress[INET_ADDRSTRLEN];

  WSADATA wsa;

  pS->socket_id = 0;

  if ( WSAStartup(MAKEWORD(2, 2), &wsa) != 0 ) {
    UDP_CheckError( "error: WSAStartup() failed" );
    return UDP_FALSE;
  }

  /* Create UDP socket */
  pS->socket_id = (int32_t)socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if ( pS->socket_id < 0 ) {
    UDP_CheckError("Socket_open_as_client::socket");
    return UDP_FALSE;
  }

  /* Set send buffer size limit */
  opt_buflen = UDP_MTU_MAX_MAX_BYTES;
  ret = setsockopt(
    pS->socket_id,
    SOL_SOCKET,
    SO_SNDBUF,
    (char *)&opt_buflen,
    sizeof(opt_buflen)
  );
  if ( ret < 0 ) {
    UDP_CheckError("Socket_open_as_client::setsockopt<buffer lenght>");
    return UDP_FALSE;
  }

  /*\
   | Set send time-outs
   | Windows: it is used a non-blocking
   | socket if defined time-out <= 400 ms
  \*/

  timeout.tv_sec = 0;
  timeout.tv_usec = UDP_RECV_SND_TIMEOUT_MS * 1000;

  ret = setsockopt(
    pS->socket_id,
    SOL_SOCKET,
    SO_SNDTIMEO,
    (char *)&timeout,
    sizeof(timeout)
  );
  if ( ret < 0 ) {
    UDP_CheckError("Socket_open_as_client::setsockopt<timeout>");
    return UDP_FALSE;
  }

  /* Clear the address structures */
  memset( &pS->sock_addr, 0, sizeof(pS->sock_addr) );
  pS->sock_addr_len = sizeof(pS->sock_addr);

  /* Set the address structures */
  pS->sock_addr.sin_family = AF_INET;
  pS->sock_addr.sin_port   = htons( (uint16_t)port);
  InetPton( AF_INET, addr, (PVOID)&pS->sock_addr.sin_addr.s_addr );

  /* Connect to server. */
  pS->connected = conn;
  if ( conn == UDP_TRUE ) {
    ret = connect(
      pS->socket_id,
      (const struct sockaddr *)&pS->sock_addr,
      pS->sock_addr_len
    );
    if ( ret < 0 ) {
      UDP_CheckError( "Socket_open_as_client::connect" );
      closesocket(pS->socket_id);
      pS->socket_id = -1;
      exit(1);
    }
  }

  inet_ntop(
    AF_INET,
    &(pS->sock_addr.sin_addr.s_addr),
    ipAddress,
    INET_ADDRSTRLEN
  );
  UDP_printf("======================================\n");
  UDP_printf("CLIENT\n");
  UDP_printf("address: %s\n", ipAddress);
  UDP_printf("port:    %d\n", ntohs(pS->sock_addr.sin_port));
  UDP_printf("======================================\n");
  return UDP_TRUE;
}

/*\
 | Open socket
\*/

int
Socket_open_as_server( SocketData * pS, int bind_port ) {

  int ret, yes;
  struct timeval timeout;
  char ipAddress[INET_ADDRSTRLEN];
  WSADATA wsa;

  pS->socket_id = 0;

  if ( WSAStartup(MAKEWORD(2, 2), &wsa) != 0 ) {
    UDP_CheckError( "error: WSAStartup() failed" );
    return UDP_FALSE;
  }

  /* Create UDP socket */
  pS->socket_id = (int32_t)socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
  if ( pS->socket_id < 0 ) {
    UDP_CheckError("Socket_open_as_server::socket");
    return UDP_FALSE;
  }

  yes = 1;
  ret = setsockopt(
    pS->socket_id,
    SOL_SOCKET,
    SO_REUSEADDR,
    (char const *)&yes,
    sizeof(yes)
  );
  if ( ret < 0 ) {
    UDP_CheckError("Socket_open_as_server::setsockopt<reuseaddr>");
    return UDP_FALSE;
  }

  /*\
   | Set receive time-outs
   | Windows: it is used a non-blocking
   | socket if defined time-out <= 400 ms
  \*/

  timeout.tv_sec = 0;
  timeout.tv_usec = UDP_RECV_SND_TIMEOUT_MS * 1000;

  ret = setsockopt(
    pS->socket_id,
    SOL_SOCKET,
    SO_RCVTIMEO,
    (char *)&timeout,
    sizeof(timeout)
  );
  if ( ret < 0 ) {
    UDP_CheckError("Socket_open_as_server::setsockopt<timeout>");
    return UDP_FALSE;
  }

  /*\
   | If it is a server, bind socket to port
  \*/

  memset((char *)&pS->sock_addr, 0, sizeof(pS->sock_addr));
  pS->sock_addr.sin_family      = AF_INET;
  pS->sock_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  pS->sock_addr.sin_port        = htons(bind_port);
  pS->sock_addr_len             = sizeof(pS->sock_addr);

  ret = bind( pS->socket_id, (struct sockaddr *) &pS->sock_addr, pS->sock_addr_len );
  if ( ret < 0 ) {
    UDP_CheckError("Socket_open_as_server::bind");
    return UDP_FALSE;
  }

  inet_ntop(
    AF_INET,
    &(pS->sock_addr.sin_addr.s_addr),
    ipAddress,
    INET_ADDRSTRLEN
  );
  UDP_printf("======================================\n");
  UDP_printf("SERVER\n");
  UDP_printf("address: %s\n", ipAddress);
  UDP_printf("port:    %d\n", ntohs(pS->sock_addr.sin_port));
  UDP_printf("======================================\n");

  return UDP_TRUE;
}

/*\
 | Close socket
\*/

int
Socket_close( SocketData * pS ) {
  if ( closesocket(pS->socket_id) < 0 ) {
    UDP_CheckError( "error: setsockopt() failed" );
    WSACleanup();
    return UDP_FALSE;
  }
  WSACleanup();
  return UDP_TRUE;
}

/*\
 |                      _
 |   ___  ___ _ __   __| |    _ __ __ ___      __
 |  / __|/ _ \ '_ \ / _` |   | '__/ _` \ \ /\ / /
 |  \__ \  __/ | | | (_| |   | | | (_| |\ V  V /
 |  |___/\___|_| |_|\__,_|___|_|  \__,_| \_/\_/
 |                      |_____|
\*/

int
Socket_send_raw(
  SocketData *  pS,
  uint8_t const message[],
  uint32_t      message_size
) {
  int n_byte_sent;
  if ( pS->connected == UDP_TRUE ) {
    UDP_printf("Socket_send_raw::send\n");
    n_byte_sent = send( pS->socket_id, message, (size_t) message_size, 0 );
  } else {
    UDP_printf("Socket_send_raw::sendto\n");
    n_byte_sent = sendto(
      pS->socket_id,
      message,
      (size_t) message_size,
      0,
      (struct sockaddr *) &pS->sock_addr,
      pS->sock_addr_len
    );
  }
  if ( n_byte_sent == (int) message_size ) {
    return UDP_TRUE;
  } else {
    UDP_CheckError("Socket_send_raw");
    return UDP_FALSE;
  }
}

/*\
 |                     _
 |   _ __ ___  ___ ___(_)_   _____     _ __ __ ___      __
 |  | '__/ _ \/ __/ _ \ \ \ / / _ \   | '__/ _` \ \ /\ / /
 |  | | |  __/ (_|  __/ |\ V /  __/   | | | (_| |\ V  V /
 |  |_|  \___|\___\___|_| \_/ \___|___|_|  \__,_| \_/\_/
 |                               |_____|
\*/

int
Socket_receive_raw(
  SocketData * pS,
  uint8_t      message[],
  uint32_t     message_size
) {
  int ret = recvfrom(
    pS->socket_id,
    message,
    (size_t) message_size,
    0,
    (struct sockaddr *) &pS->sock_addr, &pS->sock_addr_len
  );
  return ret; /* if < 0 no data received */
}
/*\
 |   __  __       _ _   _               _
 |  |  \/  |_   _| | |_(_) ___ __ _ ___| |_
 |  | |\/| | | | | | __| |/ __/ _` / __| __|
 |  | |  | | |_| | | |_| | (_| (_| \__ \ |_
 |  |_|  |_|\__,_|_|\__|_|\___\__,_|___/\__|
\*/

#ifndef UDP_NO_MULTICAST_SUPPORT

int
MultiCast_open_as_listener(
  SocketData * pS,
  char const   group_address[],
  int          group_port
) {

  int ret, yes;
  struct ip_mreq mreq;
  WSADATA wsa;

  pS->connected = UDP_FALSE;
  pS->socket_id = 0;

  if ( WSAStartup(MAKEWORD(2, 2), &wsa) != 0 ) {
    UDP_CheckError( "error: WSAStartup() failed" );
    return UDP_FALSE;
  }

  /* Create UDP socket */
  pS->socket_id = (int32_t)socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if ( pS->socket_id < 0 ) {
    UDP_CheckError("MultiCast_open_as_listener::socket");
    return UDP_FALSE;
  } else {
    UDP_printf("MultiCast_open_as_listener::socket...OK.\n");
  }

  /* Preparatios for using Multicast */
  InetPton(AF_INET, group_address, (PVOID)&mreq.imr_multiaddr.s_addr);
  mreq.imr_interface.s_addr = htonl(INADDR_ANY);
  ret = setsockopt(
    pS->socket_id,
    IPPROTO_IP,
    IP_ADD_MEMBERSHIP,
    (char const *)&mreq,
    sizeof(mreq)
  );
  if ( ret < 0 ) {
    UDP_CheckError("MultiCast_open_as_listener::setsockopt<mreq>");
    return UDP_FALSE;
  }

  /* allow multiple sockets to use the same PORT number */
  yes = 1;
  ret = setsockopt( pS->socket_id, SOL_SOCKET, SO_REUSEADDR, (char*) &yes, sizeof(yes) );
  if ( ret < 0 ) {
    UDP_CheckError("MultiCast_open_as_listener::setsockopt<SO_REUSEADDR>");
    return UDP_FALSE;
  }

#if 0
  ret = setsockopt( pS->socket_id, SOL_SOCKET, SO_REUSEPORT, (char*) &yes, sizeof(yes) );
  if ( ret < 0 ) {
    UDP_CheckError("MultiCast_open_as_listener::setsockopt<SO_REUSEPORT>");
    return UDP_FALSE;
  }
#endif

  memset((char *)&pS->sock_addr, 0, sizeof(pS->sock_addr));
  pS->sock_addr.sin_family      = AF_INET;
  pS->sock_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  pS->sock_addr.sin_port        = htons(group_port);
  pS->sock_addr_len             = sizeof(pS->sock_addr);

  /* bind to receive address */
  ret = bind( pS->socket_id, (struct sockaddr *) &pS->sock_addr, sizeof(pS->sock_addr) );
  if ( ret < 0 ) {
    UDP_CheckError("MultiCast_open_as_listener::bind");
    return UDP_FALSE;
  }
  return UDP_TRUE;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

int
MultiCast_open_as_sender(
  SocketData * pS,
  char const   group_address[],
  int          group_port
) {

  WORD    wVersionRequested;
  WSADATA wsaData;
  int     err;

  /* Use the MAKEWORD(lowbyte, highbyte) macro declared in Windef.h */
  wVersionRequested = MAKEWORD(2, 2);

  err = WSAStartup(wVersionRequested, &wsaData);
  if (err != 0) {
    /* Tell the user that we could not find a usable */
    /* Winsock DLL.                                  */
    UDP_printf("WSAStartup failed with error: %d\n", err);
	  return UDP_FALSE;
  } else {
    UDP_printf("UDP STREAMING Opening the datagram socket...OK.\n");
  }

  /* Create UDP socket */
  pS->connected = UDP_FALSE;
  pS->socket_id = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if ( pS->socket_id < 0 ) {
    UDP_CheckError("MultiCast_open_as_sender::socket");
    return UDP_FALSE;
  } else {
    UDP_printf("MultiCast_open_as_sender::socket...OK.\n");
  }

  memset((char *)&pS->sock_addr, 0, sizeof(pS->sock_addr));
  pS->sock_addr.sin_family      = AF_INET;
  InetPton(AF_INET, group_address, (PVOID)&pS->sock_addr.sin_addr.s_addr);
  pS->sock_addr.sin_port        = htons(group_port);
  pS->sock_addr_len             = sizeof(pS->sock_addr);

  return UDP_TRUE;
}

#endif

