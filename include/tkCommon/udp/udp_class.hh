
/* ============================================================================
 UDP communication with limited packed size
 ============================================================================ */

#ifndef __UDP_CLASS_HH
#define __UDP_CLASS_HH

#include "udp_C_class.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iostream>
#include <exception>

#if __cplusplus >= 201103L
  #define NORETURN [[ noreturn ]]
#else
  #define NORETURN
#endif

class InterruptException : public std::exception {
public:
  InterruptException( char const s[] ) : S(s) {}
  InterruptException( std::string const & s ) : S(s) {}
  std::string S;
};

class Socket {

  bool       server_is_running;
  SocketData data;

  #ifndef _MSC_VER
  NORETURN
  static
  void
  sig_to_exception( int s ) {
    std::stringstream ss;
    ss << "signal number " << s << " found";
    throw InterruptException(ss.str());
  }
  #endif

public:

  Socket();

  void
  open_as_client( char const addr[], int port, bool conn )
  { Socket_open_as_client( &data, addr, port, (conn?UDP_TRUE:UDP_FALSE) ); }

  void
  open_as_server( int port )
  { Socket_open_as_server( &data, port ); }

#ifndef UDP_NO_MULTICAST_SUPPORT

  int
  open_multicast_as_client(
    char const group_address[],
    int        group_port
  ) {
    return MultiCast_open_as_sender(
      &data, group_address, group_port
    );
  }

  int
  open_multicast_as_listener(
    char const group_address[],
    int        group_port
  ) {
    return MultiCast_open_as_listener(
      &data, group_address, group_port
    );
  }

#endif

  void
  server_start()
  { server_is_running = true; }

  void
  server_stop()
  { server_is_running = false; }

  bool
  server_running() const
  { return server_is_running; }

  void
  set_timeout_ms( uint64_t tout_ms )
  { data.timeout_ms = tout_ms; }

  bool
  close() {
    int ok = Socket_close( &data );
    return ok == UDP_TRUE;
  }

  void
  check() {
    if ( data.socket_id == -1 ) {
      exit(-1);
    } else {
      std::cerr << "Opened socket id = " << data.socket_id << '\n';
    }
  }

  /* Send message function */
  int
  send(
    int32_t  buffer_id,
    uint8_t  buffer[],
    uint32_t buffer_size
  ) {
    return Socket_send( &data, buffer_id, buffer, buffer_size );
  }

  /* Receive message function */
  int
  receive(
    int32_t & buffer_id,
    int32_t & buffer_len,
    uint8_t   buffer[],
    uint32_t  buffer_size,
    uint64_t  start_time
  ) {
    return Socket_receive( &data,
                           &buffer_id,
                           &buffer_len,
                           buffer,
                           buffer_size,
                           start_time );
  }

};

#endif
