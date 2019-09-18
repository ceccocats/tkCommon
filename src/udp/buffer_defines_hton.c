#include "tkCommon/udp/buffer_defines.h"

#ifdef UDP_ON_WINDOWS
  #include <Winsock.h>
#else
  #include <arpa/inet.h>
#endif

static
uint64_t
htonll_local( uint64_t n ) {
  uint64_t lo, hi;
  if ( 1 == htonl(1) ) return n;
  hi = (uint64_t)htonl( *((uint32_t*)&n) );
  n = n>>32;
  lo = (uint64_t)htonl( *((uint32_t*)&n) );
  return (hi << 32) + lo;
}

/* -------------------------------------------------- */

uint32_t
int8_to_buffer( int8_t in, uint8_t buffer[1] ) {
  buffer[0] = (uint8_t)in;
  return sizeof(int8_t);
}

uint32_t
uint8_to_buffer( uint8_t in, uint8_t buffer[1] ) {
  buffer[0] = in;
  return sizeof(uint8_t);
}

uint32_t
int16_to_buffer( int16_t in, uint8_t buffer[2] ) {
  uint16_t tmp = htons( (uint16_t) in );
  memcpy( buffer, &tmp, sizeof(int16_t) );
  return sizeof(int16_t);
}

uint32_t
uint16_to_buffer( uint16_t in, uint8_t buffer[2] ) {
  uint16_t tmp = htons( in );
  memcpy( buffer, &tmp, sizeof(uint16_t)  );
  return sizeof(uint16_t);
}

uint32_t
int32_to_buffer( int32_t in, uint8_t buffer[4] ) {
  uint32_t tmp = htonl( (uint32_t) in );
  memcpy( buffer, &tmp, sizeof(int32_t) );
  return sizeof(int32_t);
}

uint32_t
uint32_to_buffer( uint32_t in, uint8_t buffer[4] ) {
  uint32_t tmp = htonl( in );
  memcpy( buffer, &tmp, sizeof(uint32_t) );
  return sizeof(uint32_t);
}

uint32_t
int64_to_buffer( int64_t in, uint8_t buffer[8] ) {
  uint64_t tmp = htonll_local( (uint64_t) in );
  memcpy( buffer, &tmp, sizeof(int64_t) );
  return sizeof(int64_t);
}

uint32_t
uint64_to_buffer( uint64_t in, uint8_t buffer[8] ) {
  uint64_t tmp = htonll_local( in );
  memcpy( buffer, &tmp, sizeof(uint64_t)  );
  return sizeof(uint64_t);
}

#ifdef PACK_FLOAT

  static
  uint64_t
  maskOfBits( uint32_t NBITS ) {
    uint64_t one = 1;
    return (one<<NBITS)-one;
  }

  static
  uint64_t
  pack754( double f, uint32_t bits, uint32_t expbits ) {
    double   fnorm;
    uint64_t sign, exp, significand, res, zero, one;
    uint32_t significandbits = bits - expbits - 1; /* -1 for sign bit */
    zero = 0;
    one  = 1;
    /* special case NaN and INF */
    if ( f != f ) return ~zero; /* NaN */
    if ( f*0.0 != 0.0 ) return maskOfBits(expbits)<<significandbits; /* INF */
    if ( f == 0.0 ) return zero; /* get this special case out of the way */
    /* check sign and begin normalization */
    fnorm = f;
    sign  = 0;
    if ( f < 0 ) { sign = one<<(bits-1); fnorm = -f; }

    /* get the normalized form of f and track the exponent
       get the biased exponent */
    exp = maskOfBits(expbits-1); /* bias */
    while ( fnorm >= 2.0 ) { fnorm /= 2.0; ++exp; }
    while ( fnorm <  1.0 ) { fnorm *= 2.0; --exp; }
    fnorm -= 1.0;
    /* calculate the binary form (non-float) of the significand data */
    significand = fnorm * ((one<<significandbits) + 0.5);
    /* return the final answer */
    res = sign | (exp<<significandbits) | significand;
    return res;
  }

  uint32_t
  float_to_buffer( float in, uint8_t buffer[4] ) {
    uint64_t res64 = pack754( in, 32, 8 );
    uint32_t res32 = (uint32_t)res64;
    uint32_to_buffer( res32, buffer );
    return sizeof(float);
  }

  uint32_t
  double_to_buffer( double in, uint8_t buffer[8] ) {
    uint64_t res64 = pack754( in, 64, 11 );
    uint32_t res32 = (uint32_t)res64;
    uint32_to_buffer( res32, buffer );
    return sizeof(double);
  }

#else

  uint32_t
  float_to_buffer( float in, uint8_t buffer[4] ) {
    union FloatInt {
      float    f;
      uint32_t i;
    } tmp;
    tmp.f = in;
    return uint32_to_buffer( tmp.i, buffer );
  }

  uint32_t
  double_to_buffer( double in, uint8_t buffer[8] ) {
    union DoubleInt {
      double   d;
      uint64_t i;
    } tmp;
    tmp.d = in;
    return uint64_to_buffer( tmp.i, buffer );
  }

#endif
