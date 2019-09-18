#include "tkCommon/udp/buffer_defines.h"

#ifdef UDP_ON_WINDOWS
  #include <Winsock.h>
#else
  #include <arpa/inet.h>
#endif

static
uint64_t
ntohll_local( uint64_t n ) {
  uint64_t lo, hi;
  if ( 1 == ntohl(1) ) return n;
  hi = (uint64_t)ntohl( *((uint32_t*)&n) );
  n = n>>32;
  lo = (uint64_t)ntohl( *((uint32_t*)&n) );
  return (hi << 32) + lo;
}

/* ---------------------------------------------------------------------------- */

uint32_t
buffer_to_uint8( uint8_t const buffer[1], uint8_t * out ) {
  *out = buffer[0];
  return sizeof(uint8_t);
}

uint32_t
buffer_to_int8( uint8_t const buffer[1], int8_t * out ) {
  *((uint8_t*)out) = buffer[0];
  return sizeof(int8_t);
}

uint32_t
buffer_to_uint16( uint8_t const buffer[2], uint16_t * out ) {
  uint16_t tmp;
  memcpy( &tmp, buffer, sizeof(uint16_t) );
  *out = ntohs( tmp );
  return sizeof(uint16_t);
}

uint32_t
buffer_to_int16( uint8_t const buffer[2], int16_t * out ) {
  uint16_t tmp;
  memcpy( &tmp, buffer, sizeof(int16_t) );
  *((uint16_t*)out) = ntohs( tmp );
  return sizeof(int16_t);
}

uint32_t
buffer_to_uint32( uint8_t const buffer[4], uint32_t * out ) {
  uint32_t tmp;
  memcpy( &tmp, buffer, sizeof(uint32_t) );
  *out = ntohl( tmp );
  return sizeof(uint32_t);
}

uint32_t
buffer_to_int32( uint8_t const buffer[4], int32_t * out ) {
  uint32_t tmp;
  memcpy( &tmp, buffer, sizeof(int32_t) );
  *((uint32_t*)out) = ntohl( tmp );
  return sizeof(int32_t);
}

uint32_t
buffer_to_uint64( uint8_t const buffer[8], uint64_t * out ) {
  uint64_t tmp;
  memcpy( &tmp, buffer, sizeof(uint64_t) );
  *out = ntohll_local( tmp );
  return sizeof(uint64_t);
}

uint32_t
buffer_to_int64( uint8_t const buffer[8], int64_t * out ) {
  uint64_t tmp;
  memcpy( &tmp, buffer, sizeof(int64_t) );
  *((uint64_t*)out) = ntohll_local( tmp );
  return sizeof(int64_t);
}

#ifdef PACK_FLOAT

  static
  uint64_t
  maskOfBits( uint32_t NBITS ) {
    uint64_t one = 1;
    return (one<<NBITS)-one;
  }

  static
  double
  unpack754( uint64_t i, uint32_t bits, uint32_t expbits ) {
    double res;
    uint32_t significandbits = bits - expbits - 1; /* -1 for sign bit */
    if ( i == 0 ) return 0;
    /* pull the significand */
    res = (double)(i & maskOfBits(bits-expbits-1));
    res /= (double)(1<<significandbits); /* mask */
    res += 1.0; /* add the one back on */
    /* deal with the exponent */
    uint64_t bias  = maskOfBits(expbits-1);
    uint64_t shift = (i>>significandbits) & maskOfBits(expbits);
    while ( shift > bias ) { res *= 2.0; --shift; }
    while ( shift < bias ) { res /= 2.0; ++shift; }
    /* sign it */
    if ( i & (((uint64_t)1)<<(bits-1)) ) res = -res;
    return res;
  }

  uint32_t
  buffer_to_float( uint8_t const buffer[4], float *out) {
    uint32_t tmp32;
    buffer_to_uint32( buffer, &tmp32 );
    uint64_t tmp64 = (uint64_t)tmp32;
    double tmpd = unpack754( tmp64, 32, 8 );
    *out = (float)tmpd;
    return sizeof(float);
  }

  uint32_t
  buffer_to_double( uint8_t const buffer[8], double *out ) {
    uint64_t tmp64;
    buffer_to_uint64( buffer, &tmp64 );
    *out = unpack754( tmp64, 32, 8 );
    return sizeof(double);
  }

#else

  uint32_t
  buffer_to_float( uint8_t const buffer[8], float *out) {
    union FloatInt {
      float    f;
      uint32_t i;
    } tmp;
    uint32_t res = buffer_to_uint32( buffer, &tmp.i );
    *out = tmp.f;
    return res;
  }

  uint32_t
  buffer_to_double( uint8_t const buffer[8], double *out ) {
    union DoubleInt {
      double   d;
      uint64_t i;
    } tmp;
    uint32_t res = buffer_to_uint64( buffer, &tmp.i );
    *out = tmp.d;
    return res;
  }

#endif
