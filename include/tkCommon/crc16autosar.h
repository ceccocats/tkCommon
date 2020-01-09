#pragma once
#include <stdint.h>
#include <iostream>


namespace tk { namespace common { namespace crc{

uint16_t crcTable16bit[256];
uint16_t polynominal16bit = 0x1021;
bool isinit = false;


inline void Crc16TableGenerator()
{
    uint16_t remainder;
    uint16_t topBit = 0x8000;
    uint32_t ui32Dividend;

    for (ui32Dividend = 0; ui32Dividend < 256; ui32Dividend++)
    {
        remainder = ui32Dividend << 8;

        for (uint8_t bit = 0; bit < 8; bit++)
        {
            if (0 == (remainder & topBit))
            {
                remainder <<= 1;
            }
            else
            {
                remainder = (remainder << 1) ^ polynominal16bit;
            }
        }

        crcTable16bit[ui32Dividend] = remainder;
    }
}

inline uint16_t CalculateCRC16(const uint8_t *crc_DataPtr, uint32_t crc_Length)
{
    uint32_t ui32Counter;
    uint8_t temp;
    uint16_t crc = 0xFFFF;

    if(isinit == false){
        Crc16TableGenerator();
        isinit = true;
    }

    for (ui32Counter = 0U; ui32Counter < crc_Length; ui32Counter++)
    {
        temp = *crc_DataPtr;
        crc = (crc << 8) ^ crcTable16bit[(uint8_t)((crc >> 8) ^ temp)];
        crc_DataPtr++;
    }

    return crc;
}

}}}