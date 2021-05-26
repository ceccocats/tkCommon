#include "tkCommon/data/ImageData.h"

namespace tk { namespace data {

    template <>
    const DataType ImageData_gen<uint8_t>::type = DataType::IMAGEU8;
    template <>
    const DataType ImageData_gen<float>::type = DataType::IMAGEF;
    template <>
    const DataType ImageData_gen<uint16_t>::type = DataType::IMAGEU16;
}}
