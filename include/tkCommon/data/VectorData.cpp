#include "tkCommon/data/VectorData.h"
#include "tkCommon/data/ImageData.h"

namespace tk { namespace data {
    template <class T>
    const DataType VectorData<T>::type = DataType::VECTOR;

    template <>
    const DataType VectorData<ImageData>::type = DataType::VECTOR;
}}
