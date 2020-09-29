#include "tkCommon/gui/utils/Buffer.h"

namespace tk { namespace gui {

template <>
void Buffer<int>::init(){
    type = GL_INT;
}

template <>
void Buffer<float>::init(){
    type = GL_FLOAT;
}

template <>
void Buffer<double>::init(){
    type = GL_DOUBLE;
}

}}