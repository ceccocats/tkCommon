#include "tkCommon/gui/utils/Buffer.h"

namespace tk { namespace gui {

template <>
void Buffer<int>::init(bool staticDraw){
    type = GL_INT;
    memType = staticDraw ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW ;
}

template <>
void Buffer<float>::init(bool staticDraw){
    type = GL_FLOAT;
    memType = staticDraw ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW ;
}

template <>
void Buffer<double>::init(bool staticDraw){
    type = GL_DOUBLE;
    memType = staticDraw ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW ;
}

}}