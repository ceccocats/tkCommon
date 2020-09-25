#include "tkCommon/gui/Buffer.h"

namespace tk { namespace gui {

template <>
void Buffer<int>::init(){
    type = GL_INT;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

template <>
void Buffer<float>::init(){
    type = GL_FLOAT;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

template <>
void Buffer<double>::init(){
    type = GL_DOUBLE;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

}}