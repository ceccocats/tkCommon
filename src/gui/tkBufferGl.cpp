#include "tkCommon/gui/tkBufferGl.h"

namespace tk { namespace gui {

template <>
void tkBufferGl<int>::init(){
    type = GL_INT;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

template <>
void tkBufferGl<float>::init(){
    type = GL_FLOAT;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

template <>
void tkBufferGl<double>::init(){
    type = GL_DOUBLE;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

}}