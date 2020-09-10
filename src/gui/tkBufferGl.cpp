#include "tkCommon/gui/tkBufferGl.h"

namespace tk { namespace gui {
template <>
void tkBufferGl<int>::setVertexAttrib(){

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    //check if vertexAttribs is set
    if(vertexAttribs.size() == 0){
        clsErr("You need to fill vertexAttribsPointer\n");
        return;
    }

    for(int i = 0; i < vertexAttribs.size(); i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, vertexAttribs[i].vertexSize, 
            GL_INT, GL_FALSE, vertexAttribs[i].dataSize * sizeof(int), (void*)(vertexAttribs[i].offset * sizeof(int)));
    }
    glBindVertexArray(0);
}

template <>
void tkBufferGl<float>::setVertexAttrib(){

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    //check if vertexAttribs is set
    if(vertexAttribs.size() == 0){
        clsErr("You need to fill vertexAttribsPointer\n");
        return;
    }

    for(int i = 0; i < vertexAttribs.size(); i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, vertexAttribs[i].vertexSize, 
            GL_FLOAT, GL_FALSE, vertexAttribs[i].dataSize * sizeof(float), (void*)(vertexAttribs[i].offset * sizeof(float)));
    }
    glBindVertexArray(0);
}

template <>
void tkBufferGl<double>::setVertexAttrib(){

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    //check if vertexAttribs is set
    if(vertexAttribs.size() == 0){
        clsErr("You need to fill vertexAttribsPointer\n");
        return;
    }

    for(int i = 0; i < vertexAttribs.size(); i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, vertexAttribs[i].vertexSize, 
            GL_DOUBLE, GL_FALSE, vertexAttribs[i].dataSize * sizeof(double), (void*)(vertexAttribs[i].offset * sizeof(double)));
    }
    glBindVertexArray(0);
}

}}