#include "tkCommon/gui/shader/heightmap.h"

int tk::gui::shader::heightmap::users = 0;

tk::gui::shader::heightmap::heightmap(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/heightmap.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/heightmap.frag";

    vertexPointer.resize(2);

    map.init();
    
    shader.init(vertex, fragment, geometry);
}

void
tk::gui::shader::heightmap::close(){
    heightmap::users--;
    if(heightmap::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::heightmap::calculateIndex(int rows, int cols){

    int n = cols * rows;
    if(prec != n){
        prec = n;

        int triangles = (cols-1) * (rows-1) * 6;
        index.resize(1,triangles);
        int pos = 0;

        for(int r = 0; r < rows-1; r++){
            for(int c = 0; c < cols-1; c++){

                unsigned int vertex = r * cols + c;
                index.cpu.data[pos + 0] = vertex;
                index.cpu.data[pos + 1] = vertex + 1;
                index.cpu.data[pos + 2] = vertex + cols + 1;
                index.cpu.data[pos + 3] = vertex;
                index.cpu.data[pos + 4] = vertex + cols;
                index.cpu.data[pos + 5] = vertex + cols + 1;
                pos += 6;
            }
        }  
        tkASSERT(pos == index.size());
    }
}

void 
tk::gui::shader::heightmap::draw(glm::mat4& modelview, tk::math::Mat<float>& cords, tk::math::Mat<float>& colors, int rows, int cols){

    int n = rows * cols * 3;

    map.setData(cords.cpu.data,n);
    map.setData(colors.cpu.data,n,n);

    // 3 point
    vertexPointer[0] = {3, 3, 0};
    vertexPointer[1] = {3, 3, n};
    map.setVertexAttribs(vertexPointer);

    calculateIndex(cols,rows);
    map.setIndexVector(index.cpu.data,index.size());

    shader.use();
    shader.setMat4("modelview",modelview);

    map.use();
    int triangles = (cols-1) * (rows-1) * 2;
    glDrawElements(GL_TRIANGLES, triangles * 3, GL_UNSIGNED_INT, 0);
    map.unuse();

    shader.unuse();

    glCheckError();
}