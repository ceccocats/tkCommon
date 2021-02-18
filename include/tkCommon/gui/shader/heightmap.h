#pragma once
/**
 * @file    stripline.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw a heightmap
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a heightmap
 * 
 */
class heightmap : public tk::gui::shader::generic
{
    private:
        tk::gui::Buffer<float> map;
        tk::math::Mat<unsigned int> index;
        int prec = 0;
        
        void calculateIndex(int rows, int cols){

            int n = cols * rows;
            if(prec != n){
                prec = n;

                int triangles = (cols-1) * (rows-1) * 6;
                index.resize(1,triangles);
                int pos = 0;

                for(int r = 0; r < rows-1; r++){
                    for(int c = 0; c < cols-1; c++){

                        unsigned int vertex = r * cols + c;

                        index.data_h[pos] = vertex;
                        pos++;

                        index.data_h[pos] = vertex + 1;
                        pos++;

                        index.data_h[pos] = vertex + cols + 1;
                        pos++;

                        index.data_h[pos] = vertex;
                        pos++;

                        index.data_h[pos] = vertex + cols;
                        pos++;

                        index.data_h[pos] = vertex + cols + 1;
                        pos++;
                    }
                }  
                tkASSERT(pos == index.size());
            }
        }

    public:
        bool init(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/heightmap.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/heightmap.frag";

            vertexPointer.resize(2);

            map.init();
            
            return shader.init(vertex, fragment, geometry);
        }

        void draw(glm::mat4& modelview, tk::math::Mat<float>& cords, tk::math::Mat<float>& colors, int rows, int cols){

            int n = rows * cols * 3;

            map.setData(cords.data_h,n);
            map.setData(colors.data_h,n,n);

            // 3 point
            vertexPointer[0] = {3, 3, 0};
            vertexPointer[1] = {3, 3, n};
            map.setVertexAttribs(vertexPointer);

            calculateIndex(cols,rows);
            map.setIndexVector(index.data_h,index.size());

            shader.use();
            shader.setMat4("modelview",modelview);

            map.use();
            int triangles = (cols-1) * (rows-1) * 2;
            glDrawElements(GL_TRIANGLES, triangles * 3, GL_UNSIGNED_INT, 0);
            map.unuse();

            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}