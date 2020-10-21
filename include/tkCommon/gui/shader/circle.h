#pragma once
/**
 * @file    circle.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class for draw a circle
 * @version 0.1
 * @date    2020-10-21
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"
#include "tkCommon/gui/shader/linesMonocolor.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a circle
 * 
 */
class circle : public tk::gui::shader::generic
{
    private:
        linesMonocolor*                             linesShader;
        tk::math::Mat<linesMonocolor::line_color_t> linesBufData;
        tk::gui::Buffer<float>                      linesGlBuf;

        int resolution;

    public:
        
        circle(int resolution = 32){
            linesShader      = new linesMonocolor();
            this->resolution = resolution;

            linesGlBuf.init();
            linesGlBuf.setData(NULL,resolution*3);
            linesBufData.resize(1,resolution);
        }

        void makeCircle(float x, float y, float z, float radius){
            for(int i = 0; i < resolution; i++){
                float angle = 2.0f * M_PI * (float)i/(float)resolution;
                linesBufData.atCPU(0,i).x = x + cos(angle) * radius;
                linesBufData.atCPU(0,i).y = y + sin(angle) * radius;
                linesBufData.atCPU(0,i).z = z;
            }
            linesGlBuf.setData((float*)linesBufData.data_h,resolution*3);
        }

        void makeCircles(tk::math::Mat<float> points, float radius){
            for(int i = 0; i < resolution; i++){
                float angle = 2.0f * M_PI * (float)i/(float)resolution;
                linesBufData.atCPU(0,i).x = points.atCPU(0,0) + cos(angle) * radius;
                linesBufData.atCPU(0,i).y = points.atCPU(0,1) + sin(angle) * radius;
                linesBufData.atCPU(0,i).z = points.atCPU(0,2);
            }
            linesGlBuf.setData((float*)linesBufData.data_h,resolution*3);
        }

        void draw(tk::gui::Color_t color = tk::gui::color::WHITE, float size = 1.0f){

		    linesShader->draw(&linesGlBuf, resolution, size, color, GL_LINE_LOOP);
        }

        bool close(){
            linesGlBuf.release();
            linesShader->close();
            delete linesShader;
        }
};

}}}