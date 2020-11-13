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
#include "tkCommon/math/Vec.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a circle
 * 
 */
class circle : public tk::gui::shader::generic
{
    private:
        bool isFilled;
        int  multiCircles;
        linesMonocolor* linesShader;
        tk::gui::Buffer<float> linesGlBuf;
        tk::math::Vec<linesMonocolor::line_color_t> linesBufData;

    public:
        int resolution;
        
        circle(int resolution = 32){
            linesShader      = new linesMonocolor();
            this->resolution = resolution;
            this->isFilled   = false;

            linesGlBuf.init();
            linesGlBuf.setData(NULL,resolution*3);
            linesBufData.resize(resolution);
        }

        void makeCircle(float x, float y, float z, float radius){
            isFilled = true;
            for(int i = 0; i < resolution; i++){
                float angle = 2.0f * M_PI * (float)i/(float)resolution;
                linesBufData[i].x = x + cos(angle) * radius;
                linesBufData[i].y = y + sin(angle) * radius;
                linesBufData[i].z = z;
            }
            linesGlBuf.setData((float*)linesBufData.data_h,resolution*3);
            multiCircles = 0;
        }

        void makeCircles(tk::math::Mat<float> points, float radius){
            isFilled = true;
            for(int j = 0; j < points.cols(); j++){
                float x = points(0,j);
                float y = points(1,j);
                float z = points(2,j);
                for(int i = 0; i < resolution-1; i++){
                    float angle = 2.0f * M_PI * (float)i/(float)resolution;
                    linesBufData[i].x = x + cos(angle) * radius;
                    linesBufData[i].y = y + sin(angle) * radius;
                    linesBufData[i].z = z;
                }
                linesBufData[resolution-1].x = x + cos(0) * radius;
                linesBufData[resolution-1].y = y + sin(0) * radius;
                linesBufData[resolution-1].z = z;

                linesGlBuf.setData((float*)linesBufData.data_h,resolution*3,j*resolution*3);
            }
            multiCircles = points.cols();
        }

        void makeCircles(float* points, int n, float radius){
            isFilled = true;
            for(int j = 0; j < n; j++){
                float x = points[j*3 + 0];
                float y = points[j*3 + 1];
                float z = points[j*3 + 2];
                for(int i = 0; i < resolution-1; i++){
                    float angle = 2.0f * M_PI * (float)i/(float)resolution;
                    linesBufData[i].x = x + cos(angle) * radius;
                    linesBufData[i].y = y + sin(angle) * radius;
                    linesBufData[i].z = z;
                }
                linesBufData[resolution-1].x = x + cos(0) * radius;
                linesBufData[resolution-1].y = y + sin(0) * radius;
                linesBufData[resolution-1].z = z;

                linesGlBuf.setData((float*)linesBufData.data_h,resolution*3,j*resolution*3);
            }
            multiCircles = n;
        }

        void draw(tk::gui::Color_t color = tk::gui::color::WHITE, float size = 1.0f){
            if(isFilled == false)
                return;
            if(multiCircles == 0){
		        linesShader->draw(&linesGlBuf, resolution, size, color, GL_LINE_LOOP);
            }else{
                for(int i = 0; i < multiCircles; i++)
                    linesShader->draw(&linesGlBuf, resolution, size, color, GL_LINE_LOOP, (i*(resolution*3)));
            }
        }

        bool close(){
            linesGlBuf.release();
            linesShader->close();
            delete linesShader;
        }
};

}}}