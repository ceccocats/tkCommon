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
        std::vector<linesMonocolor::line_color_t> linesBufData;

    public:
        int resolution;
        
        circle(int resolution = 32);
        void makeCircle(float x, float y, float z, float radius);
        void makeCircles(tk::math::Mat<float> points, float radius);
        void makeCircles(float* points, int n, float radius);
        void draw(glm::mat4& modelview, tk::gui::Color_t color = tk::gui::color::WHITE, float size = 1.0f);
        bool close();
};

}}}