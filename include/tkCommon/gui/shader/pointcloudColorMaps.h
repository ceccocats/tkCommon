#pragma once
/**
 * @file    pointcloudColorMaps.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw pointcloud using an axis or feature in colormaps
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"
#include <dirent.h>

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z 1 X Y Z 1...]
 * 
 */
class pointcloudColorMaps  : public tk::gui::shader::generic
{
    private:
        std::map<std::string,tk::gui::Shader*> shaders;
        static int users;

        pointcloudColorMaps();

    public:
        std::vector<std::string> colormaps;

        static pointcloudColorMaps* getInstance(){
            static pointcloudColorMaps instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, std::string name, tk::gui::Buffer<float>* buffer, 
            int nPoints, float minValue, float maxValue, int axis = -1, float alpha = 1.0);
        void close();
};

}}}