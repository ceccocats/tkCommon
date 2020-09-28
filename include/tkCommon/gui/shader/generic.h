#pragma once

#include "tkCommon/gui/utils/CommonViewer.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief shader generic class
 * 
 */
class generic
{
    protected:
        tk::gui::Shader shader;
        glm::mat4 modelview;
        std::vector<tk::gui::vertexAttribs_t> vertexPointer;
};

}}}