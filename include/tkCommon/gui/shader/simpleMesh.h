#pragma once
#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

class simpleMesh : public tk::gui::shader::generic
{
    private:
        glm::vec4 pointColor;
        int users;

        simpleMesh();
    public:
        static simpleMesh* getInstance() {
            static simpleMesh instance;
            instance.users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3 lightPos, bool useLight, tk::gui::Color_t color = tk::gui::color::WHITE);
        void draw(glm::mat4& modelview, const std::vector<tk::gui::Buffer<float>> &vectorBuffer, const std::vector<glm::mat4> &poses, glm::vec3 lightPos, bool useLight, tk::gui::Color_t color = tk::gui::color::WHITE);
        void close();
};
}}}