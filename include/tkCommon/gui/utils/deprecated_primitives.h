#pragma once
#include <opencv2/core.hpp>

namespace tk { namespace gui { namespace prim {

void drawAxis(float s, float lw = 2.5);
void drawTexture(uint32_t tex, float sx, float sy);
void genTexture(cv::Mat img, GLuint &tex);


}}} // namespace name
