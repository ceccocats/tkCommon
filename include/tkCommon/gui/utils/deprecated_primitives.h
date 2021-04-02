#pragma once
#include <opencv2/core.hpp>
#include "tkCommon/common.h"

namespace tk { namespace gui { namespace prim {

void drawAxis(float s, float lw = 2.5);
void drawTexture(uint32_t tex, float sx, float sy);
void genTexture(cv::Mat img, GLuint &tex);
void drawCircle(tk::common::Vector3<float> pose, float r, int res=32, bool filled=false);
void drawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size = {1,1,1}, bool filled=false);



}}} // namespace name
