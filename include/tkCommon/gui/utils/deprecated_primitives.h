#pragma once

namespace tk { namespace gui { namespace prim {

void drawAxis(float s, float lw = 2.5);
void drawTexture(uint32_t tex, float sx, float sy);
void genTexture(uint8_t *data, int width, int height, GLuint &tex);


}}} // namespace name
