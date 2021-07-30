#include "tkCommon/gui/shader/circle.h"

tk::gui::shader::circle::circle(int resolution){
    linesShader      = linesMonocolor::getInstance();
    this->resolution = resolution;
    this->isFilled   = false;

    linesGlBuf.init();
    linesGlBuf.setData(NULL,resolution*3);
    linesBufData.resize(resolution);
}

void 
tk::gui::shader::circle::makeCircle(float x, float y, float z, float radius){
    isFilled = true;
    for(int i = 0; i < resolution; i++){
        float angle = 2.0f * M_PI * (float)i/(float)resolution;
        linesBufData[i].x = x + cos(angle) * radius;
        linesBufData[i].y = y + sin(angle) * radius;
        linesBufData[i].z = z;
    }
    linesGlBuf.setData((float*)linesBufData.data(),resolution*3);
    multiCircles = 0;
}

void 
tk::gui::shader::circle::makeCircles(tk::math::Mat<float> points, float radius){
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

        linesGlBuf.setData((float*)linesBufData.data(),resolution*3,j*resolution*3);
    }
    multiCircles = points.cols();
}

void 
tk::gui::shader::circle::makeCircles(float* points, int n, float radius){
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

        linesGlBuf.setData((float*)linesBufData.data(),resolution*3,j*resolution*3);
    }
    multiCircles = n;
}

void 
tk::gui::shader::circle::draw(glm::mat4& modelview, tk::gui::Color_t color, float size){
    if(isFilled == false)
        return;
    if(multiCircles == 0){
        linesShader->draw(modelview, &linesGlBuf, resolution, size, color, GL_LINE_LOOP);
    }else{
        for(int i = 0; i < multiCircles; i++)
            linesShader->draw(modelview, &linesGlBuf, resolution, size, color, GL_LINE_LOOP, (i*(resolution*3)));
    }
}

bool 
tk::gui::shader::circle::close(){
    linesGlBuf.release();
    linesShader->close();
    return true;
}