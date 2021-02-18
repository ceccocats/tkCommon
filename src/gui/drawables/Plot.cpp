#include "tkCommon/gui/drawables/Plot.h"

void 
tk::gui::Plot::updateLimits() {
    float x = points.head(0).x;
    float y = points.head(0).y;
    float z = points.head(0).z;

    if(points.size() == 1) {
        mins = { x, y, z };
        maxs = { x, y, z };
    }
    if(x < mins.x()) mins.x() = x;
    if(y < mins.y()) mins.y() = y;
    if(z < mins.z()) mins.z() = z;
    if(x > maxs.x()) maxs.x() = x;
    if(y > maxs.y()) maxs.y() = y;
    if(z > maxs.z()) maxs.z() = z;
}

void 
tk::gui::Plot::updateDatas(){
    if(this->type == type_t::LINE){
        int pos = points.position % points.dim;
        glData.setData((float*)(points.array+pos),(nPoints-pos)*3,0);
        glData.setData((float*)points.array,pos*3,(nPoints-pos)*3);
        return;
    }
    if(this->type == type_t::CIRCLES){
        shaderCircle->makeCircles((float*)points.array,nPoints,circleRay);
        return;
    }
    if(this->type == type_t::AXIS){
        glData.setData((float*)points.array,nPoints*3,0);
        glData.setData((float*)rots.array,nPoints*3,nPoints*3);
        return;
    }
}


tk::gui::Plot::Plot(std::string name, int maxDataDim, type_t type, float drawSize){
    tkASSERT(drawSize >= 1.0);

    this->name = name;
    points.setDim(maxDataDim);
    rots.setDim(maxDataDim);
    this->type = type;

    circleRay = drawSize;
    circleRes = 32;
    lineW     = drawSize;
    pointW    = drawSize;
    poseW     = drawSize;

    this->drawSize = drawSize;

    color     = tk::gui::color::WHITE;

    menu.resize(3);
    menu[0] = item0.c_str();
    menu[1] = item1.c_str();
    menu[2] = item2.c_str();

    value = maxDataDim;
}

tk::gui::Plot::~Plot(){

}

void 
tk::gui::Plot::addPoint(tk::common::Tfpose tf) {
    tk::common::Vector3<float> rot   = tk::common::tf2rot(tf);
    tk::common::Vector3<float> point = tk::common::tf2pose(tf);
    points.add(point);
    rots.add(rot);

    if(nPoints < points.dim){
        nPoints++;
    }
    if(nPoints > points.dim){
        nPoints = points.dim;
    }
    newPoint = true;
    updateLimits();
}
void 
tk::gui::Plot::addPoint(tk::math::Vec3<float> pt) {
    addPoint(tk::common::odom2tf(pt.x(), pt.y(), pt.z(), 0,0,0));
}

void 
tk::gui::Plot::onInit(tk::gui::Viewer *viewer){
    shaderLine    = new tk::gui::shader::linesMonocolor();
    shaderAxis    = new tk::gui::shader::axisPlot();
    shaderCircle  = new tk::gui::shader::circle();
    glData.init();
}

void 
tk::gui::Plot::draw(tk::gui::Viewer *viewer){
    if(newPoint == true){
        newPoint = false;
        updateDatas();
    }
    
    if(this->type == type_t::LINE){
        shaderLine->draw(drawview,&glData,nPoints,drawSize,color,GL_LINE_STRIP);
        return;
    }

    if(this->type == type_t::CIRCLES){
        shaderCircle->draw(drawview,color,drawSize);
        return;
    }

    if(this->type == type_t::AXIS){
        shaderAxis->draw(drawview,&glData,nPoints,drawSize);
        return;
    }
}

void 
tk::gui::Plot::imGuiInfos(){
}

void 
tk::gui::Plot::imGuiSettings(){
    if(ImGui::SliderInt("Dim",&value,1, 100000)){
        points.setDim(value);
    }
    if(ImGui::Combo("Draw mode", (int*)&type, menu.data(), menu.size())){
        updateDatas();
    }
    ImGui::SliderFloat("Size",&drawSize,1.0f,20.0f,"%.1f");
    ImGui::ColorEdit3("Color", color.color);

    if(this->type == type_t::CIRCLES){
        ImGui::SliderFloat("Radious",&circleRay,1.0f,60.0f,"%.1f");
        return;
    }
}

void 
tk::gui::Plot::onClose(){
    glData.release();
    shaderLine->close();
    delete shaderLine;
    shaderCircle->close();
    delete shaderCircle; 
    shaderAxis->close();
    delete shaderAxis;
}

std::string 
tk::gui::Plot::toString(){
    return name;
}