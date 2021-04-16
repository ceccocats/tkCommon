#include "tkCommon/gui/drawables/PolyLine.h"

tk::gui::PolyLine::PolyLine(tk::gui::line* line, std::string name, tk::gui::Color_t col){
    update      = true;
    this->line  = line;
    this->name = name;
    this->color = col;
    glData.init();
}

tk::gui::PolyLine::~PolyLine(){

}

/*
void 
tk::gui::PolyLine::updateRef(tk::gui::line* line){
    this->line = line;   
    update = true;
}
*/

void 
tk::gui::PolyLine::onInit(tk::gui::Viewer *viewer){
    shader = tk::gui::shader::linesMonocolor::getInstance();
}

void 
tk::gui::PolyLine::draw(tk::gui::Viewer *viewer){
    if(line->isChanged(counter) || update){
        update = false;
        
        line->lockRead();
        glData.setData((float*)line->points.data(),line->points.size()*3);
        line->unlockRead();               
    }

    auto shaderLine = (tk::gui::shader::linesMonocolor*)shader;
    shaderLine->draw(drwModelView,&glData,glData.size()/3,lineSize,color,GL_LINE_STRIP);	
}

void 
tk::gui::PolyLine::imGuiSettings(){
    ImGui::ColorEdit4("Color", color.color);
    ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
}

void 
tk::gui::PolyLine::imGuiInfos(){
    ImGui::Text("%s","TODO: in futuro stampare numero punti");
}

void 
tk::gui::PolyLine::onClose(){
    auto shaderLine = (tk::gui::shader::linesMonocolor*)shader;
    shaderLine->close();
    delete shaderLine;
    glData.release();
}
