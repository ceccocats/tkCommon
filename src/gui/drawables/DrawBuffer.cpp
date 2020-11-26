#include "tkCommon/gui/drawables/DrawBuffer.h"

tk::gui::DrawBuffer::DrawBuffer(){
    this->name = "DrawBuffer";
}

tk::gui::DrawBuffer::DrawBuffer(std::string name){
    this->name = name;
}

tk::gui::DrawBuffer::DrawBuffer(std::string name, int n, ...){
    va_list arguments; 

    va_start(arguments, n);   
    this->drawables.resize(n);

    for(int i = 0; i < n; i++){
        this->drawables[i]  = va_arg(arguments, tk::gui::Drawable*);
    }
    va_end(arguments);
    this->name = name;
}

tk::gui::DrawBuffer::~DrawBuffer(){
    for(int i = 0; i < drawables.size(); i++){
        delete drawables[i];
    }
}

void 
tk::gui::DrawBuffer::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < drawables.size(); i++){
        drawables[i]->onInit(viewer);
    }
}

void 
tk::gui::DrawBuffer::beforeDraw(tk::gui::Viewer *viewer) {
    for(int i = 0; i < drawables.size(); i++){
        drawables[i]->beforeDraw(viewer);
    }            
}


void 
tk::gui::DrawBuffer::draw(tk::gui::Viewer *viewer){
    for(int i = 0; i < drawables.size(); i++){
        glPushMatrix();
        glMultMatrixf(drawables[i]->tf.matrix().data());
        drawables[i]->draw(viewer);
        glPopMatrix();
    }
}

void 
tk::gui::DrawBuffer::imGuiInfos(){
    //for(int i = 0; i < drawables.size(); i++){
    //    drawables[i]->imGuiInfos();
    //}
}

void 
tk::gui::DrawBuffer::imGuiSettings(){
    //for(int i = 0; i < drawables.size(); i++){
    //    drawables[i]->imGuiSettings();
    //}
}

void 
tk::gui::DrawBuffer::onClose(){
    for(int i = 0; i < drawables.size(); i++){
        drawables[i]->onClose();
    }
}

std::string 
tk::gui::DrawBuffer::toString(){
    return name;
}