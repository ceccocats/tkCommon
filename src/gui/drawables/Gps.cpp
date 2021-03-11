#include "tkCommon/gui/drawables/Gps.h"

tk::gui::Gps::Gps(const std::string& name, int nPos, tk::gui::Color_t color){
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->name      = name;
    circles.resize(40);
}

tk::gui::Gps::Gps(tk::data::GpsData* gps, const std::string& name, int nPos, tk::gui::Color_t color){
    this->data      = gps;
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->name      = name;
    circles.resize(40);
}

tk::gui::Gps::~Gps(){

}

void 
tk::gui::Gps::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < circles.size(); i++){
        circles[i] = new tk::gui::shader::circle();
    }
}

void 
tk::gui::Gps::updateData(tk::gui::Viewer *viewer){
    this->tf = data->header.tf;
    if(!geoConv.isInitialised() && data->sats > 3 && data->lat!=0 && data->lon!=0 && data->heigth!=0) {
        geoConv.initialiseReference(data->lat,data->lon,data->heigth);
    }
    
    print.str("");
    print<<(*data);
    if (geoConv.isInitialised())
        geoConv.geodetic2Enu(data->lat,data->lon,data->heigth,&x, &y, &z);
    z = 0.0f; //non using z

    if (geoConv.isInitialised()){
        lastPos = (lastPos+1) % nPos;
        circles[lastPos]->makeCircle(x,y,z,data->cov(0, 0));  
    } 
}

void 
tk::gui::Gps::drawData(tk::gui::Viewer *viewer){
    if (geoConv.isInitialised()){
        for(int i = 0; i < nPos; i++){
            circles[i]->draw(drwModelView,color,lineSize);
        }   	
    }
}

void 
tk::gui::Gps::imGuiSettings(){
    ImGui::ColorEdit4("Color", color.color);
    ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
    ImGui::SliderInt("Last poses gps",&nPos,1,40);
}

void 
tk::gui::Gps::imGuiInfos(){
    ImGui::Text("%s",print.str().c_str());
}

void 
tk::gui::Gps::onClose(){
    for(int i = 0; i < circles.size(); i++){
        circles[i]->close();
        delete circles[i];
    }
}