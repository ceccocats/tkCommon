#include "tkCommon/gui/drawables/Gps.h"

tk::gui::Gps::Gps(const std::string& name, int nPos, tk::gui::Color_t color){
    this->color   = color;  
    this->nPos    = nPos;
    this->lastPos = -1;
    this->name    = name;
    circles.resize(MAX_POSES);
}

tk::gui::Gps::Gps(tk::data::GpsData* gps, const std::string& name, int nPos, tk::gui::Color_t color) :Gps(name,nPos,color){
    this->data     = gps;  
}

tk::gui::Gps::~Gps(){

}

void 
tk::gui::Gps::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < circles.size(); i++){
        circles[i] = new tk::gui::shader::circle();
        circles[i]->makeCircle(0,0,0,0);  
    }
}

void 
tk::gui::Gps::updateData(tk::gui::Viewer *viewer){

    tk::data::GpsData* gps = (tk::data::GpsData*)data;

    this->tf = gps->header.tf;
    if(!geoConv.isInitialised() && gps->sats > 3 && gps->lat!=0 && gps->lon!=0 && gps->heigth!=0) {
        geoConv.initialiseReference(gps->lat,gps->lon,gps->heigth);
    }
    
    print.str("");
    print<<(*gps);
    if (geoConv.isInitialised())
        geoConv.geodetic2Enu(gps->lat,gps->lon,gps->heigth,&x, &y, &z);
    z = 0.0f; //non using z

    if (geoConv.isInitialised()){
        lastPos = (lastPos+1) % nPos;
        circles[lastPos]->makeCircle(x,y,z,gps->cov(0, 0));  
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
    ImGui::SliderInt("Last poses gps",&nPos,1,MAX_POSES);
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