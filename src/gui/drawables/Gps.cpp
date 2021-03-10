#include "tkCommon/gui/drawables/Gps.h"

tk::gui::Gps::Gps(const std::string& name, int nPos, tk::gui::Color_t color){
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->name      = name;
    this->updateGps = true;
    circles.resize(40);
}

tk::gui::Gps::Gps(tk::data::GpsData* gps, const std::string& name, int nPos, tk::gui::Color_t color){
    this->gps       = gps;
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->name      = name;
    this->updateGps = true;
    circles.resize(40);
}

tk::gui::Gps::~Gps(){

}

void 
tk::gui::Gps::updateRef(tk::data::GpsData* gps){
    mtxUpdate.lock();
    this->gps_tmp = gps;
    update = true;
    mtxUpdate.unlock();
}

void 
tk::gui::Gps::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < circles.size(); i++){
        circles[i] = new tk::gui::shader::circle();
    }
}

void 
tk::gui::Gps::draw(tk::gui::Viewer *viewer){
    if(update){
        mtxUpdate.lock();
        update = false;
        gps = gps_tmp;
        mtxUpdate.unlock();
        updateGps = true;
    }

    if(gps == nullptr){
        return;
    }

    if(gps->isChanged(counter)){
        updateGps = true;
    }

    if(updateGps) {
        updateGps = false;

        gps->lockRead();
        this->tf = gps->header.tf;
        if(!geoConv.isInitialised() && gps->sats > 3 && gps->lat!=0 && gps->lon!=0 && gps->heigth!=0) {
            geoConv.initialiseReference(gps->lat,gps->lon,gps->heigth);
        }
        
        print.str("");
        print<<(*gps);
        if (geoConv.isInitialised())
            geoConv.geodetic2Enu(gps->lat,gps->lon,gps->heigth,&x, &y, &z);
        gps->unlockRead();
        
        if (geoConv.isInitialised()){
            lastPos = (lastPos+1) % nPos;
            circles[lastPos]->makeCircle(x,y,0.0f,gps->cov(0, 0));  
        }              
    }
        
    //Draw
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

std::string 
tk::gui::Gps::toString(){
    return name;
}