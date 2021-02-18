#include "tkCommon/gui/drawables/Gps.h"

tk::gui::Gps::Gps(const std::string& name, int nPos, tk::gui::Color_t color){
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->initted   = false;
    this->name      = name;
    circles.resize(40);
}

tk::gui::Gps::Gps(tk::data::GpsData* gps, const std::string& name, int nPos, tk::gui::Color_t color){
    this->gps       = gps;
    this->color     = color;  
    this->nPos      = nPos;
    this->lastPos   = -1;
    this->initted   = true;
    this->name      = name;
    circles.resize(40);
}

tk::gui::Gps::~Gps(){

}

void 
tk::gui::Gps::updateRef(tk::data::GpsData* gps){
    this->gps = gps;   
    initted = update = true;
}

void 
tk::gui::Gps::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < circles.size(); i++){
        circles[i] = new tk::gui::shader::circle();
    }
}

void 
tk::gui::Gps::draw(tk::gui::Viewer *viewer){
    if(initted == true){
        if(gps->isChanged(counter) || update){
            update = false;

            gps->lockRead();
            if(!geoConv.isInitialised() && gps->sats > 3 && gps->lat!=0 && gps->lon!=0 && gps->heigth!=0) {
                geoConv.initialiseReference(gps->lat,gps->lon,gps->heigth);
            }
            
            print.str("");
            print<<(*gps);
            double x, y, z;
            if (geoConv.isInitialised())
                geoConv.geodetic2Enu(gps->lat,gps->lon,gps->heigth,&x, &y, &z);
            gps->unlockRead();
            
            float RAGGIO = 2.0f; //TODO: BOSI
            lastPos = (lastPos+1) % nPos;
            circles[lastPos]->makeCircle(x,y,z,RAGGIO);                
        }

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