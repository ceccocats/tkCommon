#include "tkCommon/gui/drawables/Gps.h"
#include "tkCommon/gui/utils/deprecated_primitives.h"

using namespace tk::gui;

Gps::Gps(const std::string& name, tk::projection::ProjectionType prj_type, int nPos, tk::gui::Color_t color)
{
    this->color   = color;  
    this->nPos    = nPos;
    this->lastPos = -1;
    this->name    = name;
    circles.resize(MAX_POSES);

    switch (prj_type) {
    case tk::projection::ProjectionType::UTM:
        {
            proj = new tk::projection::UtmProjector();
        }
        break;
    default:
        {
            tkWRN("Unsupported projection");
        }
        break;
    }
}

Gps::Gps(tk::data::GpsData* gps, const std::string& name, tk::projection::ProjectionType prj_type, int nPos, tk::gui::Color_t color) :Gps(name,prj_type,nPos,color)
{
    this->data     = gps;  
}

void 
Gps::onInit(tk::gui::Viewer *viewer)
{
    for(int i = 0; i < circles.size(); i++){
        circles[i] = new tk::gui::shader::circle();
        circles[i]->makeCircle(0,0,0,0);  
    }
}

void 
Gps::updateData(tk::gui::Viewer *viewer)
{
    auto gps = dynamic_cast<tk::data::GpsData*>(data);

    print.str("");
    print<<(*gps);

    this->tf = gps->header.tf;

    if (!proj->hasReference()){
        proj->init(49.8955, 8.8978, gps->heigth);
    }

    // get projected values    
    tk::math::Vec3d point = proj->forward(gps->lat, gps->lon, gps->heigth);

    // apply message tf
    this->tf = tk::common::odom2tf(point.x(), point.y(), point.z(), gps->angle.z()).matrix() ;
    //this->tf = tk::common::odom2tf(point.x(), point.y(), point.z(), atan2( (double)gps->speed.y(), (double)gps->speed.x() ) ).matrix() ;

    lastPos = (lastPos+1) % nPos;
    circles[lastPos]->makeCircle(point.x(), point.y(), point.z(), gps->cov(0, 0));
}

void 
Gps::drawData(tk::gui::Viewer *viewer){
    // global position 
    glm::mat4 drw = viewer->getModelView();
    for(int i = 0; i < nPos; i++)
        circles[i]->draw(drw,color,lineSize);
    
    tk::gui::prim::drawAxis(1.0);
}

void 
Gps::imGuiSettings(){
    ImGui::ColorEdit4("Color", color.color);
    ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
    ImGui::SliderInt("Last poses gps",&nPos,1,MAX_POSES);
}

void 
Gps::imGuiInfos(){
    ImGui::Text("%s",print.str().c_str());
}

void 
Gps::onClose(){
    for(int i = 0; i < circles.size(); i++){
        circles[i]->close();
        delete circles[i];
    }
}

void 
Gps::setOrigin(double aOriginLat, double aOriginLon, double aOriginEle)
{
    if (!proj->hasReference())
        proj->init(aOriginLat, aOriginLon, aOriginEle);
}