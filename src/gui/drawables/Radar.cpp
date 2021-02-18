#include "tkCommon/gui/drawables/Radar.h"

namespace tk { namespace gui {
    Radar::Radar()
    {
        
    }

    Radar::Radar(const std::string& name)
    {
        far_drw     = new tk::gui::Cloud4f("far");
        near_drw    = new tk::gui::Cloud4f("near");
        this->name  = name;
    }

    Radar::Radar(tk::data::RadarData* radar, const std::string& name)
    {
        far_drw     = new tk::gui::Cloud4f("far");
        near_drw    = new tk::gui::Cloud4f("near");
        this->radar = radar;
        this->name  = name;
    }
    
    Radar::~Radar()
    {

    }

    void 
    Radar::updateRef(tk::data::RadarData* radar)
    {
        this->radar = radar;   
        far_drw->updateRef(&radar->far);
        near_drw->updateRef(&radar->near);
        initted = update = true;
    }

    void 
    Radar::onInit(tk::gui::Viewer *viewer)
    {
        far_drw->pointSize  = 5.0f;
        near_drw->pointSize = 5.0f;
        
        viewer->add(far_drw);
        viewer->add(near_drw);
    }

    void 
    Radar::draw(tk::gui::Viewer *viewer)
    {
        if (initted) {
            if(radar->isChanged(counter) || update) {
                update = false;

                radar->lockRead();
                print.str("");
                print<<(*radar); 
                radar->unlockRead();
            }     
        }
    }

    void 
    Radar::imGuiSettings()
    {
        ImGui::BeginGroup();
        ImGui::BeginChild("FAR", ImVec2{-1, ImGui::GetContentRegionAvail().y/2}, true);
        ImGui::TextDisabled("FAR");
        far_drw->imGuiSettings();
        ImGui::EndChild();
        ImGui::BeginChild("NEAR", ImVec2{-1, ImGui::GetContentRegionAvail().y}, true);
        ImGui::TextDisabled("NEAR");
        near_drw->imGuiSettings();
        ImGui::EndChild();
        ImGui::EndGroup(); 
    }

    void 
    Radar::imGuiInfos()
    {
        ImGui::Text("%s",print.str().c_str());
    }

    void 
    Radar::onClose()
    {
        
    }
    
    std::string 
    Radar::toString()
    {
        return name;
    }
}}