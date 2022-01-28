#include "tkCommon/gui/drawables/Kistler.h"

using namespace tk::gui;

Kistler::Kistler(std::string name)
{
    this->name  = name;
}

Kistler::Kistler(tk::data::KistlerData* data, std::string name)
{
    this->name  = name;
    this->data  = data;
}

Kistler::~Kistler()
{

}

void 
Kistler::imGuiInfos()
{
    ImGui::Text("%s",print.str().c_str());
}

void 
Kistler::imGuiSettings()
{

}

void 
Kistler::onClose()
{

}

void 
Kistler::drawData(tk::gui::Viewer *viewer)
{

}

void 
Kistler::updateData(tk::gui::Viewer *viewer)
{
    tk::data::KistlerData* kistler = dynamic_cast<tk::data::KistlerData*>(data);
    print.str("");
    print<<(*kistler);
}
