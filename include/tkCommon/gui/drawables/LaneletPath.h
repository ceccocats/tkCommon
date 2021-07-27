#pragma once

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/simpleMesh.h"
#include "tkCommon/gui/utils/SimpleMesh.h"
#include "tkCommon/lanelet/Lanelet.h"

namespace tk { namespace gui {
class LaneletPath : public Drawable {
public:
     LaneletPath(const std::string& aName = "Lanelet2_path");

    void onInit(tk::gui::Viewer *viewer);
    void beforeDraw(tk::gui::Viewer *viewer) final;
    void draw(tk::gui::Viewer *viewer) final;
    void imGuiSettings() final;
    void imGuiInfos() final;
    void onClose() final;
#ifdef LANELET_ENABLED
    void updateRef(lanelet::routing::LaneletPath *path);
#endif
private:
    std::mutex  mtx;
    bool        mUpdate;
    float       mDistance;
#ifdef LANELET_ENABLED
    lanelet::routing::LaneletPath   mPath;
#endif
    tk::gui::SimpleMesh             mPathMesh;
    tk::gui::Buffer<float>          mGlPathData;
    tk::gui::Color_t                mPathColor;
};
}}