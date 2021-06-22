#pragma once

#ifdef LANELET_ENABLED
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Area.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/primitives/Area.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/primitives/Point.h>
#include <lanelet2_core/primitives/Polygon.h>
#include <lanelet2_core/utility/Units.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_io/io_handlers/Factory.h>
#include <lanelet2_io/io_handlers/Writer.h>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_routing/Route.h>
#include <lanelet2_routing/RoutingCost.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_routing/RoutingGraphContainer.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>
#endif

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/simpleMesh.h"
#include "tkCommon/gui/utils/SimpleMesh.h"

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