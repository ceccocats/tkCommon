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
#endif

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/simpleMesh.h"
#include "tkCommon/gui/utils/SimpleMesh.h"
#include "tkCommon/gui/utils/PerlinNoise.h"

namespace tk { 
namespace gui {

class LaneletMap : public Drawable {
public:
     LaneletMap(const std::string& aConfPath, const std::string& aName = "Lanelet2_map");
    ~LaneletMap() = default;

    void onInit(tk::gui::Viewer *viewer);
    void beforeDraw(tk::gui::Viewer *viewer) final;
    void draw(tk::gui::Viewer *viewer) final;
    void imGuiSettings() final;
    void imGuiInfos() final;
    void onClose() final;
private:
    std::string mMapPath;
    std::string mConfPath;
    double      mOriginLat, mOriginLon;
    tk::math::Vec2f mMapMin, mMapMax, mMapSize;

    bool    mUpdate;
    bool    mInitted = false;
 
    std::vector<tk::gui::SimpleMesh>    mBuildingMesh, mGreenlandMesh, mParkingMesh, mRoadMesh, mLineMesh;

    bool                                mShowBuilding, mShowRoad, mShowGreenland, mShowParking;
    std::vector<glm::mat4>              mGlBuildingPose, mGlGreenlandPose, mGlParkingPose, mGlRoadPose, mGlLinesPose;
    std::vector<tk::gui::Buffer<float>> mGlBuildingData, mGlGreenlandData, mGlParkingData, mGlRoadData, mGlLinesData;
    tk::gui::Color_t                    mBuildingColor, mGrennlandColor, mParkingColor, mRoadColor, mLineColor;

    tk::gui::PerlinNoise *pn;

#ifdef LANELET_ENABLED
    tk::gui::SimpleMesh createBuilding(const lanelet::Area &area, bool height);
    tk::gui::SimpleMesh createRoad(const lanelet::Lanelet &lane);
    tk::gui::SimpleMesh createLine(lanelet::ConstLineString3d line);
#endif
};
}}