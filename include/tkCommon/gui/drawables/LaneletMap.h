#pragma once

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/simpleMesh.h"
#include "tkCommon/gui/utils/SimpleMesh.h"
#include "tkCommon/gui/utils/PerlinNoise.h"
#include "tkCommon/lanelet/LaneletInterface.h"
#include "tkCommon/gui/drawables/Mesh.h"

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

    class Semaphore {
        public:
            tk::gui::Mesh* mBody;
            tk::gui::Mesh* mRed;
            tk::gui::Mesh* mGreen;
            tk::gui::Mesh* mYellow;
            void setRed();
            void setGreen();
            void setYellow();
    };

    std::vector<Semaphore> mSemList;
    
private:
    std::string             mConfPath;
    tk::common::LaneletInterface     mLanelet;
    tk::math::Vec2f         mMapMin, mMapMax, mMapSize;

    bool    mUpdate;
    bool    mInitted = false;
 
    std::vector<tk::gui::SimpleMesh>    mBuildingMesh, mGreenlandMesh, mParkingMesh, mRoadMesh, mLineMesh;

    bool                                mShowBuilding, mShowRoad, mShowGreenland, mShowParking;
    std::vector<glm::mat4>              mGlBuildingPose, mGlGreenlandPose, mGlParkingPose, mGlRoadPose, mGlLinesPose, mGlTrafficLightPose;
    std::vector<tk::gui::Buffer<float>> mGlBuildingData, mGlGreenlandData, mGlParkingData, mGlRoadData, mGlLinesData;
    tk::gui::Color_t                    mBuildingColor, mGrennlandColor, mParkingColor, mRoadColor, mLineColor;

    tk::gui::PerlinNoise *pn;

    std::vector<tk::common::Tfpose> mSemTF;

#ifdef LANELET_ENABLED
    tk::gui::SimpleMesh createBuilding(const lanelet::Area &area, bool height);
    tk::gui::SimpleMesh createRoad(const lanelet::Lanelet &lane);
    tk::gui::SimpleMesh createLine(lanelet::ConstLineString3d line, float width);
#endif
};
}}