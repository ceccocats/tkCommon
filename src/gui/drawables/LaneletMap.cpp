#include "tkCommon/gui/drawables/LaneletMap.h"

using namespace tk::gui;

LaneletMap::LaneletMap(const std::string& aConfPath, const std::string& aName)
{
    name                = aName;
    mConfPath           = aConfPath;
    mUpdate             = false;
    mShowBuilding       = true;
    mShowRoad           = true;
    mShowGreenland      = true;
    mShowParking        = true;
    mBuildingColor.set((uint8_t) 145, (uint8_t) 145, (uint8_t) 145, (uint8_t) 255);
    mGrennlandColor.set((uint8_t) 0, (uint8_t) 134, (uint8_t) 24, (uint8_t) 255);
    mParkingColor.set((uint8_t) 0, (uint8_t) 117, (uint8_t) 138, (uint8_t) 255);
    mRoadColor.set((uint8_t) 50, (uint8_t) 50, (uint8_t) 50, (uint8_t) 255);
    mLineColor.set(1.0f, 1.0f, 1.0f, 1.0f);

    
    mInitted = true;
}

void
LaneletMap::onInit(Viewer *viewer)
{
#ifdef LANELET_ENABLED
    mLanelet.init(mConfPath);

    if (mLanelet.mMap == nullptr || mLanelet.mMap->empty()) {
        tkERR("Empty map.\n");
        return;
    }

    // init perlin noise
    pn = new tk::gui::PerlinNoise(11);

    // get map boundary
    mMapMin.x() = mMapMin.y() = std::numeric_limits<float>::max();
    mMapMax.x() = mMapMax.y() = std::numeric_limits<float>::min();
    for (const auto& point : mLanelet.mMap->pointLayer) {
        if (point.x() < mMapMin.x()) mMapMin.x() = point.x();
        if (point.y() < mMapMin.y()) mMapMin.y() = point.y();
        if (point.x() > mMapMax.x()) mMapMax.x() = point.x();
        if (point.y() > mMapMax.y()) mMapMax.y() = point.y();
    }
    mMapSize.x() = mMapMax.x() - mMapMin.x();
    mMapSize.y() = mMapMax.y() - mMapMin.y();

    // generate area mesh
    for(const auto& area : mLanelet.mMap->areaLayer) {
        if (area.attribute(lanelet::AttributeName::Subtype) == "building") {
            mBuildingMesh.push_back(createBuilding(area, true));
        }
        if (area.attribute(lanelet::AttributeName::Subtype) == "vegetation") {
            mGreenlandMesh.push_back(createBuilding(area, false));
        }
        if (area.attribute(lanelet::AttributeName::Subtype) == "parking") {
            mParkingMesh.push_back(createBuilding(area, false));
        }
    }

    // generate road mesh
    for(const auto& lane : mLanelet.mMap->laneletLayer) {
        auto right = lane.rightBound3d();
        if (right.attribute(lanelet::AttributeName::Subtype) == "solid") 
            mLineMesh.push_back(createLine(right));
        
        auto left = lane.leftBound3d();
        if (left.attribute(lanelet::AttributeName::Subtype) == "solid") 
            mLineMesh.push_back(createLine(left));

        mRoadMesh.push_back(createRoad(lane));
    }

    mGlBuildingPose.resize(mBuildingMesh.size());
    mGlBuildingData.resize(mBuildingMesh.size());
    for (int i = 0; i < mGlBuildingData.size(); ++i)
        mGlBuildingData[i].init();

    mGlGreenlandPose.resize(mGreenlandMesh.size());
    mGlGreenlandData.resize(mGreenlandMesh.size());
    for (int i = 0; i < mGlGreenlandData.size(); i++)
        mGlGreenlandData[i].init();

    mGlParkingPose.resize(mParkingMesh.size());
    mGlParkingData.resize(mParkingMesh.size());
    for (int i = 0; i < mGlParkingData.size(); i++)
        mGlParkingData[i].init();

    mGlRoadPose.resize(mRoadMesh.size());
    mGlRoadData.resize(mRoadMesh.size());
    for (int i = 0; i < mGlRoadData.size(); ++i)
        mGlRoadData[i].init();
    
    mGlLinesPose.resize(mLineMesh.size());
    mGlLinesData.resize(mLineMesh.size());
    for (int i = 0; i < mLineMesh.size(); ++i)
        mGlLinesData[i].init();

    shader  = tk::gui::shader::simpleMesh::getInstance(); 
    mUpdate = true;
#else
    tkWRN("You need to install lanelet2 to use this drawable.")
   
    shader              = tk::gui::shader::simpleMesh::getInstance();
    mShowBuilding       = false;
    mShowRoad           = false;
    mShowGreenland      = false;
    mShowParking        = false;
    mUpdate             = false;
#endif
}

void 
LaneletMap::beforeDraw(tk::gui::Viewer *viewer)
{
    
    if (mUpdate) {
        mUpdate = false;
        float* data;
        int n;

        // copy building to GPU
        for (int i = 0; i < mBuildingMesh.size(); ++i) {    
            data = mBuildingMesh[i].vertexBufferPositionNormal(n);
            mGlBuildingData[i].setData(data, n); 
            mGlBuildingPose[i] = glm::make_mat4x4(mBuildingMesh[i].pose.matrix().data());  
        }
        mBuildingMesh.clear();

        // copy greenland to GPU
        for (int i = 0; i < mGreenlandMesh.size(); ++i) {    
            data = mGreenlandMesh[i].vertexBufferPositionNormal(n);
            mGlGreenlandData[i].setData(data, n);   
            mGlGreenlandPose[i] = glm::make_mat4x4(mGreenlandMesh[i].pose.matrix().data());  
        }
        mGreenlandMesh.clear();

        // copy parking to GPU
        for (int i = 0; i < mParkingMesh.size(); ++i) {    
            data = mParkingMesh[i].vertexBufferPositionNormal(n);
            mGlParkingData[i].setData(data, n);   
            mGlParkingPose[i] = glm::make_mat4x4(mParkingMesh[i].pose.matrix().data());
        }
        mParkingMesh.clear();

        // copy road to GPU
        for (int i = 0; i < mRoadMesh.size(); ++i) {    
            data = mRoadMesh[i].vertexBufferPositionNormal(n);
            mGlRoadData[i].setData(data, n);     
            mGlRoadPose[i] = glm::make_mat4x4(mRoadMesh[i].pose.matrix().data());
        }
        mRoadMesh.clear();

        // copy lines to GPU
        tk::common::Tfpose tf;
        for (int i = 0; i < mLineMesh.size(); ++i) {
            data = mLineMesh[i].vertexBufferPositionNormal(n);
            mGlLinesData[i].setData(data, n);   
            tf.matrix() = mLineMesh[i].pose.matrix() * tk::common::odom2tf(0.0, 0.0, 0.1, 0.0).matrix();
            mGlLinesPose[i] = glm::make_mat4x4(tf.matrix().data());      
        }
        mLineMesh.clear();
    }
}

void 
LaneletMap::draw(tk::gui::Viewer *viewer)
{
    auto shaderLanelet = (tk::gui::shader::simpleMesh*)shader;
    if (mShowRoad) {
        shaderLanelet->draw(drwModelView, mGlRoadData, mGlRoadPose, viewer->getLightPos(), false, mRoadColor);
        shaderLanelet->draw(drwModelView, mGlLinesData, mGlLinesPose, viewer->getLightPos(), false, mLineColor);
    }
    if (mShowGreenland)
        shaderLanelet->draw(drwModelView, mGlGreenlandData, mGlGreenlandPose, viewer->getLightPos(), false, mGrennlandColor);
    
    if (mShowParking)
        shaderLanelet->draw(drwModelView, mGlParkingData, mGlParkingPose, viewer->getLightPos(), false, mParkingColor);
    
    if (mShowBuilding)
        shaderLanelet->draw(drwModelView, mGlBuildingData, mGlBuildingPose, viewer->getLightPos(), true, mBuildingColor);
}

void 
LaneletMap::imGuiSettings()
{
#ifdef LANELET_ENABLED
    ImGui::Checkbox("Show road", &mShowRoad);
    ImGui::ColorEdit4("Road color", mRoadColor.color);
    ImGui::ColorEdit4("Line color", mLineColor.color);
    ImGui::Separator();
    ImGui::Checkbox("Show building", &mShowBuilding);
    ImGui::ColorEdit4("Building color", mBuildingColor.color);
    ImGui::Separator();
    ImGui::Checkbox("Show greenland", &mShowGreenland);
    ImGui::ColorEdit4("Grennland color", mGrennlandColor.color);
    ImGui::Separator();
    ImGui::Checkbox("Show parking", &mShowParking);
    ImGui::ColorEdit4("Parking color", mParkingColor.color);
#endif
}

void 
LaneletMap::imGuiInfos()
{
}

void 
LaneletMap::onClose()
{
    auto shaderLanelet = (tk::gui::shader::simpleMesh*)shader;
    shaderLanelet->close();

    for (int i = 0; i < mGlBuildingData.size(); ++i)
        mGlBuildingData[i].release();
    for (int i = 0; i < mGlGreenlandData.size(); ++i)
        mGlGreenlandData[i].release();
    for (int i = 0; i < mGlParkingData.size(); ++i)
        mGlParkingData[i].release();
    for (int i = 0; i < mGlRoadData.size(); ++i)
        mGlRoadData[i].release();
    for (int i = 0; i < mGlLinesData.size(); ++i)
        mGlLinesData[i].release();
}

#ifdef LANELET_ENABLED
tk::gui::SimpleMesh 
LaneletMap::createBuilding(const lanelet::Area &area, bool height)
{
    tk::gui::SimpleMesh             mesh;
    std::vector<tk::math::Vec2f>    line;
    auto polygon = area.outerBoundPolygon();
    float h = 0.0f;
    
    // convert
    line.resize(polygon.size());
    for (int i = 0; i < polygon.size(); ++i) {
        line[i].x() = polygon[i].x();
        line[i].y() = polygon[i].y();
    }
    
    // calc building height with perlin noise
    if (height) {
        float noise = 20 * pn->noise((line[0].x() - mMapMin.x()) / mMapSize.x(), (line[0].y() - mMapMin.y()) / mMapSize.y(), 0.8);
        noise = noise - floor(noise);
        h = 6.0 + 40.0 * noise;
    }
    
    // create mesh
    mesh.createPrism(line, h);

    return mesh;
}

tk::gui::SimpleMesh 
LaneletMap::createRoad(const lanelet::Lanelet &lane)
{
    tk::gui::SimpleMesh             mesh;
    std::vector<tk::math::Vec2f>    line;
    auto polygon = lane.polygon2d();

    // convert
    line.resize(polygon.size() + 1);
    for (int i = 0; i < polygon.size(); ++i) {
        line[i].x() = polygon[i].x();
        line[i].y() = polygon[i].y();
    }
    line[line.size() - 1].x() = polygon[0].x();
    line[line.size() - 1].y() = polygon[0].y();

    // create mesh
    mesh.createPrism(line, 0.0);

    return mesh;
}

tk::gui::SimpleMesh 
LaneletMap::createLine(lanelet::ConstLineString3d line)
{
    tk::gui::SimpleMesh             mesh;
    std::vector<tk::math::Vec2f>    l;
    
    for (int i = 0; i < line.size(); ++i) {
        if ((i > 0) && (l[i - 1].x() == line[i].x()) && (l[i - 1].y() == line[i].y())) 
            continue;
        
        l.push_back(tk::math::Vec2f(line[i].x(), line[i].y()));
    }

    mesh.createLine(l, 0.1);

    return mesh;
}
#endif
