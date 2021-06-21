#include "tkCommon/gui/drawables/LaneletPath.h"

using namespace tk::gui;

LaneletPath::LaneletPath(const std::string& aName)
{
    this->name      = aName;
    this->mUpdate   = false;
    this->mPathColor.set((uint8_t) 255, (uint8_t) 255, (uint8_t) 0, (uint8_t) 255);
}

void 
LaneletPath::onInit(tk::gui::Viewer *viewer)
{
    mGlPathData.init();
    shader = tk::gui::shader::simpleMesh::getInstance();

#ifndef LANELET_ENABLED
    tkWRN("You need to install lanelet2 to use this drawable.")
#endif
}

void 
LaneletPath::beforeDraw(tk::gui::Viewer *viewer)
{
#ifdef LANELET_ENABLED
    mtx.lock();
    if (mUpdate) {
        std::vector<tk::math::Vec2f>    l;
        int idx = 0;
        mDistance = 0.0f;
        for (const auto& lane : mPath) {
            mDistance += lanelet::geometry::length(lanelet::utils::toHybrid(lane.centerline2d()));

            auto line = lane.centerline2d();
            for (int i = 0; i < line.size(); ++i) {
                if ((idx > 0) && (l[idx - 1].x() == line[i].x()) && (l[idx - 1].y() == line[i].y())) 
                    continue;
                
                l.push_back(tk::math::Vec2f(line[i].x(), line[i].y()));
                ++idx;
            }
        }
        mPathMesh.createLine(l, 2.0);
        mDistance /= 1000.0f; // convert to km/h
       

        // copy path to GPU
        float* data;
        int n;
        data = mPathMesh.vertexBufferPositionNormal(n);
        mGlPathData.setData(data, n);    
        
        mUpdate = false;
    }
    mtx.unlock();
#endif
}

void 
LaneletPath::draw(tk::gui::Viewer *viewer)
{   
    if (mPathMesh.size() > 0) {
        auto shaderLanelet = (tk::gui::shader::simpleMesh*)shader;

        tk::common::Tfpose tf;
        tf.matrix() = mPathMesh.pose.matrix() * tk::common::odom2tf(0.0, 0.0, 0.5, 0.0).matrix();
        
        glm::mat4 MVP           = drwModelView * glm::make_mat4x4(tf.data());
        shaderLanelet->draw(MVP, &mGlPathData, mGlPathData.size()/6, viewer->getLightPos(), true, mPathColor);
    }
}

void 
LaneletPath::imGuiSettings()
{
#ifdef LANELET_ENABLED
    ImGui::ColorEdit4("Path color", mPathColor.color);
#endif
}

void 
LaneletPath::imGuiInfos()
{
#ifdef LANELET_ENABLED
    ImGui::Text("Path lenght %f", mDistance);
#endif
}

void 
LaneletPath::onClose()
{
    auto shaderLanelet = (tk::gui::shader::simpleMesh*)shader;
    shaderLanelet->close();

    mGlPathData.release();
}

#ifdef LANELET_ENABLED
void 
LaneletPath::updateRef(lanelet::routing::LaneletPath *path)
{
    mtx.lock();

    this->mPath     = *path;
    this->mUpdate   = true;

    mtx.unlock();
}
#endif