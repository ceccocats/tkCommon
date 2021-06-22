#include "tkCommon/gui/utils/SimpleMesh.h"

using namespace tk::gui;

SimpleMesh::SimpleMesh()
{
    pose = tk::common::odom2tf(0.0, 0.0, 0.0);
    mMesh.resize(0);
}

SimpleMesh::~SimpleMesh()
{

}

void 
SimpleMesh::createLine(std::vector<tk::math::Vec2f> &aLine, float width)
{
    std::vector<tk::math::Vec2f> normals = calcLineNormals(aLine);
    std::vector<tk::math::Vec2f> left    = offsetLine(aLine, normals, width/2.0f);
    std::vector<tk::math::Vec2f> right   = offsetLine(aLine, normals, -width/2.0f);

    std::vector<tk::math::Vec2f> outer;
    outer.resize(aLine.size()*2 + 1);
    std::copy(left.begin(), left.end(), outer.begin());
    std::reverse_copy(right.begin(), right.end(), outer.begin() + left.size());

    createPrism(outer, 0.0f);
}

std::vector<tk::math::Vec2f> 
SimpleMesh::calcLineNormals(std::vector<tk::math::Vec2f> &aLine)
{
    std::vector<tk::math::Vec2f> normals;
    normals.resize(aLine.size());

    float dx, dy;
    tk::math::Vec2f n;
    for (int i = 0; i < aLine.size(); ++i) {
        if (i == (aLine.size() - 1)) {
            dx      = aLine[i].x() - aLine[i-1].x();
            dy      = aLine[i].y() - aLine[i-1].y();
            n.x()   = -dy; 
            n.y()   =  dx;

            normals[i].writableMatrix() = n.matrix().normalized();
            break;
        }

        dx      = aLine[i+1].x() - aLine[i].x();
        dy      = aLine[i+1].y() - aLine[i].y();
        n.x()   = -dy; 
        n.y()   =  dx;

        if (i == 0) {
            normals[i].writableMatrix() = n.matrix().normalized();
        } else {
            normals[i].writableMatrix() = ((n.matrix().normalized() + normals[i - 1].matrix())/2.0f).normalized();
        }
    }

    return normals;
}

std::vector<tk::math::Vec2f> 
SimpleMesh::offsetLine(std::vector<tk::math::Vec2f> &aLine, std::vector<tk::math::Vec2f> &aLineNormals, float aOffset)
{
    std::vector<tk::math::Vec2f> offLine;
    offLine.resize(aLine.size());

    for (int i = 0; i < offLine.size(); ++i)
        offLine[i].writableMatrix() = aLine[i].matrix() + aOffset * aLineNormals[i].matrix();
    
    return offLine;
}

void 
SimpleMesh::createPrism(std::vector<tk::math::Vec2f> &aBase, double aHeight, bool aCalcCentroid)
{
    Eigen::Vector3f p1, p2, p3, N;
    std::vector<std::array<float, 2>> outer_polyline;
    std::vector<std::vector<std::array<float, 2>>> earcut_polygon;

    if (aCalcCentroid) {
        // get centroid
        float cx = 0;
        float cy = 0;
        for (int i = 0; i < aBase.size() - 1; ++i) {
            cx += aBase[i].x();
            cy += aBase[i].y();
        }
        cx /= aBase.size()-1;
        cy /= aBase.size()-1;

        // add point to polyline
        for (int i = 0; i < aBase.size() - 1; ++i) {
            outer_polyline.push_back({aBase[i].x() - cx, aBase[i].y() - cy});
        }

        pose = tk::common::odom2tf(cx, cy, aHeight/2.0f, 0.0);
    } else {
        // add point to polyline
        for (int i = 0; i < aBase.size() - 1; ++i) {
            outer_polyline.push_back({aBase[i].x(), aBase[i].y()});
        }
        pose = tk::common::odom2tf(0, 0, 0, 0);
    }
    
    // add outer polyline
    earcut_polygon.push_back(outer_polyline);

    // calc triangulation
    std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(earcut_polygon);

    if (aHeight != 0)
        mMesh.resize(indices.size() + earcut_polygon[0].size()*6);
    else 
        mMesh.resize(indices.size());

    // add roof mesh
    for (int i = 0; i < indices.size(); ++i) {
        mMesh[i].at(0) = (float) earcut_polygon[0].at(indices[i])[0];
        mMesh[i].at(1) = (float) earcut_polygon[0].at(indices[i])[1]; 
        mMesh[i].at(2) = aHeight/2.0f;

        // calc normal 
        if ((i+1) % 3 == 0) {
            p1 = {mMesh[i - 2].at(0), mMesh[i - 2].at(1), mMesh[i - 2].at(2)};
            p2 = {mMesh[i - 1].at(0), mMesh[i - 1].at(1), mMesh[i - 1].at(2)};
            p3 = {mMesh[i - 0].at(0), mMesh[i - 0].at(1), mMesh[i - 0].at(2)};

            // N = (p2 - p1) x (p3 - p1)
            N = (p2 - p1).cross(p3 - p1);

            // p1
            mMesh[i - 2].at(3) = N.x();
            mMesh[i - 2].at(4) = N.y();
            mMesh[i - 2].at(5) = N.z();

            // p2
            mMesh[i - 1].at(3) = N.x();
            mMesh[i - 1].at(4) = N.y();
            mMesh[i - 1].at(5) = N.z();

            // p3
            mMesh[i - 0].at(3) = N.x();
            mMesh[i - 0].at(4) = N.y();
            mMesh[i - 0].at(5) = N.z();
        }
    }
    // add lateral mesh
    if (aHeight != 0) {
        for (int i = 0; i < earcut_polygon[0].size(); ++i) {
            int j = (i+1) % earcut_polygon[0].size();

            p1 = {(float) earcut_polygon[0].at(i)[0], (float) earcut_polygon[0].at(i)[1], (float)  aHeight/2.0f};
            p2 = {(float) earcut_polygon[0].at(j)[0], (float) earcut_polygon[0].at(j)[1], (float)  aHeight/2.0f};
            p3 = {(float) earcut_polygon[0].at(i)[0], (float) earcut_polygon[0].at(i)[1], (float) -aHeight/2.0f};
            N = (p2 - p1).cross(p3 - p1);

            mMesh[indices.size() + i*6 + 0].at(0) = p1.x(); 
            mMesh[indices.size() + i*6 + 0].at(1) = p1.y(); 
            mMesh[indices.size() + i*6 + 0].at(2) = p1.z();
            mMesh[indices.size() + i*6 + 0].at(3) = N.x();
            mMesh[indices.size() + i*6 + 0].at(4) = N.y();
            mMesh[indices.size() + i*6 + 0].at(5) = N.z();

            mMesh[indices.size() + i*6 + 1].at(0) = p2.x(); 
            mMesh[indices.size() + i*6 + 1].at(1) = p2.y(); 
            mMesh[indices.size() + i*6 + 1].at(2) = p2.z();
            mMesh[indices.size() + i*6 + 1].at(3) = N.x();
            mMesh[indices.size() + i*6 + 1].at(4) = N.y();
            mMesh[indices.size() + i*6 + 1].at(5) = N.z();

            mMesh[indices.size() + i*6 + 2].at(0) = p3.x(); 
            mMesh[indices.size() + i*6 + 2].at(1) = p3.y(); 
            mMesh[indices.size() + i*6 + 2].at(2) = p3.z();
            mMesh[indices.size() + i*6 + 2].at(3) = N.x();
            mMesh[indices.size() + i*6 + 2].at(4) = N.y();
            mMesh[indices.size() + i*6 + 2].at(5) = N.z();



            p1 = {(float) earcut_polygon[0].at(i)[0], (float) earcut_polygon[0].at(i)[1], (float) -aHeight/2.0f};
            p2 = {(float) earcut_polygon[0].at(j)[0], (float) earcut_polygon[0].at(j)[1], (float)  aHeight/2.0f};
            p3 = {(float) earcut_polygon[0].at(j)[0], (float) earcut_polygon[0].at(j)[1], (float) -aHeight/2.0f};
            N = (p2 - p1).cross(p3 - p1);
            
            mMesh[indices.size() + i*6 + 3].at(0) = p1.x();  
            mMesh[indices.size() + i*6 + 3].at(1) = p1.y();  
            mMesh[indices.size() + i*6 + 3].at(2) = p1.z();
            mMesh[indices.size() + i*6 + 3].at(3) = N.x();
            mMesh[indices.size() + i*6 + 3].at(4) = N.y();
            mMesh[indices.size() + i*6 + 3].at(5) = N.z();
            
            mMesh[indices.size() + i*6 + 4].at(0) = p2.x();  
            mMesh[indices.size() + i*6 + 4].at(1) = p2.y();  
            mMesh[indices.size() + i*6 + 4].at(2) = p2.z();
            mMesh[indices.size() + i*6 + 4].at(3) = N.x();
            mMesh[indices.size() + i*6 + 4].at(4) = N.y();
            mMesh[indices.size() + i*6 + 4].at(5) = N.z();
            
            mMesh[indices.size() + i*6 + 5].at(0) = p3.x();  
            mMesh[indices.size() + i*6 + 5].at(1) = p3.y();  
            mMesh[indices.size() + i*6 + 5].at(2) = p3.z();
            mMesh[indices.size() + i*6 + 5].at(3) = N.x();
            mMesh[indices.size() + i*6 + 5].at(4) = N.y();
            mMesh[indices.size() + i*6 + 5].at(5) = N.z();
        }
    }
}

float* 
SimpleMesh::vertexBufferPositionNormal(int &n)
{
    n = mMesh.size() *6;
    return (float*) mMesh.data();
}

