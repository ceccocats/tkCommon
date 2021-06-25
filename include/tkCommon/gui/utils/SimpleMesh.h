#pragma once

#include <vector>
#include <array>

#include "tkCommon/common.h"
#include "tkCommon/math/Vec.h"
#include "tkCommon/gui/utils/earcut.hpp"

namespace tk { namespace gui {

    class SimpleMesh {
    public:
         SimpleMesh();
        ~SimpleMesh();

        void createLine(std::vector<tk::math::Vec2f> &aLine, float width);
        //void createPolygon(std::vector<tk::math::Vec2f> &aOuterBound, double z);
        void createPrism(std::vector<tk::math::Vec2f> &aBase, double aHeight = 0.0, bool aCalcCentroid = true);

        float* vertexBufferPositionNormal(int &n);

        int size() { return mMesh.size(); }

        tk::common::Tfpose  pose;
    private:
        std::vector<std::array<float, 6>> mMesh;

        std::vector<tk::math::Vec2f> calcLineNormals(std::vector<tk::math::Vec2f> &aLine);
        std::vector<tk::math::Vec2f> offsetLine(std::vector<tk::math::Vec2f> &aLine, std::vector<tk::math::Vec2f> &aLineNormals, float aOffset);
    };
}}