#pragma once
#include "tkCommon/common.h"

namespace tk{namespace data{

struct ObjectData2D_t{
    tk::common::Rect<float>    box;
    tk::common::Vector4<float> color;
    std::string label;

    ObjectData2D_t &operator=(const ObjectData2D_t& s){
        box = s.box;
        color = s.color;
        label = s.label;
    }
};

struct ObjectData3D_t{
    tk::common::Vector3<float> pose;
    tk::common::Vector3<float> size;
    tk::common::Vector4<float> color;
    std::string label;

    ObjectData3D_t &operator=(const ObjectData3D_t& s){
        pose = s.pose;
        size = s.size;
        color = s.color;
        label = s.label;
    }
};

template <class T>
struct LineData_t{
    std::vector<T> points;
    tk::common::Vector4<float> color;

    LineData_t &operator=(const LineData_t& s){
        points = s.points;
        color = s.color;
    }
    void push(T point){
        points.push_back(point);
    }
    void pop(){
        points.pop_back();
    }
};

template <class T>
struct BoundaryData_t{
    std::vector<T> points;
    std::vector<tk::common::Vector4<float>> color;

    BoundaryData_t &operator=(const BoundaryData_t& s){
        points = s.points;
        color = s.color;
    }
    void push(T point, tk::common::Vector4<float> col){
        points.push_back(point);
        color.push_back(col);
    }
    void pop(){
        points.pop_back();
        color.pop_back();
    }
};

typedef LineData_t<tk::common::Vector2<float>> LineData2D_t;
typedef LineData_t<tk::common::Vector3<float>> LineData3D_t;

typedef BoundaryData_t<tk::common::Vector2<float>> BoundaryData2D_t;
typedef BoundaryData_t<tk::common::Vector3<float>> BoundaryData3D_t;

}}
