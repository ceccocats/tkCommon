#pragma once
#include "tkCommon/common.h"
#include "tkCommon/data/HeaderData.h"

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

struct LinesData_t {
public:
    struct LineInfo_t {
        int camIdx;
        char type;
    };

    tk::data::HeaderData header;
    std::vector<LineData3D_t> data;
    std::vector<LineInfo_t> info;

    matvar_t *toMatVar(std::string name = "lines") {
        size_t dims[] = {data.size(), 1};
        matvar_t *array = Mat_VarCreate(name.c_str(),MAT_C_CELL,MAT_T_CELL,2,dims,NULL,0);

        for(int i=0; i<data.size(); i++) {
            Eigen::MatrixXf p(3, data[i].points.size());
            for(int j=0; j<p.cols(); j++) {
                p(0, j) = data[i].points[j].x;
                p(1, j) = data[i].points[j].y;
                p(2, j) = data[i].points[j].z;
            }
            matvar_t *var = tk::common::eigenXf2matvar(p, "line");
            Mat_VarSetCell(array, i,var);
        }
        return array;
    }

    bool fromMatVar(matvar_t *var) {

        int n = var->dims[0];
        for(int i=0; i<n; i++) {
            matvar_t *pvar = Mat_VarGetCell(var, i);
            Eigen::MatrixXf mat = tk::common::matvar2eigenXf(pvar);

            LineData3D_t line;
            for(int j=0; j<mat.cols(); j++) {
                tk::common::Vector3<float> p;
                p.x = mat(0, j);
                p.y = mat(1, j);
                p.z = mat(2, j);
                line.points.push_back(p);
            }
            data.push_back(line);
        }
        return true;
    }
};


}}
