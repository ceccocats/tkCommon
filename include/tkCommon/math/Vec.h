
#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <initializer_list>
#include "tkCommon/cuda.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace math {

template<class T>
class Vec : public tk::math::Mat<T>{
public:
    Vec() : Mat<T>(){

    }

    ~Vec(){

    }

    friend std::ostream& operator<<(std::ostream& os, Vec<T>& s) {
        os<<"vec( "<<s.rows()<<" )";
        return os;
    }

    __host__ void 
    resize(int n) {
        Mat<T>::resize(n,1);
    }

    __host__ void 
    resize(int r, int c) {
        tkASSERT(c == 1, "this is a vector!");
        Mat<T>::resize(r,c);
    }


};

template<class T>
class Vec2 : public MatStatic<T,2,1> {
public:

    Vec2() : MatStatic<T,2,1>(){
    }

    Vec2(T x, T y) {
        this->x() = x;
        this->y() = y;
    }

    ~Vec2() {
    }

    T& x() { return this->mData[0]; }
    T& y() { return this->mData[1]; }
};

template<class T>
class Vec3 : public MatStatic<T,3,1> {
public:

    Vec3() : MatStatic<T,3,1>(){
    }

    Vec3(T x, T y, T z) {
        this->x() = x;
        this->y() = y;
        this->z() = z;
    }

    ~Vec3() {
    }

    T& x() { return this->mData[0]; }
    T& y() { return this->mData[1]; }
    T& z() { return this->mData[2]; }
};

template<class T>
class Vec4 : public MatStatic<T,4,1> {
public:

    Vec4() : MatStatic<T,4,1>(){
    }

    Vec4(T x, T y, T z, T w) {
        this->x() = x;
        this->y() = y;
        this->z() = z;
        this->w() = w;
    }

    ~Vec4() {
    }

    T& x() { return this->mData[0]; }
    T& y() { return this->mData[1]; }
    T& z() { return this->mData[2]; }
    T& w() { return this->mData[3]; }
};


typedef Vec2<double> Vec2d;
typedef Vec3<double> Vec3d;
typedef Vec4<double> Vec4d;

typedef Vec2<float> Vec2f;
typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;


}}