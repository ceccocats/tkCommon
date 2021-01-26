
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

template<class T, int N = -1>
class Vec : public tk::math::Mat<T,N,1>{
public:
    Vec() : Mat<T,N,1>(){

    }

    ~Vec(){

    }

    friend std::ostream& operator<<(std::ostream& os, Vec<T,N>& s) {
        os<<"vec( "<<s.rows()<<" )";
        return os;
    }

    __host__ void 
    resize(int n) {
        Mat<T,N,1>::resize(n,1);
    }

};

template<class T>
class Vec2 : public Vec<T,2> {
public:

    Vec2() : Vec<T,2>(){
    }

    Vec2(T x, T y) {
        this->x() = x;
        this->y() = y;
    }

    ~Vec2() {
    }

    T& x() { return this->cpu.data[0]; }
    T& y() { return this->cpu.data[1]; }
};

template<class T>
class Vec3 : public Vec<T,3> {
public:

    Vec3() : Vec<T,3>(){
    }

    Vec3(T x, T y, T z) {
        this->x() = x;
        this->y() = y;
        this->z() = z;
    }

    ~Vec3() {
    }

    T& x() { return this->cpu.data[0]; }
    T& y() { return this->cpu.data[1]; }
    T& z() { return this->cpu.data[2]; }
};

template<class T>
class Vec4 : public Vec<T,4> {
public:

    Vec4() : Vec<T,4>(){
    }

    Vec4(T x, T y, T z, T w) {
        this->x() = x;
        this->y() = y;
        this->z() = z;
        this->w() = w;
    }

    ~Vec4() {
    }

    T& x() { return this->cpu.data[0]; }
    T& y() { return this->cpu.data[1]; }
    T& z() { return this->cpu.data[2]; }
    T& w() { return this->cpu.data[3]; }
};


typedef Vec2<double> Vec2d;
typedef Vec3<double> Vec3d;
typedef Vec4<double> Vec4d;

typedef Vec2<float> Vec2f;
typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;


}}