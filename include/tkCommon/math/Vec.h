
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

template<class T, int N>
class VecStatic : public MatStatic<T, N, 1> {
public: 

    VecStatic() : MatStatic<T,N,1>() {
        for(int i=0; i<N; i++) {
            this->mData[i] = 0;
        }
    }

    VecStatic<T,N>& operator += ( VecStatic<T,N> const& s ) {
        for(int i=0; i<N; i++) {
            this->mData[i] += s.mData[i];
        }
        return *this;
    }

    template<int DIM>
    VecStatic<T,DIM> sub() {
        VecStatic<T,DIM> v;
        memcpy(v.data(), this->mData, sizeof(T)*DIM);
        return v;
    }

    T dist(VecStatic<T,N> const& s, int DIM = N) {
        T d = 0;
        for(int i=0; i<DIM; i++) {
            d += (this->mData[i] - s.mData[i])*(this->mData[i] - s.mData[i]);
        }
        return sqrt(d);
    }

    friend std::ostream& 
    operator<<(std::ostream& os, VecStatic<T,N>& s) {
        os<<"vec"<<N<<" ( ";
        for(int i=0; i<N; i++)
            os<<s.mData[i]<<" ";
        os<<")";
        return os;
    }
};


template<class T>
class Vec2 : public VecStatic<T,2> {
public:

    Vec2() : VecStatic<T,2>(){
    }

    Vec2(T x, T y) {
        this->x() = x;
        this->y() = y;
    }

    ~Vec2() {
    }

    T& x() { return this->mData[0]; }
    T& y() { return this->mData[1]; }

    T x() const { return this->mData[0]; }
    T y() const { return this->mData[1]; }
};

template<class T>
class Vec3 : public VecStatic<T,3> {
public:

    Vec3() : VecStatic<T,3>(){
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

    T x() const { return this->mData[0]; }
    T y() const { return this->mData[1]; }
    T z() const { return this->mData[2]; }
};

template<class T>
class Vec4 : public VecStatic<T,4> {
public:

    Vec4() : VecStatic<T,4>(){
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

    T x() const { return this->mData[0]; }
    T y() const { return this->mData[1]; }
    T z() const { return this->mData[2]; }
    T w() const { return this->mData[3]; }
};

Vec3<double> quat2euler(Vec4<double> aQuaternion);
Vec4<double> euler2quat(Vec3<double> aEuler);

typedef Vec2<double> Vec2d;
typedef Vec3<double> Vec3d;
typedef Vec4<double> Vec4d;

typedef Vec2<float> Vec2f;
typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;
}}