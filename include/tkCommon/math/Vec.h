
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

    friend std::ostream& operator<<(std::ostream& os, const Vec<T>& s) {
        os<<"vec( "<<s.cols()<<" )";
        return os;
    }

    __host__ void 
    resize(int r, int c) {
        tkASSERT(r == 1, "You can't create a 2D vector");
        Mat<T>::resize(1,c);
    }

    __host__ void 
    resize(int n) {
        Mat<T>::resize(1,n);
    }

    __host__ T&
    operator()(int n) {
        //tkASSERT(n < this->_size, "Out of memory " + std::to_string(n) + " of " + std::to_string(this->_size));
        return this->data_h[n]; 
    }

    __host__ T&
    operator[](int n) {
        //tkASSERT(n < this->_size, "Out of memory " + std::to_string(n) + " of " + std::to_string(this->_size));
        return this->data_h[n]; 
    }
};

template<class T, int N>
class VecStatic : public tk::math::MatStatic<T,1,N>{
public:
    VecStatic() : MatStatic<T,1,N>(){
        for(int i = 0; i < N; i++){
            this->data_h[i] = 0;
        }
    }
    ~VecStatic() {}
    
    friend std::ostream& operator<<(std::ostream& os, const VecStatic<T,N>& s) {
        if(N > 4){
            os<<"vec("<<N<<")";
        }else{
            os<<"vec"<<N<<"(";
            for(int i = 0; i < N; i++){
                os<<s.data_h[i];
                if(i<N-1)
                    os<<", ";
            }
            os<<")";
        }
        return os;
    }

     __host__ T&
    operator()(int n) {
        tkASSERT(n < this->_size, "Out of memory, "+std::to_string(n)+" > "+std::to_string(this->_size)+"\n")
        return this->data_h[n]; 
    }

    __host__ T&
    operator[](int n) {
        tkASSERT(n < this->_size, "Out of memory, "+std::to_string(n)+" > "+std::to_string(this->_size)+"\n")
        return this->data_h[n]; 
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

    T& x(){return this->data_h[0];}
    T& y(){return this->data_h[1];}
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

    T& x(){return this->data_h[0];}
    T& y(){return this->data_h[1];}
    T& z(){return this->data_h[2];}
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

    T& x(){return this->data_h[0];}
    T& y(){return this->data_h[1];}
    T& z(){return this->data_h[2];}
    T& w(){return this->data_h[3];}
};


typedef Vec2<double> Vec2d;
typedef Vec3<double> Vec3d;
typedef Vec4<double> Vec4d;

typedef Vec2<float> Vec2f;
typedef Vec3<float> Vec3f;
typedef Vec4<float> Vec4f;


}}