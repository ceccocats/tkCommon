
#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <initializer_list>
#include "tkCommon/CudaCommon.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace math {

template<class T, int N>
class Vec : public tk::math::MatStatic<T,1,N>{
public:
    Vec() : MatStatic<T,1,N>(){
        for(int i = 0; i < N; i++){
            this->data_h[i] = 0;
        }
    }
    ~Vec() {}
    
    friend std::ostream& operator<<(std::ostream& os, const Vec<T,N>& s) {
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

    T& x(){return this->data_h[0];}
    T& y(){return this->data_h[1];}
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

    T& x(){return this->data_h[0];}
    T& y(){return this->data_h[1];}
    T& z(){return this->data_h[2];}
};

}}