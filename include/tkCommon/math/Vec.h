
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

namespace tk { namespace math {

template<class T>
class Vec2 : public tk::math::MatDump {
public:
    T x, y;
    Vec2() {
        x = y = 0;
    }
    Vec2(T x, T y) {
        this.x = x;
        this.y = y;
    }
    ~Vec2() {}
    
    friend std::ostream& operator<<(std::ostream& os, const Vec2& s) {
        os<<"vec2("<<s.x<<", "<<s.y<<")";
        return os;
    }
};

template<class T>
class Vec3 : public Vec2<T> {
public:
    T z;

    Vec3() {
        this.x = this.y = this.z = 0;
    }
    Vec3(T x, T y, T z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
    ~Vec3() {
    }
    friend std::ostream& operator<<(std::ostream& os, const Vec3& s) {
        os<<"vec3("<<s.x<<", "<<s.y<<", "<<s.z<<")";
        return os;
    }
};

}}