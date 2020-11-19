#pragma once
#include <vector>
#include "tkCommon/math/Vec.h"
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace data{

class box{
    public:
        float x, y, z;
        float w, h, d;

        box(){
            x = y = z = w = h = d = 0;
        }

        ~box(){
        }

        box(const box& b){
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            this->w = b.w;
            this->h = b.h;
            this->d = b.d;
        }

        box(float x, float y, float w, float h){
            this->x = x;
            this->y = y;
            this->w = w;
            this->h = h;
        }
 
        box(float x, float y, float z, float w, float h, float d){
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;
            this->h = h;
            this->d = d;
        }

        box& 
        operator=(const box& b){
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            this->w = b.w;
            this->h = b.h;
            this->d = b.d;
        }
};

}}