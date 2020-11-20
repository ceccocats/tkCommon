#pragma once
#include <vector>
#include "tkCommon/math/Vec.h"
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace data{

class box{
    public:
        float x, y, z; //Position
        float w, h, d; //Size
        float r, p, y; //roll, pitch, yaw

        box(){
            x = y = z = w = h = d = r = p = y = 0;
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

        box(float x, float y, float w, float h, float r, float p){
            this->x = x;
            this->y = y;
            this->w = w;
            this->h = h;
            this->r = r;
            this->p = p;
        }
 
        box(float x, float y, float z, float w, float h, float d, float r, float p, float y){
            this->x = x;
            this->y = y;
            this->z = z;
            this->w = w;
            this->h = h;
            this->d = d;
            this->r = r;
            this->p = p;
            this->y = y;
        }

        box& 
        operator=(const box& b){
            this->x = b.x;
            this->y = b.y;
            this->z = b.z;
            this->w = b.w;
            this->h = b.h;
            this->d = b.d;
            this->r = b.r;
            this->p = b.p;
            this->y = b.y;
        }
};

}}