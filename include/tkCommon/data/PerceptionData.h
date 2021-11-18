#pragma once
#include <vector>
#include "tkCommon/math/Vec.h"
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace data{

class box{
    public:
        tk::common::Vector3<float> pose; //Position: X,Y,Z
        tk::common::Vector3<float> size; //Size:     W,H,D
        tk::common::Vector3<float> rot;  //Rotation: roll, pitch, yaw
        float &x = pose.x();
        float &y = pose.y();
        float &z = pose.z();
        float &w = size.x();
        float &h = size.y();
        float &d = size.z();

        box(){
            pose = tk::common::Vector3<float>(0,0,0);
            size = tk::common::Vector3<float>(0,0,0);
            rot  = tk::common::Vector3<float>(0,0,0);
        }

        ~box(){
        }

        box(const box& b){
            this->pose = b.pose;
            this->size = b.size;
            this->rot = b.rot;
        }

        box(float x, float y, float w, float h){
            pose = tk::common::Vector3<float>(x,y,0);
            size = tk::common::Vector3<float>(w,h,0);
            rot  = tk::common::Vector3<float>(0,0,0);
        }
 
        box(float x, float y, float z, float w, float h, float d, float roll, float pitch, float yaw){
            pose = tk::common::Vector3<float>(x,y,z);
            size = tk::common::Vector3<float>(w,h,d);
            rot  = tk::common::Vector3<float>(roll,pitch,yaw);
        }

        box& 
        operator=(const box& b){
            this->pose = b.pose;
            this->size = b.size;
            this->rot = b.rot;
            return *this;
        }
};

class object {
    public:

        enum Class : uint8_t{
            PEDESTRIAN  ,
            CAR         ,
            TRUCK       ,
            BUS         ,
            MOTOBIKE    ,
            CYCLE       ,
            RIDER       ,
            LIGHT		,
            ROADSIGN	,
            TRAIN   	,
            NOT_CLS     
        };


    public:
        box   cam;
        box   world;
        float  confidence;
        float  score = 0.0f;
        Class cl;
        int id = -1;
        int sensorID = 0;
        timeStamp_t ts;

        object& 
        operator=(const object& b){
            this->world = b.world;
            this->confidence = b.confidence;
            this->score = b.score;
            this->cl = b.cl;
            this->id = b.id;
            this->sensorID = b.sensorID;
            this->ts = b.ts;
            return *this;
        }
};

class ObjectBuffer : public std::vector<object>, public tk::rt::Lockable {};

}}