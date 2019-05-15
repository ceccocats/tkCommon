#include <iostream>

namespace tk{ namespace common{ 

struct semaphore{

    int     status;
};

struct carPosition{

    float   latitude;
    float   longitude;
    float   angle;
};

struct Obj{

    float   x;      //KF
    float   y;      //KF
    float   yaw;    //KF
    float   speed;  //KF
    float   acc;    //KF
    float   x0;
    float   y0;
    float   z0;
    float   x1;
    float   y1;
    float   z1;
    int     cl;     //CLASS
};

static const int    MAXOBJ = 50;

struct ObjFusion{

    int     n;
    Obj     obj[MAXOBJ];
};

}}