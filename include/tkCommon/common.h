#pragma once

#include <iostream>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include "Eigen/Dense"

namespace tk { namespace common {

    // Vector 4 class
    template <class T> 
    class Vector4 {
        public:
            T x, y, z, i;

            Vector4() {
                x = y = z = i = 0;
            }

            Vector4(T x, T y, T z, T i) {
                this->x = x;
                this->y = y;
                this->z = z;
                this->i = i;
            }

            ~Vector4() {}

            friend std::ostream& operator<<(std::ostream& os, const Vector4& v) {
                os << "v4(" << v.x <<", "<< v.y <<", "<< v.z <<", "<< v.i <<")";
                return os;
            }  
    };

    // Vector3 class
    template <class T> 
    class Vector3 {
        public:        
            T x, y, z;

            Vector3() {
                x = y = z = 0;
            }

            Vector3(T x, T y, T z) {
                this->x = x;
                this->y = y;
                this->z = z;
            }

            ~Vector3() {}

            friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
                os << "v3(" << v.x <<", "<< v.y <<", "<< v.z <<")";
                return os;
            }  
    };

    // Vector2 class
    template <class T> 
    class Vector2 {
        public:        
            T x, y;

            Vector2() {
                x = y = 0;
            }

            Vector2(T x, T y) {
                this->x = x;
                this->y = y;
            }

            ~Vector2() {}

            friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
                os << "v2(" << v.x <<", "<< v.y <<")";
                return os;
            }  
    };

    typedef Eigen::Isometry3f Tfpose;
    typedef uint64_t TimeStamp;

    inline static Tfpose odom2tf(float x, float y, float yaw) {
        Eigen::Quaternionf quat = 
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())*
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())*
            Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        Tfpose isometry = Eigen::Isometry3f::Identity();
        isometry.linear() = quat.toRotationMatrix();
        isometry.translation() = Eigen::Vector3f(x, y, 0) ;
        return isometry;
    }

    inline static bool readOdom(std::ifstream &is, Tfpose &out, uint64_t &stamp) {
        
        float x, y, yaw;
        if(!is)
            return false;
  
        if(is>>x>>y>>yaw>>stamp) {
            out = odom2tf(x, y, yaw);
            return true;
        }
        return false;
    }

    inline static bool readGPS(std::ifstream &is, Eigen::Vector3d& xyz) {
        
        float x, y, z;
        if(!is)
            return false;

        if(is>>x>>y>>z) {
            xyz << x, y, z;
            std::cout<<"gps_xyz: "<<x <<" "<<y<<" "<<z<<"\n";
            return true;
        }
        return false;
    }

    inline static bool readOdomFile(std::string file_name, Tfpose &odom_out, uint64_t &odom_stamp) {

        std::ifstream is(file_name);
        bool state = true;
        if(is) state = state && readOdom(is, odom_out, odom_stamp);
        return state;
    }

    inline static Vector3<float> tf2pose(Tfpose tf) {
                
        Eigen::Vector3f p = tf.translation(); 
        Vector3<float> out(p[0], p[1], p[2]);
        return out;
    }


    inline static bool isclose(double x, double y, double r_tol=1.e-5, double a_tol=1.e-8) {
        return fabs(x-y) <= a_tol + r_tol * fabs(y);
    }

    inline static Vector3<float> tf2rot(Tfpose tf) {

        double psi, theta, phi;
        Eigen::MatrixXd R = tf.matrix().cast<double>();

        /*
        From a paper by Gregory G. Slabaugh (undated),
        "Computing Euler angles from a rotation matrix
        */
        phi = 0.0;
        if ( isclose(R(2,0),-1.0) ) {
            theta = M_PI/2.0;
            psi = atan2(R(0,1),R(0,2));
        } else if ( isclose(R(2,0),1.0) ) {
            theta = -M_PI/2.0;
            psi = atan2(-R(0,1),-R(0,2));
        } else {
            theta = -asin(R(2,0));
            double cos_theta = cos(theta);
            psi = atan2(R(2,1)/cos_theta, R(2,2)/cos_theta);
            phi = atan2(R(1,0)/cos_theta, R(0,0)/cos_theta);
        }
        
        Vector3<float> out(psi, theta, phi);
        return out;
    }

}}