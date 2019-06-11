#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include "Eigen/Dense"
#include "geodetic_conv.h"

namespace tk { namespace common {

    /**
     * Vector of 4 elements [ x y z i ]
     * @tparam T
     */
    template <class T> 
    class Vector4 {
        public:
            T x, y, z, i;

            /**
             * init all to zero
             */
            Vector4() {
                x = y = z = i = 0;
            }

            /**
             * init with values
             * @param x
             * @param y
             * @param z
             * @param i
             */
            Vector4(T x, T y, T z, T i) {
                this->x = x;
                this->y = y;
                this->z = z;
                this->i = i;
            }

            ~Vector4() {}

            /**
             * override ostream to a nice print
             * @param os
             * @param v
             * @return
             */
            friend std::ostream& operator<<(std::ostream& os, const Vector4& v) {
                os << "v4(" << v.x <<", "<< v.y <<", "<< v.z <<", "<< v.i <<")";
                return os;
            }  
    };


    /**
     * Vector of 3 elements [ x y z ]
     * @tparam T
     */
    template <class T>
    class Vector3 {
        public:        
            T x, y, z;

            /**
             * init all to zero
             */
            Vector3() {
                x = y = z = 0;
            }

            /**
             * init with values
             * @param x
             * @param y
             * @param z
             */
            Vector3(T x, T y, T z) {
                this->x = x;
                this->y = y;
                this->z = z;
            }

            ~Vector3() {}

            /**
             * override ostream for a nice print
             * @param os
             * @param v
             * @return
             */
            friend std::ostream& operator<<(std::ostream& os, const Vector3& v) {
                os << "v3(" << v.x <<", "<< v.y <<", "<< v.z <<")";
                return os;
            }  
    };

    /**
     * Vector of 2 elements [ x y ]
     * @tparam T
     */
    template <class T>
    class Vector2 {
        public:        
            T x, y;

            /**
             * init all to zero
             */
            Vector2() {
                x = y = 0;
            }

            /**
             * init with values
             * @param x
             * @param y
             */
            Vector2(T x, T y) {
                this->x = x;
                this->y = y;
            }

            ~Vector2() {}

            /**
             * override ostream for a nice print
             * @param os
             * @param v
             * @return
             */
            friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
                os << "v2(" << v.x <<", "<< v.y <<")";
                return os;
            }  
    };

    /**
     *  RotoTranslation transform
     *  it is implemented as a matrix 4x4
     *      r r r x
     *      r r r y
     *      r r r z
     *      0 0 0 1
     *  [ r ]    : 3d rotation matrix
     *  [x y z ] : 3d translation
     */
    typedef Eigen::Isometry3f Tfpose;

    /**
     * Vector of 3x3 Matrix
     * used by GICP
     */
    typedef std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > MatricesVector;

    /**
     * Timestamp value
     * espessed in microseconds from epoch
     */
    typedef uint64_t TimeStamp;

    /**
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param yaw rotation
     * @return transform
     */
    inline Tfpose odom2tf(float x, float y, float yaw) {
        Eigen::Quaternionf quat = 
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())*
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())*
            Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        Tfpose isometry = Eigen::Isometry3f::Identity();
        isometry.linear() = quat.toRotationMatrix();
        isometry.translation() = Eigen::Vector3f(x, y, 0) ;
        return isometry;
    }

    /**
     * Read odometry from ifstream as a TfPose
     * Format:
     * x y yaw stamp
     *
     * @param is input stream
     * @param out output transform
     * @param stamp output timestamp of odometry
     * @return true if correctly readed
     */
    inline bool readOdom(std::ifstream &is, Tfpose &out, uint64_t &stamp) {
        
        float x, y, yaw;
        if(!is)
            return false;
  
        if(is>>x>>y>>yaw>>stamp) {
            out = odom2tf(x, y, yaw);
            return true;
        }
        return false;
    }

    /**
     * Read GPS data from ifstream
     * Format1 [ direct xyz ENU coords ] :
     * x y z
     * Format2 [ gps raw data ] :
     * lat lon height
     * quality hdop n_satellites
     *
     * @param is input stream
     * @param xyz output 3d position
     * @param geoconv input geodetic reference of the map
     * @return true if correcty readed
     */
    inline bool readGPS(std::ifstream &is, Eigen::Vector3d& xyz, GeodeticConverter &geoconv) {
        
        double gpsX, gpsY, gpsH;
        if(!is)
            return false;

        if(is>>gpsX>>gpsY>>gpsH) {
            
            // check if it need to be converted from lat/lon to XY
            double lat, lon, h, hdop;
            int quality, nsat;            
            if(is>>quality>>hdop>>nsat) {
                // data is in LAT/LON
                lat = gpsX;
                lon = gpsY;
                h = gpsH;

                if(lat == 0 || lon == 0)
                    return false;

                // init at first data
                if(!geoconv.isInitialised())
                    geoconv.initialiseReference(lat, lon, h);
            
                geoconv.geodetic2Enu(lat, lon, h, &gpsX, &gpsY, &gpsH);
            } 
            
            // data is in XYZ
            xyz << gpsX, gpsY, gpsH;
            std::cout<<"gps_xyz: "<<gpsX<<" "<<gpsY<<" "<<gpsH<<"\n";
            return true;
        }
        return false;
    }

    /**
     * Read odometry from file
     * @param file_name file path
     * @param odom_out readed odometry
     * @param odom_stamp readed odometry stamp
     * @return true if correctly readed
     */
    inline bool readOdomFile(std::string file_name, Tfpose &odom_out, uint64_t &odom_stamp) {

        std::ifstream is(file_name);
        bool state = true;
        if(is) state = state && readOdom(is, odom_out, odom_stamp);
        return state;
    }

    /**
     * Extract 3d translation from TfPose
     * @param tf
     * @return
     */
    inline Vector3<float> tf2pose(Tfpose tf) {
                
        Eigen::Vector3f p = tf.translation(); 
        Vector3<float> out(p[0], p[1], p[2]);
        return out;
    }


    /**
     * check if x is close to y
     * formula: fabs(x-y) <= a_tol + r_tol * fabs(y)
     * @param x x value
     * @param y y value
     * @param r_tol
     * @param a_tol
     * @return
     */
    inline bool isclose(double x, double y, double r_tol=1.e-5, double a_tol=1.e-8) {
        return fabs(x-y) <= a_tol + r_tol * fabs(y);
    }

    /**
     * Extract 3d rotation from TfPose
     * @param tf
     * @return
     */
    inline Vector3<float> tf2rot(Tfpose tf) {

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

    inline Eigen::Vector3f unproj_3d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
                                     float screenW, float screenH, Eigen::Vector3f pos) {

        Eigen::Matrix4f invProjView = (p_mat*v_mat).inverse();
        float viewX = 0, viewY = 0; // screen pos
        float viewWidth  = screenW;
        float viewHeight = screenH;

        float x = pos(0);
        float y = pos(1);
        float z = pos(2);

        x = x - viewX;
        y = viewHeight - y - 1;
        y = y - viewY;

        Eigen::Vector4f in;
        in(0) = (2 * x) / viewWidth - 1;
        in(1) = (2 * y) / viewHeight - 1;
        in(2) = 2 * z - 1;
        in(3) = 1.0;

        Eigen::Vector4f out = invProjView*in;

        float w = 1.0 / out[3];
        out(0) *= w;
        out(1) *= w;
        out(2) *= w;

        pos << out(0), out(1), out(2);
        return pos;
    }


    inline Eigen::Vector2f proj_2d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
                                   float screenW, float screenH, Eigen::Vector4f pos) {

        pos = v_mat*pos;
        pos = p_mat*pos;

        Eigen::Vector2f p;
        p(0) = pos(0) / pos(3);
        p(1) = -pos(1) / pos(3);
        p(0) = (p(0) + 1) * screenW / 2;
        p(1) = (p(1) + 1) * screenH / 2;
        return p;
    }


}}
