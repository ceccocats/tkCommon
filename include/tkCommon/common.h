#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <iomanip>
#include "utils.h"
#include "timer.h"
#include "Eigen/Dense"
#include "geodetic_conv.h"

namespace tk { namespace common {

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

    typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXu8;

    /**
     * Rect [ x y w h ]
     * @tparam T
     */
    template <class T> 
    class Rect {
        public:
            T x, y, w, h;

            /**
             * init all to zero
             */
            Rect() {
                x = y = w = h = 0;
            }

            /**
             * init with values
             * @param x
             * @param y
             * @param z
             * @param i
             */
            Rect(T x, T y, T w, T h) {
                this->x = x;
                this->y = y;
                this->w = w;
                this->h = h;
            }

            ~Rect() {}

            /**
             * override ostream to a nice print
             * @param os
             * @param v
             * @return
             */
            friend std::ostream& operator<<(std::ostream& os, const Rect& v) {
                os << "Rect(" << v.x <<", "<< v.y <<", "<< v.w <<", "<< v.h <<")";
                return os;
            }  
    };

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

            T dist(Vector3<T> v) {
                T dx = x-v.x;
                T dy = y-v.y;
                T dz = z-v.z;
                return sqrt(dx*dx + dy*dy + dz*dz);
            }

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
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param yaw rotation
     * @return transform
     */
    inline Tfpose odom2tf(float x, float y, float z, float yaw) {
        Eigen::Quaternionf quat = 
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())*
            Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())*
            Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        Tfpose isometry = Eigen::Isometry3f::Identity();
        isometry.linear() = quat.toRotationMatrix();
        isometry.translation() = Eigen::Vector3f(x, y, z) ;
        return isometry;
    }

    /**
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param z   translation upward
     * @param roll  rotation x axle
     * @param pitch rotation y axle
     * @param yaw   rotation z axle
     * @return transform
     */
    inline Tfpose odom2tf(float x, float y, float z, float roll, float pitch, float yaw) {
        Eigen::Quaternionf quat =
                Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())*
                Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())*
                Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        Tfpose isometry = Eigen::Isometry3f::Identity();
        isometry.linear() = quat.toRotationMatrix();
        isometry.translation() = Eigen::Vector3f(x, y, z) ;
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

    /**
     * project 2d point to 3d point
     * @param p_mat projection matrix
     * @param v_mat model view matrix
     * @param screenW
     * @param screenH
     * @param pos expressed as (x, y, zplane)
     * @return 3d point
     */
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


    /**
     * Project 3d point to 2d screen coordinate
     * @param p_mat projection matrix
     * @param v_mat model view matrix
     * @param screenW
     * @param screenH
     * @param pos expressed as (x, y, z, 1)
     * @return 2d screen position
     */
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


    /**
     * dump memory as hexdump
     * @param os
     * @param buffer
     * @param bufsize
     * @param showPrintableChars
     * @return
     */
    inline std::ostream& hex_dump(std::ostream& os, const void *buffer,
                                  std::size_t bufsize, bool showPrintableChars = true) {
        if (buffer == nullptr) {
            return os;
        }
        auto oldFormat = os.flags();
        auto oldFillChar = os.fill();
        constexpr std::size_t maxline{8};
        // create a place to store text version of string
        char renderString[maxline+1];
        char *rsptr{renderString};
        // convenience cast
        const unsigned char *buf{reinterpret_cast<const unsigned char *>(buffer)};

        for (std::size_t linecount=maxline; bufsize; --bufsize, ++buf) {
            os << std::setw(2) << std::setfill('0') << std::hex
               << static_cast<unsigned>(*buf) << ' ';
            *rsptr++ = std::isprint(*buf) ? *buf : '.';
            if (--linecount == 0) {
                *rsptr++ = '\0';  // terminate string
                if (showPrintableChars) {
                    os << " | " << renderString;
                }
                os << '\n';
                rsptr = renderString;
                linecount = std::min(maxline, bufsize);
            }
        }
        // emit newline if we haven't already
        if (rsptr != renderString) {
            if (showPrintableChars) {
                for (*rsptr++ = '\0'; rsptr != &renderString[maxline+1]; ++rsptr) {
                    os << "   ";
                }
                os << " | " << renderString;
            }
            os << '\n';
        }

        os.fill(oldFillChar);
        os.flags(oldFormat);
        return os;
    }

    /**
     * Serialize an eigen matrix in binary
     * @tparam T data type
     * @tparam R number of rows (-1 is dynamic)
     * @tparam C number of cols (-1 is dynamic)
     * @param m
     * @param os
     * @return
     */
    template <class T, int R=-1, int C=-1>
    inline bool serializeMatrix(Eigen::Matrix<T, R, C> &m, std::ofstream &os) {
        int size[2];
        size[0] = m.rows();
        size[1] = m.cols();
        std::cout<<"Matrix serialize: ("<<size[0]<<"x"<<size[1]<<")";

        os.write((char *)size, 2*sizeof(int));
        os.write((char *)m.data(), m.size() * sizeof(T));
        return os.is_open();
    }

    /**
     * Deserialize an eigen matrix in binary
     * @tparam T data type
     * @tparam R number of rows (-1 is dynamic)
     * @tparam C number of cols (-1 is dynamic)
     * @param m
     * @param os
     * @return
     */
    template <class T, int R=-1, int C=-1>
    inline bool deserializeMatrix(Eigen::Matrix<T, R, C> &m, std::ifstream &is) {
        int size[2] = { 0, 0 };
        is.read((char*)size, 2*sizeof(int));
        std::cout<<"Matrix deserialize: ("<<size[0]<<"x"<<size[1]<<")";
        m.resize(size[0], size[1]);
        is.read((char *)m.data(), m.size() * sizeof(T));
        return is.is_open();
    }

    /**
     * Tell if point c is on the left side respect to the segment a - b
     * @tparam T
     * @param a
     * @param b
     * @param c
     * @return
     */
    template <class T>
    bool pointIsleft(tk::common::Vector2<T> a, tk::common::Vector2<T> b, tk::common::Vector2<T> c) {
        return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0;
    }

    /**
     * Tell if point c is on the left side respect to the segment a - b
     * It is a 2d computation even with vector3
     * @tparam T
     * @param a
     * @param b
     * @param c
     * @return
     */
    template <class T>
    bool pointIsleft(tk::common::Vector3<T> a, tk::common::Vector3<T> b, tk::common::Vector3<T> c) {
        return ((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)) > 0;
    }
}}
