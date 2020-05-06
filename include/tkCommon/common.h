#pragma once

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <csignal>

#undef Success // defined by X11 cause conflicts with Eigen
#include <Eigen/Dense>

#undef None // defined by X11 cause conflicts with YAML
#include "tkCommon/utils.h"
#include "tkCommon/timer.h"
#include "tkCommon/geodetic_conv.h"
#include "tkCommon/CmdParser.h"
#include "tkCommon/terminalFormat.h"
#include "tkCommon/exceptions.h"

namespace tk { namespace common {

    /**
     * retrive current TK version from GIT repo
     * @return version hash string
     */
    std::string tkVersionGit();

    /**
     * convert version hash string to interger
     */
    int tkVersion2Int(std::string version);
    /**
     * convert version from interger to hash string
     */
    std::string tkVersion2String(int version);

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
     * check if a string end with another string
     * @param mainStr
     * @param toMatch
     * @return
     */
    inline bool endsWith(const std::string &mainStr, const std::string &toMatch) {
        if(mainStr.size() >= toMatch.size() &&
           mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
            return true;
        else
            return false;
    }
    

    /**
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param yaw rotation
     * @return transform
     */
    Tfpose odom2tf(float x, float y, float yaw);

    /**
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param yaw rotation
     * @return transform
     */
    Tfpose odom2tf(float x, float y, float z, float yaw);

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
    Tfpose odom2tf(float x, float y, float z, float roll, float pitch, float yaw);


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

            bool isZero() { return x == 0 && y == 0 && z == 0 && i == 0; } 
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

            T dist2(Vector3<T> v) {
                T dx = x-v.x;
                T dy = y-v.y;
                T dz = z-v.z;
                return dx*dx + dy*dy + dz*dz;
            }

            T dist(Vector3<T> v) {
                T dx = x-v.x;
                T dy = y-v.y;
                T dz = z-v.z;
                return sqrt(dx*dx + dy*dy + dz*dz);
            }

            T dist2_2d(Vector3<T> v) {
                T dx = x-v.x;
                T dy = y-v.y;
                return dx*dx + dy*dy;
            }

            T dist_2d(Vector3<T> v) {
                T dx = x-v.x;
                T dy = y-v.y;
                return sqrt(dx*dx + dy*dy);
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

            bool isZero() { return x == 0 && y == 0 && z == 0; }

            void applyTf(tk::common::Tfpose tf) {
                tk::common::Tfpose ptf = tk::common::odom2tf(x, y, z, 0, 0, 0);
                ptf = tf * ptf;
                x = ptf.matrix()(0, 3);
                y = ptf.matrix()(1, 3);
                z = ptf.matrix()(2, 3);
                return;
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

            bool isZero() { return x == 0 && y == 0; } 
    };

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
    bool readOdom(std::ifstream &is, Tfpose &out, uint64_t &stamp);

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
    bool readGPS(std::ifstream &is, Eigen::Vector3d& xyz, GeodeticConverter &geoconv);

    /**
     * Read odometry from file
     * @param file_name file path
     * @param odom_out readed odometry
     * @param odom_stamp readed odometry stamp
     * @return true if correctly readed
     */
    bool readOdomFile(std::string file_name, Tfpose &odom_out, uint64_t &odom_stamp);

    /**
     * Extract 3d translation from TfPose
     * @param tf
     * @return
     */
    Vector3<float> tf2pose(Tfpose tf);


    /**
     * check if x is close to y
     * formula: fabs(x-y) <= a_tol + r_tol * fabs(y)
     * @param x x value
     * @param y y value
     * @param r_tol
     * @param a_tol
     * @return
     */
    bool isclose(double x, double y, double r_tol=1.e-5, double a_tol=1.e-8);

    /**
     * Extract 3d rotation from TfPose
     * @param tf
     * @return
     */
    Vector3<float> tf2rot(Tfpose tf);

    /**
     * project 2d point to 3d point
     * @param p_mat projection matrix
     * @param v_mat model view matrix
     * @param screenW
     * @param screenH
     * @param pos expressed as (x, y, zplane)
     * @return 3d point
     */
    Eigen::Vector3f unproj_3d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
                                     float screenW, float screenH, Eigen::Vector3f pos);


    /**
     * Project 3d point to 2d screen coordinate
     * @param p_mat projection matrix
     * @param v_mat model view matrix
     * @param screenW
     * @param screenH
     * @param pos expressed as (x, y, z, 1)
     * @return 2d screen position
     */
    Eigen::Vector2f proj_2d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
                                   float screenW, float screenH, Eigen::Vector4f pos);


    /**
     * dump memory as hexdump
     * @param os
     * @param buffer
     * @param bufsize
     * @param showPrintableChars
     * @return
     */
    std::ostream& hex_dump(std::ostream& os, const void *buffer,
                                  std::size_t bufsize, bool showPrintableChars = true);

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
        //std::cout<<"Matrix serialize: ("<<size[0]<<"x"<<size[1]<<")";

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
        //std::cout<<"Matrix deserialize: ("<<size[0]<<"x"<<size[1]<<")\n";
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

    /**
     * Calculate tf from plane coeffs
     * @param coeffs [a,b,c,d] from eq: a*x + b*y + c*z + d = 0
     * @return
     */
    tk::common::Tfpose planeCoeffs2tf(Eigen::VectorXf coeffs);


    /**
     * Load YAML node from file
     * @param conf_file
     * @return
     */
    inline YAML::Node YAMLloadConf(std::string conf_file) {
        return YAML::LoadFile(conf_file);
    }

    /**
     * Get configuration from YAML node
     * @tparam T
     * @param conf yaml node
     * @param key configuration KEY
     * @param defaultVal defalt value in case of no KEY found
     * @return conf value
     */
    template<typename T>
    inline T YAMLgetConf(YAML::Node conf, std::string key, T defaultVal) {
        T val = defaultVal;
        if(conf && conf[key]) {
            val = conf[key].as<T>();
        }
        //std::cout<<"YAML "<<key<<", val: "<<val<<"\n";
        return val;
    }

    /**
     * @brief Read TF from yaml NODE, tf format [ x y z poll pitch yaw ] ( mt mt mt deg deg deg )
     * 
     * @param conf 
     * @return std::vector<tk::common::Tfpose> 
     */
    inline std::vector<tk::common::Tfpose> YAMLreadTf(YAML::Node conf) {

        std::vector<tk::common::Tfpose> tf;

        int size = conf.size();
        if(!conf[0].IsSequence())
            size = 1;

        std::vector<float> tmp;
        tf.resize(size);
        for (int i = 0; i < size; i++) {
            if(conf[i].IsSequence())
                tmp = conf[i].as<std::vector<float>>();
            else
                tmp = conf.as<std::vector<float>>();
            tf[i] = tk::common::odom2tf(tmp[0], tmp[1], tmp[2], toRadians(tmp[3]), toRadians(tmp[4]), toRadians(tmp[5]));

            tmp.clear();
        }

        return tf;
    }

    inline tk::common::Tfpose geodetic2tf(GeodeticConverter &geoconv, double lat, double lon, double height, double roll, double pitch, double yaw) {
        double x, y, z;
        geoconv.geodetic2Enu(lat, lon, height, &x, &y, &z);
        return tk::common::odom2tf(x,y,z,roll,pitch,yaw);
    }
}}
