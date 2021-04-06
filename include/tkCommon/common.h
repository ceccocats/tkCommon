#pragma once

#include <iostream>
#include <vector>
#include <unordered_map>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <csignal>

#undef Success // defined by X11 cause conflicts with Eigen
#include <Eigen/Dense>

#undef None // defined by X11 cause conflicts with YAML
#include "tkCommon/time.h"
#include "tkCommon/utils.h"
#include "tkCommon/geodetic_conv.h"
#include "tkCommon/CmdParser.h"
#include "tkCommon/exceptions.h"
#include "tkCommon/Map.h"

extern const char* tkCommon_PATH;
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
     * Convert odometry to TfPose
     * @param x   translation forward
     * @param y   translation sideways
     * @param z   translation upward
     * @param qx  rotation quaternion x 
     * @param qy  rotation quaternion y 
     * @param qz  rotation quaternion z 
     * @param qw  rotation quaternion w 
     * @return transform
     */
    Tfpose odom2tf(float x, float y, float z, float qx, float qy, float qz, float qw);


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


    Eigen::Isometry3f tfInterpolate(const Eigen::Isometry3f& t1,
                                    const Eigen::Isometry3f& t2, const double ratio);


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
        tkDBG("Loading YAML: "<<conf_file<<"\n");
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

namespace tk { namespace math {
    template<class T> class Vec2;
    template<class T> class Vec3;
    template<class T> class Vec4;
}}

namespace tk { namespace common {

    template<typename T>
    using Vector2 = tk::math::Vec2<T>;
    template<typename T>
    using Vector3 = tk::math::Vec3<T>;
    template<typename T>
    using Vector4 = tk::math::Vec4<T>;

    /**
     * Extract 3d translation from TfPose
     * @param tf
     * @return
     */
    Vector3<float> tf2pose(Tfpose tf);
    /**
     * Extract 3d rotation from TfPose
     * @param tf
     * @return
     */
    Vector3<float> tf2rot(Tfpose tf);


    /**
     * Tell if point c is on the left side respect to the segment a - b
     * @tparam T
     * @param a
     * @param b
     * @param c
     * @return
     */
    template <class T>
    inline bool pointIsleft(tk::common::Vector2<T> a, tk::common::Vector2<T> b, tk::common::Vector2<T> c) {
        return ((b.x() - a.x())*(c.y() - a.y()) - (b.y() - a.y())*(c.x() - a.x())) > 0;
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
    inline bool pointIsleft(tk::common::Vector3<T> a, tk::common::Vector3<T> b, tk::common::Vector3<T> c) {
        return ((b.x() - a.x())*(c.y() - a.y()) - (b.y() - a.y())*(c.x() - a.x())) > 0;
    }

}}


#include "tkCommon/math/Mat.h"
#include "tkCommon/math/Vec.h"