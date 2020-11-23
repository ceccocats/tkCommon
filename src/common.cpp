#include <tkCommon/common.h>

const char* tkCommon_PATH = TKPROJ_PATH;

namespace tk { namespace common {

    Tfpose odom2tf(float x, float y, float z, float roll, float pitch, float yaw) {
        Eigen::MatrixXf t(4,4);
        float A = std::cos (yaw),  B = sin (yaw),  C  = std::cos (pitch), D  = sin (pitch),
              E = std::cos (roll), F = sin (roll), DE = D*E,         DF = D*F;

        t (0, 0) = A*C;  t (0, 1) = A*DF - B*E;  t (0, 2) = B*F + A*DE;  t (0, 3) = x;
        t (1, 0) = B*C;  t (1, 1) = A*E + B*DF;  t (1, 2) = B*DE - A*F;  t (1, 3) = y;
        t (2, 0) = -D;   t (2, 1) = C*F;         t (2, 2) = C*E;         t (2, 3) = z;
        t (3, 0) = 0;    t (3, 1) = 0;           t (3, 2) = 0;           t (3, 3) = 1;
        
        Tfpose isometry;
        isometry.matrix() = t;
        return isometry;
    }

    Tfpose odom2tf(float x, float y, float yaw) {
        return odom2tf(x,y,0,0,0,yaw);
    }

    Tfpose odom2tf(float x, float y, float z, float yaw) {
        return odom2tf(x,y,z,0,0,yaw);
    }

    Tfpose odom2tf(float x, float y, float z, float qx, float qy, float qz, float qw) {
        Eigen::Quaternionf quat(qx, qy, qz, qw);
        Tfpose isometry = Eigen::Isometry3f::Identity();
        isometry.linear() = quat.toRotationMatrix();
        isometry.translation() = Eigen::Vector3f(x, y, z) ;
        return isometry;
    }

    bool readOdom(std::ifstream &is, Tfpose &out, uint64_t &stamp) {

        float x, y, yaw;
        if(!is)
            return false;

        if(is>>x>>y>>yaw>>stamp) {
            out = odom2tf(x, y, yaw);
            return true;
        }
        return false;
    }

    bool readGPS(std::ifstream &is, Eigen::Vector3d& xyz, GeodeticConverter &geoconv) {

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

    bool readOdomFile(std::string file_name, Tfpose &odom_out, uint64_t &odom_stamp) {

        std::ifstream is(file_name);
        bool state = true;
        if(is) state = state && readOdom(is, odom_out, odom_stamp);
        return state;
    }

    template <typename T, int ROWS>
    Eigen::Matrix<T, ROWS, 1> Interpolate_(const Eigen::Matrix<T, ROWS, 1>& v1,
                                          const Eigen::Matrix<T, ROWS, 1>& v2, const double ratio) {
        // Safety check sizes
        tkASSERT(v1.size() == v2.size(), "Vectors v1 and v2 must be the same size")
        
        // Interpolate
        // This is the numerically stable version, rather than  (p1 + (p2 - p1) * real_ratio)
        return ((v1 * (1.0 - ratio)) + (v2 * ratio));
    }

    inline Eigen::Quaternionf Interpolate_(const Eigen::Quaternionf& q1,
                                           const Eigen::Quaternionf& q2, const double ratio)
    {
        // Interpolate
        return q1.slerp(ratio, q2);
    }

    Eigen::Isometry3f tfInterpolate(const Eigen::Isometry3f& t1,
                                    const Eigen::Isometry3f& t2, const double ratio) {
        // Safety check ratio
        tkASSERT(ratio >= 0.0, "interpolation ratio not in 0-1 range: " + std::to_string(ratio));
        tkASSERT(ratio <= 1.0, "interpolation ratio not in 0-1 range: " + std::to_string(ratio));
        
        // Interpolate
        const Eigen::Vector3f v1 = t1.translation();
        const Eigen::Quaternionf q1(t1.rotation());
        const Eigen::Vector3f v2 = t2.translation();
        const Eigen::Quaternionf q2(t2.rotation());
        const Eigen::Vector3f vint = Interpolate_(v1, v2, ratio);
        const Eigen::Quaternionf qint = Interpolate_(q1, q2, ratio);
        const Eigen::Isometry3f tint = ((Eigen::Translation3f)vint) * qint;
        return tint;
    }


    Vector3<float> tf2pose(Tfpose tf) {

        Eigen::Vector3f p = tf.translation();
        Vector3<float> out(p[0], p[1], p[2]);
        return out;
    }

    bool isclose(double x, double y, double r_tol, double a_tol) {
        return fabs(x-y) <= a_tol + r_tol * fabs(y);
    }

    Vector3<float> tf2rot(Tfpose tf) {

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

    Eigen::Vector3f unproj_3d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
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

    Eigen::Vector2f proj_2d(Eigen::Matrix4f p_mat, Eigen::Matrix4f v_mat,
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

    tk::common::Tfpose planeCoeffs2tf(Eigen::VectorXf coeffs) {
        if(coeffs.size() != 4)
            return tk::common::Tfpose::Identity();

        // a*x + b*y + c*z + d = 0
        double a = coeffs(0);
        double b = coeffs(1);
        double c = coeffs(2);
        double d = coeffs(3);

        tk::common::Vector3<double> rot;
        tk::common::Vector3<double> p0, p1;

        p0.x = 0; p0.y = -50; p0.z = -(p0.x*a +p0.y*b +d)/c;
        p1.x = 0; p1.y = +50; p1.z = -(p1.x*a +p1.y*b +d)/c;
        rot.x = atan2(p0.z - p1.z, p0.y - p1.y);
        if(rot.x != rot.x)
            rot.x =0;

        p0.x = -50; p0.y = 0; p0.z = -(p0.x*a +p0.y*b +d)/c;
        p1.x = +50; p1.y = 0; p1.z = -(p1.x*a +p1.y*b +d)/c;
        rot.y = atan2(p0.z - p1.z, p0.x - p1.x);
        if(rot.y != rot.y)
            rot.y =0;

        rot.z =0;

        double h = -d/c;
        if(h != h)
            h =0;

        return tk::common::odom2tf(0, 0, h, rot.x, rot.y, rot.z);
    }
}}