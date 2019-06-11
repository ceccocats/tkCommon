#pragma once 
#include "tkCommon/common.h"
#include <GLFW/glfw3.h>

namespace tk { namespace gui  {

class MouseView3D
{
  public:
    MouseView3D();

    //4x4 matrix in col-major format
    const Eigen::Matrix4f *getModelView() const
    {
        return &m_modelView;
    }

    //4x4 matrix in col-major format
    const Eigen::Matrix4f *getProjection() const
    {
        return &m_projection;
    }

    const float *getEye() const {
        return m_eye;
    }

    void setWindowAspect(float aspect);

    void setFov(float fovRads) {
        m_fovRads = fovRads;
    }

    void mouseDown(int button, float x, float y);
    void mouseUp(int button, float x, float y);
    void mouseMove(float x, float y);
    void mouseWheel(float dx, float dy);

    void setCenter(float x, float y, float z);
    const float* getCenter() const {
        return m_center;
    }

    tk::common::Vector3<float> getWorldPos() {
        tk::common::Vector3<float> p;
        p.x = m_center[0];
        p.y = m_center[1];
        p.z = m_center[2];
        return p;
    }
    void setWorldPos(tk::common::Vector3<float> p) {
        setCenter(p.x, p.y, p.z);
    }
    void setAngle(tk::common::Vector3<float> a) {
        m_angles[0] = a.z;
        m_angles[1] = a.y;
    }
    tk::common::Vector3<float> getAngle() {
        tk::common::Vector3<float> a;
        a.z = m_angles[0];
        a.y = m_angles[1];
        a.x = 0;
        return a;
    }
    tk::common::Vector3<float> unproject(float zplane);
    tk::common::Vector2<float> project(tk::common::Vector3<float> point);


    tk::common::Vector3<float> getPointOnGround();

    GLFWwindow*     window;
    tk::common::Vector2<float>  screenPos;

  private:
    Eigen::Matrix4f m_modelView;
    Eigen::Matrix4f m_projection;

    float m_windowAspect; // width/height
    float m_fovRads;
    float m_center[3];
    float m_up[3];
    float m_eye[3];

    float m_zNear;
    float m_zFar;

    // MOUSE NAVIGATION VARIABLES
    float m_startAngles[2];
    float m_startCenter[3];

    float m_radius;
    float m_angles[2];

    bool m_mouseLeft;
    bool m_mouseRight;
    float m_currentPos[2];

    void updateEye();
    void updateMatrices();

    bool m_request_pose_insert = false;
};

}}