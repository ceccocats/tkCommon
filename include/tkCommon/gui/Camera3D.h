#pragma once 
#include "tkCommon/common.h"
#include <GLFW/glfw3.h>

namespace tk { namespace gui  {

class Camera3D
{
  public:
    Camera3D();

    //4x4 matrix in col-major format
    const Eigen::Matrix4f &getModelView() const
    {
        return m_modelView;
    }

    //4x4 matrix in col-major format
    const Eigen::Matrix4f &getProjection() const
    {
        return m_projection;
    }

    const Eigen::Vector3f &getEye() const {
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

    void setCenter(Eigen::Vector3f center);
    const Eigen::Vector3f &getCenter() const {
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
        setCenter(Eigen::Vector3f(p.x, p.y, p.z));
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
    tk::common::Vector3<float> project(tk::common::Vector3<float> point);


    tk::common::Vector3<float> getPointOnGround();

    GLFWwindow*     window;
    tk::common::Vector2<float>  screenPos;

    bool mouseOnGUI;


    bool getMouseLeft() {
        return m_mouseLeft;
    }
    bool getMouseRight() {
        return m_mouseRight;
    }

  private:
    Eigen::Matrix4f m_modelView;
    Eigen::Matrix4f m_projection;

    float m_windowAspect; // width/height
    float m_fovRads;
    Eigen::Vector3f m_eye;
    Eigen::Vector3f m_center;
    Eigen::Vector3f m_up;

    float vangleMin = -M_PI/2 + 0.0001;
    float vangleMax = +M_PI/2 - 0.0001;

    float m_zNear;
    float m_zFar;

    float m_radius;
    Eigen::Vector2f m_angles;

    bool m_mouseLeft;
    bool m_mouseRight;
    Eigen::Vector2f m_currentPos;

    Eigen::Vector2f m_startAngles;
    Eigen::Vector3f m_startCenter;

    void updateEye();
    void updateMatrices();
};

}}