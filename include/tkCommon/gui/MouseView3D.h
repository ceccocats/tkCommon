#pragma once 
#include "tkCommon/common.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace tk { namespace gui  {

class MouseView3D
{
  public:
    MouseView3D();

    //4x4 matrix in col-major format
    const glm::mat4x4 &getModelView() const
    {
        return m_modelView;
    }

    //4x4 matrix in col-major format
    const glm::mat4x4 &getProjection() const
    {
        return m_projection;
    }

    const glm::vec3 &getEye() const {
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

    void setCenter(glm::vec3 center);
    const glm::vec3 &getCenter() const {
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
        setCenter({p.x, p.y, p.z});
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
    glm::mat4x4 m_modelView;
    glm::mat4x4 m_projection;

    float m_windowAspect; // width/height
    float m_fovRads;
    glm::vec3 m_eye;
    glm::vec3 m_center;
    glm::vec3 m_up;

    float vangleMin = -M_PI/2 + 0.0001;
    float vangleMax = +M_PI/2 - 0.0001;

    float m_zNear;
    float m_zFar;

    float m_radius;
    glm::vec2 m_angles;

    bool m_mouseLeft;
    bool m_mouseRight;
    glm::vec2 m_currentPos;

    glm::vec2 m_startAngles;
    glm::vec3 m_startCenter;

    void updateEye();
    void updateMatrices();
};

}}