#include "tkCommon/gui/MouseView3D.h"
#include "tkCommon/gui/MathUtils.h"

#include <math.h>
#include "tkCommon/common.h"
#include <Eigen/Dense>

namespace tk { namespace gui  {

MouseView3D::MouseView3D()
    : m_windowAspect(1.0f)
    , m_fovRads(DEG2RAD(60.0f))
    , m_zNear(0.1f)
    , m_zFar(10000.f)
    , m_radius(25)
    , m_mouseLeft(false)
    , m_mouseRight(false)
{
    m_center[0] = 0;
    m_center[1] = 0;
    m_center[2] = 0.0;

    m_up[0] = 0;
    m_up[1] = 0;
    m_up[2] = 1;

    m_angles[0] = DEG2RAD(-90.0f);
    m_angles[1] = DEG2RAD(45.0f);

    m_currentPos[0] = 0.0f;
    m_currentPos[1] = 0.0f;

    updateEye();
    updateMatrices();
}

void MouseView3D::updateEye()
{
    m_eye[0] = m_radius * cos(m_angles[1]) * cos(m_angles[0]) + m_center[0];
    m_eye[1] = m_radius * cos(m_angles[1]) * sin(m_angles[0]) + m_center[1];
    m_eye[2] = m_radius * sin(m_angles[1]) + m_center[2];
}

void MouseView3D::mouseDown(int button, float x, float y)
{
    if(mouseOnGUI)
        return;

    m_currentPos[0] = x;
    m_currentPos[1] = y;

    m_startAngles[0] = m_angles[0];
    m_startAngles[1] = m_angles[1];

    tk::common::Vector3<float> C = getPointOnGround();
    m_startCenter[0] = C.x;
    m_startCenter[1] = C.y;
    m_startCenter[2] = C.z;

    m_mouseLeft  = (button == 0);
    m_mouseRight = (button == 1);
}

void MouseView3D::mouseUp(int button, float x, float y)
{
    m_mouseLeft  = false;
    m_mouseRight = false;
}

void MouseView3D::mouseMove(float x, float y)
{
    if(mouseOnGUI)
        return;

    float pos[] = {x, y};
    screenPos = {x,y};

    if (m_mouseLeft) {
        // update deltaAngle
        m_angles[0] = m_startAngles[0] - 0.01f * (pos[0] - m_currentPos[0]);
        m_angles[1] = m_startAngles[1] + 0.01f * (pos[1] - m_currentPos[1]);

        // Limit the vertical angle (5 to 85 degrees)
        if (m_angles[1] > DEG2RAD(85))
            m_angles[1] = DEG2RAD(85);

        if (m_angles[1] < DEG2RAD(-85))
            m_angles[1] = DEG2RAD(-85);

        updateEye();
        updateMatrices();

    } else if (m_mouseRight) {
        tk::common::Vector3<float> C = getPointOnGround();
        m_center[0] += m_startCenter[0] - C.x;
        m_center[1] += m_startCenter[1] - C.y;
        m_center[2] = 0;

        updateEye();
        updateMatrices();
    }
}

void MouseView3D::mouseWheel(float dx, float dy)
{
    if(mouseOnGUI)
        return;

    float tmpRadius = m_radius - dy * 10.0f;

    if (tmpRadius > 0.0f) {
        m_radius = tmpRadius;
        updateEye();
    }
}

void MouseView3D::setWindowAspect(float aspect)
{
    m_windowAspect = aspect;
    updateMatrices();
}

void MouseView3D::setCenter(float x, float y, float z)
{
    m_center[0] = x;
    m_center[1] = y;
    m_center[2] = z;
    updateEye();
    updateMatrices();
}

void MouseView3D::updateMatrices()
{
    lookAt(m_modelView.data(), m_eye, m_center, m_up);
    perspective(m_projection.data(), m_fovRads, 1.0f * m_windowAspect, m_zNear, m_zFar);
}

tk::common::Vector3<float> MouseView3D::unproject(float zplane) {


    Eigen::Matrix4f proj(m_projection);
    Eigen::Matrix4f view(m_modelView);
    Eigen::Vector3f pos;
    pos << screenPos.x, screenPos.y, zplane;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    pos = tk::common::unproj_3d(proj, view, width, height, pos);

    tk::common::Vector3<float> p;
    p.x = pos(0);
    p.y = pos(1);
    p.z = pos(2);
    return p;
}

tk::common::Vector2<float> MouseView3D::project(tk::common::Vector3<float> p3d) {

    Eigen::Matrix4f proj(m_projection);
    Eigen::Matrix4f view(m_modelView);
    Eigen::Vector4f point;
    point << p3d.x, p3d.y, p3d.z, 1;
    
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    Eigen::Vector2f pos = tk::common::proj_2d(proj, view, width, height, point);

    tk::common::Vector2<float> p;
    p.x = pos(0);
    p.y = pos(1);
    return p;
}

tk::common::Vector3<float> MouseView3D::getPointOnGround() {

    tk::common::Vector3<float> A =  unproject(0.0);
    tk::common::Vector3<float> B =  unproject(1.0);
    float m1 = (A.z - B.z) / (A.x - B.x); 
    float q1 = (A.z - m1*A.x); 
    float m2 = (A.z - B.z) / (A.y - B.y); 
    float q2 = (A.z - m2*A.y);
    tk::common::Vector3<float> C = { -q1/m1, -q2/m2, 0 };
    return C;
}

}}