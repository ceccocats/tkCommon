#include "tkCommon/gui/Camera.h"
#include "tkCommon/math/quantile.h"
namespace tk { namespace gui {

void Camera::init() {
    // window
    fov   = 60.0f/180.0f*M_PI;
    zNear =     0.1f;
    zFar  = 10000.0f;
    zoom  = 25.0f;

    // center of scene position
    center = glm::vec3(0,0,0);

    // axes 
    up = glm::vec3(0,0,1);

    // start angles
    angles[0] = (-90.0f)/180.0f*M_PI;
    angles[1] = (45.0f )/180.0f*M_PI;

    // mouse status
    mouseOnGUI = false;
    mouseLeft  = false;
    mouseRight = false;
    mousePos     = glm::vec2(0);
    lastMousePos = glm::vec2(0);
    lastAngles   = glm::vec2(0);
    lastCenter   = glm::vec3(0);

    // update matrix
    updateEye();
    updateMatrices();
}

void Camera::updateEye() {
    eye.x = zoom * cos(angles[1]) * cos(angles[0]) + center[0];
    eye.y = zoom * cos(angles[1]) * sin(angles[0]) + center[1];
    eye.z = zoom * sin(angles[1]) + center[2];
}

void Camera::updateMatrices() {
    float aspect = float(viewport[2]) / float(viewport[3]);
    modelView  = glm::lookAt(eye, center, up);
    projection = glm::perspective(fov, aspect, zNear, zFar);
}

void Camera::setViewPort(int x, int y, int width, int height) {
    viewport = glm::ivec4(x,y,width,height);
    updateEye();
    updateMatrices();
}

void Camera::setCenter(tk::common::Vector3<float> p) {
    center.x = tk::math::lerp<float>(center.x, p.x, 0.1);
    center.y = tk::math::lerp<float>(center.y, p.y, 0.1);
    center.z = tk::math::lerp<float>(center.z, p.z, 0.1);
}


void Camera::mouseDown(int button, float x, float y) {
    if(mouseOnGUI)
        return;

    lastMousePos = glm::vec2(x, y);
    lastAngles = angles;
    lastCenter = unprojectPlane(lastMousePos);

    if(button == 0) mouseLeft  = true;
    if(button == 1) mouseRight = true;
}

void Camera::mouseUp(int button, float x, float y) {
    if(button == 0) mouseLeft  = false;
    if(button == 1) mouseRight = false;
}

void Camera::mouseMove(float x, float y) {
    if(mouseOnGUI)
        return;

    mousePos = glm::vec2(x,y);

    if (mouseLeft) {
        // update deltaAngle
        angles[0] = lastAngles[0] - 0.01f * (mousePos[0] - lastMousePos[0]);
        angles[1] = lastAngles[1] + 0.01f * (mousePos[1] - lastMousePos[1]);

        // Limit the vertical angle 
        if (angles[1] > vangleMax)
            angles[1] = vangleMax;

        if (angles[1] < vangleMin)
            angles[1] = vangleMin;

        updateEye();
        updateMatrices();

    } else if (mouseRight) {
        glm::vec3 C = unprojectPlane(mousePos);
        center[0] += lastCenter[0] - C.x;
        center[1] += lastCenter[1] - C.y;
        center[2]  = C.z;

        updateEye();
        updateMatrices();
    }
}

void Camera::mouseWheel(float dx, float dy) {
    if(mouseOnGUI)
        return;
    
    float tmpZoom = zoom - dy * 10.0f;
    if (tmpZoom > 0.0f) {
        zoom = tmpZoom;
        updateEye();
        updateMatrices();
    }
}

glm::vec3 Camera::project(glm::vec3 obj) {
    return glm::project(obj, modelView, projection, viewport);
}

glm::vec3 Camera::unprojectPlane(glm::vec2 pos) {
    pos.y = viewport[3] - pos.y;
    const glm::vec3 A = glm::unProject( 
        glm::vec3(pos.x, pos.y, 0.0), modelView, projection, viewport);
    const glm::vec3 B = glm::unProject( 
        glm::vec3(pos.x, pos.y, 1.0), modelView, projection, viewport);

    float m1 = (A.z - B.z) / (A.x - B.x); 
    float q1 = (A.z - m1*A.x); 
    float m2 = (A.z - B.z) / (A.y - B.y); 
    float q2 = (A.z - m2*A.y);
    return glm::vec3(-q1/m1, -q2/m2, 0);
}


}}

