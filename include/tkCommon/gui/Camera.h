#pragma once 
#include "tkCommon/common.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace tk { namespace gui  {

class Camera
{
  public:
    float windowAspect; // width/height
    float fov;
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;

    glm::mat4 modelView;
    glm::mat4 projection;

    float vangleMin = -M_PI/2 + 0.0001;
    float vangleMax = +M_PI/2 - 0.0001;
    float zNear;
    float zFar;
    float zoom;
    glm::vec2 angles;

    // mouse
    bool mouseLeft;
    bool mouseRight;
    bool mouseOnGUI;
    glm::vec2 mousePos;
    glm::vec2 startAngles;
    glm::vec3 m_startCenter;

     Camera() {}
    ~Camera() {}
    
    void init();
    void updateEye();
    void updateMatrices();
    void setWindowAspect(float aspect);

    // mouse input
    void mouseDown(int button, float x, float y);
    void mouseUp(int button, float x, float y);
    void mouseMove(float x, float y);
    void mouseWheel(float dx, float dy);
};

void Camera::init() {
    // window
    windowAspect = 1.0f;
    fov   = 60.0f/180.0f*M_PI;
    zNear =     0.1f;
    zFar  = 10000.0f;
    zoom  = 25.0f;

    // center of scene position
    center.x = 0;
    center.y = 0;
    center.z = 0;

    // axes 
    up.x = 0;
    up.y = 0;
    up.z = 1;

    // start angles
    angles[0] = (-90.0f)/180.0f*M_PI;
    angles[1] = (45.0f )/180.0f*M_PI;

    // mouse status
    mouseOnGUI = false;
    mouseLeft  = false;
    mouseRight = false;
    mousePos.x = 0.0f;
    mousePos.y = 0.0f;

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
    modelView  = glm::lookAt(eye, center, up);
    projection = glm::perspective(fov, windowAspect, zNear, zFar);
}

void Camera::setWindowAspect(float aspect) {
    windowAspect = aspect;
    updateMatrices();
}

void Camera::mouseDown(int button, float x, float y) {
    if(mouseOnGUI)
        return;

    mousePos.x = x;
    mousePos.y = y;
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

}}