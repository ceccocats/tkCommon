#pragma once 
#include "tkCommon/common.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace tk { namespace gui  {

class Camera
{
  public:
    glm::ivec4 viewport;
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
    glm::vec2 lastMousePos;
    glm::vec2 lastAngles;
    glm::vec3 lastCenter;

     Camera() {}
    ~Camera() {}
    
    void init();
    void updateEye();
    void updateMatrices();
    void setViewPort(int x, int y, int width, int height);
    void setCenter(tk::common::Vector3<float> p);
    void setAngle(tk::common::Vector3<float> p);

    // util functons
    glm::vec3 project(glm::vec3 obj);
    glm::vec3 unprojectPlane(glm::vec2 obj);

    // mouse input
    void mouseDown(int button, float x, float y);
    void mouseUp(int button, float x, float y);
    void mouseMove(float x, float y);
    void mouseWheel(float dx, float dy);
};

}}