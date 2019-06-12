#pragma once
#include <GL/glew.h> 
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>  
#include "tkCommon/common.h"
#include "tkCommon/gui/MouseView3D.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/gui/lodepng.h"
#include "tkCommon/gui/imgui.h"
#include "tkCommon/gui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui_impl_opengl3.h"



namespace tk { namespace gui {

    class Viewer {

    public:
        Viewer();
        ~Viewer();

        struct object3D_t {
            GLuint tex;
            std::vector<Eigen::MatrixXf> triangles;
            std::vector<tk::common::Vector3<float>> colors;
        };
        
        virtual void init();
        virtual void draw();
        
        void run();

        void setWindowName(std::string name);
        void setBackground(tk::gui::Color_t c);
        
        static int  tkLoadTexture(std::string filename, GLuint &tex);
        static int  tkLoadOBJ(std::string filename, object3D_t &obj);
        
        static void tkSetColor(tk::gui::Color_t c);
        static void tkApplyTf(tk::common::Tfpose tf);
        static void tkDrawAxis(float s = 1.0);
        static void tkDrawCircle(tk::common::Vector3<float> pose, float r, int res = 20);
        static void tkDrawSphere(tk::common::Vector3<float> pose, float r, int res = 20, bool filled = true);
        static void tkDrawCloud(Eigen::MatrixXf *data);
        static void tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius = -1.0, int nbSubdivisions = 12);
        static void tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled = true);
        static void tkDrawRectangle(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled = true);
        static void tkDrawObject3D(object3D_t *obj, float size = 1.0, bool textured = false);
        static void tkDrawTexture(GLuint tex, float s);
        static void tkDrawText(std::string text, tk::common::Vector3<float> pose, 
                               tk::common::Vector3<float> rot = tk::common::Vector3<float>{0.0, 0.0, 0.0}, 
                               tk::common::Vector3<float> scale = tk::common::Vector3<float>{1.0, 1.0, 1.0});

        static void tkViewport2D(int width, int height);

        static void errorCallback(int error, const char* description);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

        bool isRunning() {return !glfwWindowShouldClose(window);};
    
        int                     width, height;

    private:
        std::string             windowName;
        Color_t                 background = tk::gui::color::DARK_GRAY;

        GLFWwindow*             window;
        static MouseView3D      mouseView;
        static GLUquadric*      quadric;

        // font

        const char*             glsl_version = "#version 130";

        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
        static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
        static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

        static void tkDrawArrow(float length = 1.0, float radius = -1.0, int nbSubdivisions = 12);
    };
}}
