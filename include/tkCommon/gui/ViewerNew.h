#pragma once
#include <GL/glew.h> 
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include "tkCommon/common.h"
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/Drawable.h"
#include "tkCommon/gui/Shader.h"
#include "tkCommon/gui/Camera.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/gui/lodepng.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"


namespace tk { namespace gui {

    class ViewerNew : private Viewer {

    public:
        static ViewerNew *instance;

        std::string windowName = "GUI";
        Color_t     background = tk::gui::color::DARK_GRAY;

        int    width = 800, height = 800;
        float  aspectRatio = 1;
        float  xLim = 1.0; /**< 2d x coord screen limit (1.0 if quad) */  
        float  yLim = 1.0; /**< 2d y coord screen limit (fixed to 1.0) */
        double dt = 1.0/30;
        bool   running = true;

		DrawMap drawBuffer;

        Camera camera;
        //static const int        MAX_KEYS = 1024;
        //static bool             keys[MAX_KEYS];


         ViewerNew() {
            ViewerNew::instance = this;
         }
        ~ViewerNew() {}

        virtual void init();
        virtual void draw();
                void run();
                bool isRunning();
                void close();

        // add things to draw
        void add(std::string name, Drawable *data); 

    private:
        GLFWwindow *window;
        const char *glsl_version = "#version 130";

        // glfw callbacks
        static void errorCallback(int error, const char* description);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
        static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
        static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    };

    void ViewerNew::init() {
        clsMsg("init\n");

        glfwSetErrorCallback(errorCallback);
        glfwInit();

        // Create a windowed mode window and its OpenGL context
        window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
        if (!window) {
            glfwTerminate();
        }

        glfwSetScrollCallback(window, scroll_callback);
        glfwSetCursorPosCallback(window, cursor_position_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        //glfwSetKeyCallback(window, keyCallback);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // OpenGL confs
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glEnable(GL_DEPTH_TEST);
        //glDisable(GL_CULL_FACE);        
        //glDepthFunc(GL_GEQUAL);
        //glEnable(GL_LINE_SMOOTH);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        //glEnable(GL_BLEND);
        //glEnable(GL_ALPHA_TEST);
        //glDepthMask(GL_FALSE); // make the depth buffer read-only
        
        // print OpenGL status
        std::string msg = std::string{"OPENGL running on:"} + 
                          std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) + " -- " +
                          std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\n";
        clsMsg(msg);

        // set window size
        glfwGetFramebufferSize(window, &width, &height);
        clsSuc("init with resolution " + std::to_string(width) + "x" + std::to_string(height) + "\n");

        // init cameras
        camera.init();
    }

    void ViewerNew::draw() {
        drawBuffer.draw(this);
    }

    void ViewerNew::run() {
        clsMsg("run\n");
        running = true;
        timeStamp_t VIZ_DT_US = dt*1e6;
        LoopRate rate((VIZ_DT_US), "VIZ_UPDATE");
        while (running) {

            // manage window resize
            glfwGetFramebufferSize(window, &width, &height);
            aspectRatio = (float)width / (float)height;
            xLim = aspectRatio;
            yLim = 1.0;
            glViewport(0, 0, width, height);

            glMatrixMode( GL_PROJECTION );
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity() ;

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);

            camera.setWindowAspect(aspectRatio);
            glPushMatrix();

            // apply camera matrix
            glMultMatrixf(glm::value_ptr(camera.projection));
            glMultMatrixf(glm::value_ptr(camera.modelView));
            
            draw();

            glPopMatrix();

            /* Swap front and back buffers */
            glfwSwapBuffers(window);

            /* Poll for and process events */
            glfwPollEvents();
            // update running status
            running = running && !glfwWindowShouldClose(window);
            rate.wait(false);
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    }


    void ViewerNew::close() { 
        clsMsg("close\n");
        running = false;
    }

    void ViewerNew::add(std::string name, Drawable *data){
        clsMsg("add\n");
        drawBuffer.add(name, data, this);
    }

    void ViewerNew::errorCallback(int error, const char* description) {
        tk::tformat::printErr("Viewer", std::string{"error: "}+std::to_string(error)+" "+description+"\n");
    }

    void ViewerNew::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        ViewerNew::instance->camera.mouseWheel(xoffset, yoffset);
    }

    void ViewerNew::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        ViewerNew::instance->camera.mouseMove(xpos, ypos);
    }

    void ViewerNew::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if (action == GLFW_PRESS)
            ViewerNew::instance->camera.mouseDown(button, xpos, ypos);
        if (action == GLFW_RELEASE)
            ViewerNew::instance->camera.mouseUp(button, xpos, ypos);
    }
}}
