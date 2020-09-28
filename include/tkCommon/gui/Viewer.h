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
#include "tkCommon/gui/Drawable.h"
#include "tkCommon/gui/Camera.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/gui/libdrawtext/drawtext.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049


namespace tk { namespace gui {

    class Viewer {

    public:
        static Viewer *instance;

        std::string windowName = "GUI";
        Color_t     background = tk::gui::color::DARK_GRAY;

        int    width = 800, height = 800;
        float  aspectRatio = 1;
        float  xLim = 1.0; /**< 2d x coord screen limit (1.0 if quad) */  
        float  yLim = 1.0; /**< 2d y coord screen limit (fixed to 1.0) */
        double dt = 1.0/60;
        bool   running = true;

        GLint total_mem_kb = 0;
        GLint cur_avail_mem_kb = 0;

		DrawMap drawBuffer;

        Camera camera;
        //static const int        MAX_KEYS = 1024;
        //static bool             keys[MAX_KEYS];

        //Light
        glm::vec3 lightPos;

        // font
        int fontSize = 256;
        std::string fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        struct dtx_font *font;


         Viewer() {
            Viewer::instance = this;
         }
        ~Viewer() {}

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

}}

