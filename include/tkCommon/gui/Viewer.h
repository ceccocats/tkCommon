#pragma once
#include <GL/glew.h> 
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include "tkCommon/gui/utils/CommonViewer.h"
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/utils/Camera.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"
#include "tkCommon/rt/Thread.h"

#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049


namespace tk { namespace gui {

    class Viewer {

    public:
        ~Viewer();
        void start(bool useImGUI = true);
        bool isRunning();
        void stop();
        void join();
        void add(tk::gui::Drawable* obj);

        float getWidth();
        float getHeight();

        static Viewer* getInstance(){
             if (Viewer::instance == nullptr) {
                Viewer::instance = new Viewer();
            }
            return Viewer::instance;
        }

        glm::vec3 getLightPos();

        void setBackground(tk::gui::Color_t col) {
            background = col;
        }
        
        static bool disabled;

    private:
        
        Viewer();
        void  init();
        void  initDrawables();

        static void* run(void* istance);
        static void* fake_run(void* istance);
        void  runloop();

        void  beforeDraw();
        void  drawInfos();
        void  drawSettings();
        void  imguiDraw();
        void  draw();
        void  draw2D();
        
        void  follow();
        void  close();

        static Viewer *instance;

        std::map<int,tk::gui::Drawable*> drawables;
        std::map<int,tk::gui::Drawable*> newDrawables;

        tk::rt::Thread  glThread;

        std::string windowName = "tkGUI";
        Color_t     background = tk::gui::color::DARK_GRAY;

        int     width = 800;
        int     height = 800;
        float   aspectRatio = 1;
        float   xLim = 1.0;
        float   yLim = 1.0;
        double  dt = 1.0/60;
        bool    useImGUI;

        int     imguiSelected = -1;

        bool    running = false;

        Camera      camera;

        glm::vec3   lightPos;

        GLFWwindow *window;
        const char *glsl_version = "#version 130";

        GLint total_mem_kb = 0;
        GLint cur_avail_mem_kb = 0;
        const int   nUsage = 30;
        float       gpuUsage[30]; 
        const int   nFPS = 30;
        float       vizFPS[30];

        // glfw callbacks
        static void errorCallback(int error, const char* description);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
        static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
        static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    };

}}

