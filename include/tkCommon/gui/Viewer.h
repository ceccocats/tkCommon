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
#include "tkCommon/gui/Camera3D.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/gui/lodepng.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"

#include "tkCommon/data/RadarData.h"
#include "tkCommon/data/LidarData.h"
#include "tkCommon/data/ImageData.h"

#include "tkCommon/gui/libdrawtext/drawtext.h"

#include <tkCommon/terminalFormat.h>

namespace tk { namespace gui {

    class PlotManager;

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
        virtual void drawSplash();
        void run();


        static void *threadCaller(void *args) {
            Viewer *self = (Viewer*) args;
            self->init();
            self->init_mutex.unlock();
            self->run();
        }

        void initOnThread(bool splashScreen = true) {
            // enable splash screen
            splash = splashScreen;

            ts = getTimeStamp();

            //clsSuc("start viz thread\n");
            init_mutex.lock();
            pthread_create(&thread, NULL, threadCaller, (void*)this);
            init_mutex.lock();
            //clsSuc("viz thread initted\n");
            init_mutex.unlock();
        }

        void setSplash(bool splash) {
            this->splash = splash;
        }

        void joinThread() {

            while(getTimeStamp() - ts < splashTime){
                usleep(100);
            }

            // disable splash screen
            splash = false;

            pthread_join(thread, NULL);
            clsSuc("closed\n");
        }

        void setSplashTime(float t) {
            splashTime = t * 1000000;
        }

        void setIcon(std::string filename);
        void setWindowName(std::string name);
        void setBackground(tk::gui::Color_t c);
        
        static int  tkLoadTexture(std::string filename, GLuint &tex);
        static int  tkLoadOBJ(std::string filename, object3D_t &obj);
        static void tkLoadLogo(std::string filename, std::vector<common::Vector3<float>> &logo);
        
        static void tkSetColor(tk::gui::Color_t c, float alpha = -1);
        static void tkApplyTf(tk::common::Tfpose tf);
        static void tkDrawAxis(float s = 1.0);
        static void tkDrawTriangle(tk::common::Vector3<float> a, tk::common::Vector3<float> b, tk::common::Vector3<float> c, bool filled = false);
        static void tkDrawCircle(tk::common::Vector3<float> pose, float r, int res = 20, bool filled = false);
        static void tkDrawSphere(tk::common::Vector3<float> pose, float r, int res = 20, bool filled = true);
        static void tkDrawCloud(Eigen::MatrixXf *data);
        static void tkDrawCloudFeatures(Eigen::MatrixXf *points, Eigen::MatrixXf *features, int idx, float maxval=1.0);
        static void tkDrawCloudRGB(Eigen::MatrixXf *points, Eigen::MatrixXf *features, int r, int g, int b);
        static void tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius = -1.0, int nbSubdivisions = 12);
        static void tkDrawArrow(float length = 1.0, float radius = -1.0, int nbSubdivisions = 12);
        static void tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled = true);
        static void tkDrawRectangle(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled = true);
        static void tkDrawLine(tk::common::Vector3<float> p0, tk::common::Vector3<float> p1);
        static void tkDrawLine(std::vector<tk::common::Vector3<float>> poses);
        static void tkDrawPoses(std::vector<tk::common::Vector3<float>> poses, tk::common::Vector3<float> size = tk::common::Vector3<float>{0.2, 0.2, 0.2});
        static void tkDrawObject3D(object3D_t *obj, float size = 1.0, bool textured = false);
        static void tkDrawTexture(GLuint tex, float sx, float sy);
        static void tkDrawText(std::string text, tk::common::Vector3<float> pose,
                           tk::common::Vector3<float> rot = tk::common::Vector3<float>{0.0, 0.0, 0.0},
                           tk::common::Vector3<float> scale = tk::common::Vector3<float>{1.0, 1.0, 1.0});
        static void tkRainbowColor(float hue, uint8_t &r, uint8_t &g, uint8_t &b);
        static void tkSetRainbowColor(float hue);
        static void tkDrawSpeedometer(tk::common::Vector2<float> pose, float speed, float radius);

        // data 
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        static void tkDrawTf(std::string name, tk::common::Tfpose tf);
        static void tkDrawLogo(std::string file, double scale);
        static void tkDrawRadarData(tk::data::RadarData *data);

        static void tkDrawImage(tk::data::ImageData<uint8_t>& image, GLuint texture);
        static void tkSplitPanel(int count, float ratio, float xLim, int &num_cols, int &num_rows, float &w, float &h, float &x, float &y);

        static void tkDrawLiDARData(tk::data::LidarData *data);
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        static void tkViewport2D(int width, int height, int x=0, int y=0);

        static void errorCallback(int error, const char* description);
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

        bool isRunning() {return !glfwWindowShouldClose(window);};
        void close() { glfwSetWindowShouldClose(window, true); }
    
        int                     width = 800, height = 800;
        float                   aspectRatio = 1;
        float                   xLim = 1.0; /**< 2d x coord screen limit (1.0 if quad) */  
        float                   yLim = 1.0; /**< 2d y coord screen limit (fixed to 1.0) */

        static Camera3D         mouseView;
        static const int        MAX_KEYS = 1024;
        static bool             keys[MAX_KEYS];
        static std::vector<tk::gui::Color_t> colors;

        bool drawLogo = true;
        std::vector<tk::common::Vector3<float>> logo;

        double dt = 1.0/30;

        PlotManager *plotManger;

        // tread for running viz in background
        pthread_t thread;
        std::mutex init_mutex;

        // font
        static int TK_FONT_SIZE;
        std::string fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
        struct dtx_font *font;

        // tex
        GLuint                  hipertTex;


    private:
        std::string             windowName;
        std::string             icon_fname;
        Color_t                 background = tk::gui::color::DARK_GRAY;
        bool                    splash = false; // true if splash screen
        timeStamp_t             ts;
        float                   splashTime = 0;

        GLFWwindow*             window;
        static GLUquadric*      quadric;

        const char*             glsl_version = "#version 130";

        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
        static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
        static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    };

}}
#include "tkCommon/gui/PlotManager.h"
