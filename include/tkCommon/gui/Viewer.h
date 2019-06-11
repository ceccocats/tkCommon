#pragma once

#include <pangolin/pangolin.h>
#include <thread>
#include "tkCommon/common.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/gui/lodepng.h"
#include "tkCommon/gui/OBJ_Loader.h"

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
        std::thread spawn() {
            return std::thread(&Viewer::run, this);
        }

        void setWindowName(std::string name);
        void setBackground(tk::gui::Color_t c);

        int getWidth()  {return width;};
        int getHeight() {return height;};
        
        static int  tkLoadTexture(std::string filename, GLuint &tex);
        static int  tkLoadOBJ(std::string filename, object3D_t &obj);
        
        static void tkSetColor(tk::gui::Color_t c);
        static void tkApplyTf(tk::common::Tfpose tf);
        static void tkDrawAxis(float s = 2.0);
        static void tkDrawCircle(float x, float y, float z, float r, int res = 20);
        static void tkDrawCloud(Eigen::MatrixXf *data);
        static void tkDrawArrow(float length = 1.0, float radius = -1.0, int nbSubdivisions = 12);
        static void tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius = -1.0, int nbSubdivisions = 12);
        static void tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled = true);
        static void tkDrawObject3D(object3D_t *obj, float size = 1.0, bool textured = false);
        static void tkDrawTexture(GLuint tex, float s);

        static void tkViewport2D(int width, int height);

        bool isRunning() {return !pangolin::ShouldQuit();};
    
    private:
        std::string     windowName;
        Color_t         background;

        int             width, height;
    };
}}
