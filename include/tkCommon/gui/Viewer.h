#pragma once

#include <QGLViewer/qglviewer.h>
#include <QMouseEvent>
#include "lodepng.h"
#include "../common.h"

namespace tk { namespace gui {

    class Viewer : public QGLViewer {

    protected:
        virtual void draw();
        virtual void init();
        void wheelEvent(QWheelEvent *event) override;

    public:
        Viewer(QWidget *parent = nullptr);

        struct Zcol_t {
            float min, max;
        };

        struct object3D_t {
            GLuint tex;
            std::vector<Eigen::MatrixXf> triangles;
            std::vector<tk::common::Vector3<float>> colors;
        };

    protected:
        void tkApplyTf(tk::common::Tfpose tf);
        void tkDrawAxis(float s = 2.0);
        void tkDrawTexture(GLuint tex, float s);
        int  tkLoadTexture(std::string filename, GLuint &tex);
        int  tkLoadOBJ(std::string filename, object3D_t &obj);
        void tkDrawObject3D(object3D_t *obj, float size = 1.0, bool textured = false);
        void tkDrawCircle(float x, float y, float z, float r, int res = 20);
        void tkDrawCloud(Eigen::MatrixXf *data, Zcol_t *col = nullptr);
        void tkRainbowColor(float hue);
        void tkViewport2D();
    };
}}