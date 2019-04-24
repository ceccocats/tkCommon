#pragma once

#include <QGLViewer/qglviewer.h>
#include "lodepng.h"
#include "../common.h"

namespace tk { namespace gui {

    class Viewer : public QGLViewer {

    protected:
        virtual void draw();
        virtual void init();
        virtual QString helpString() const;

    public:
        Viewer(QWidget *parent = nullptr);

        struct Zcol_t {
            float min, max;
        };

        struct object3D_t {
            GLuint tex;
            Eigen::MatrixXf triangles;
        };

    protected:
        void tkApplyTf(tk::common::Tfpose tf);
        void tkDrawAxis(float s = 2.0);
        void tkDrawTexture(GLuint tex, float s);
        int  tkLoadTexture(std::string filename, GLuint &tex);
        int  tkLoadOBJ(std::string filename, object3D_t &obj);
        void tkDrawObject3D(object3D_t *obj, float size = 1.0, int GL_method = GL_TRIANGLES, bool textured = false);
        void tkDrawCircle(float x, float y, float z, float r, int res = 20);
        void tkDrawCloud(Eigen::MatrixXf *data, Zcol_t *col = nullptr);
        void tkRainbowColor(float hue);
        void tkViewport2D();
    };
}}