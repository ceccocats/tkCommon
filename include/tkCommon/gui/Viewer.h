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

    protected:
        void tkApplyTf(tk::common::Tfpose tf);
        void tkDrawAxis(float s = 2.0);
        void tkDrawTexture(GLuint tex, float s);
        int  tkLoadTexture(std::string filename, GLuint &tex);
        void tkDrawCircle(float x, float y, float z, float r);
        void tkDrawCloud(Eigen::MatrixXf *data, Zcol_t *col = nullptr);
        void tkRainbowColor(float hue);

    };
}}