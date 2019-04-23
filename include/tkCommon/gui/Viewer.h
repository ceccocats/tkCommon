#pragma once

#include <QGLViewer/qglviewer.h>
#include "lodepng.h"
#include "../common.h"

namespace tk { namespace gui {

    class Viewer : public QGLViewer {

    private:
        GLuint texures[256];

    protected:
        virtual void draw();
        virtual void init();
        virtual QString helpString() const;

    public:
        Viewer(QWidget *parent = nullptr);

    protected:
        void tkApplyTf(tk::common::Tfpose tf);
        void tkDrawAxis(float s = 2.0);
        void tkDrawTexture(GLuint tex, float s);
        int  tkLoadTexture(std::string filename, GLuint &tex);
        void tkDrawCircle(float x, float y, float z, float r);
    };
}}