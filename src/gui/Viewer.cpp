#include "tkCommon/gui/Viewer.h"

using namespace tk::gui;

// Constructor must call the base class constructor.
Viewer::Viewer(QWidget *parent) : QGLViewer(parent) {
}

void Viewer::tkApplyTf(tk::common::Tfpose tf) {
    // apply roto translation
    tk::common::Vector3<float> p = tk::common::tf2pose(tf);
    tk::common::Vector3<float> r = tk::common::tf2rot (tf);
    glTranslatef(p.x, p.y, p.z);
    glRotatef(r.x*180.0/M_PI, 1, 0, 0);
    glRotatef(r.y*180.0/M_PI, 0, 1, 0);
    glRotatef(r.z*180.0/M_PI, 0, 0, 1);
}  

void Viewer::tkDrawAxis(float s) {

    glLineWidth(2.5); 
    glBegin(GL_LINES);
    // x
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(s, 0, 0);
    // y
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, s, 0);
    // z
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, s);
    glEnd();
}

void Viewer::tkDrawTexture(GLuint tex, float s) {

    float i = -s/2;
    float j = +s/2;

    glBindTexture(GL_TEXTURE_2D, tex);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    /*
    glTexCoord2f(0, 0); glVertex3f(i, i, 0);
    glTexCoord2f(0, 1); glVertex3f(i, j, 0);
    glTexCoord2f(1, 1); glVertex3f(j, j, 0);
    glTexCoord2f(1, 0); glVertex3f(j, i, 0);
    */

    // 2d draw
    glTexCoord2f(0, 1); glVertex3f(i, i, 0);
    glTexCoord2f(0, 0); glVertex3f(i, j, 0);
    glTexCoord2f(1, 0); glVertex3f(j, j, 0);
    glTexCoord2f(1, 1); glVertex3f(j, i, 0);


    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

int Viewer::tkLoadTexture(std::string filename, GLuint &tex) {

    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, filename.c_str());
    if(error == 0) {
        if( (width % 32) != 0 || (height % 32) != 0)
            std::cout<<"please use images size multiple of 32\n";

        //upload to GPU texture
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glBindTexture(GL_TEXTURE_2D, 0);

        delete [] image;
    } else {
        tex = 0;
    }

    std::cout<<"loading "<<filename<<"    ";
    if(error == 0)
        std::cout<<" OK!!\n";
    else
        std::cout<<" ERROR: "<<error<<"\n";

    return error == 0;
}

void Viewer::tkDrawCircle(float x, float y, float z, float r, int res) {

    glBegin(GL_LINE_LOOP);
    for (int j = 0; j < res; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr = r * cosf(theta);//calculate the x component 
        float yr = r * sinf(theta);//calculate the y component 
        glVertex3f(x + xr, y + yr, z);//output vertex
    }
    glEnd(); 
}

void Viewer::tkRainbowColor(float hue) {
    if(hue <= 0.0)
        hue = 0.000001;
    if(hue >= 1.0)
        hue = 0.999999;

    int h = int(hue * 256 * 6);
    int x = h % 0x100;

    int r = 0, g = 0, b = 0;
    switch (h / 256)
    {
    case 0: r = 255; g = x;       break;
    case 1: g = 255; r = 255 - x; break;
    case 2: g = 255; b = x;       break;
    case 3: b = 255; g = 255 - x; break;
    case 4: b = 255; r = x;       break;
    //case 5: r = 255; b = 255 - x; break;
    case 5: r = 255; g = x; b = 255 - x; break;
    }
    glColor3f(float(r)/255.0, float(g)/255.0, float(b)/255.0);
}


void Viewer::tkDrawCloud(Eigen::MatrixXf *points, Zcol_t *z_col) {
        
        glBegin(GL_POINTS);
        for (int p = 0; p < points->cols(); p++) {
            Eigen::Vector4f v = points->col(p);
            if(z_col != nullptr)
                tkRainbowColor( (v(2) - z_col->min) / z_col->max );
            glVertex3f(v(0), v(1), v(2));
        }
        glEnd();
}

void Viewer::tkViewport2D() {
    //This sets up the viewport so that the coordinates (0, 0) are at the top left of the window
    glViewport(0, 0, width(), height());  
    float ratio = (float) width() / (float) height();
    glOrtho(0, width(), height(), 0, -10, 10);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    //Back to the modelview so we can draw stuff 
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void Viewer::draw() {

    tkDrawAxis();

}

void Viewer::init() {

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Restore previous viewer state.
    restoreStateFromFile();
    
    // Set background color white.
    glClearColor(1.0f,1.0f,1.0f,1.0f);

    this->camera()->setZNearCoefficient(0.00001);
    this->camera()->setZClippingCoefficient(10000.0);
}

QString Viewer::helpString() const {
    QString text("<h2>S i m p l e V i e w e r</h2>");
    text += "Use the mouse to move the camera around the object. ";
    text += "You can respectively revolve around, zoom and translate with the "
            "three mouse buttons. ";
    text += "Left and middle buttons pressed together rotate around the camera "
            "view direction axis<br><br>";
    text += "Pressing <b>Alt</b> and one of the function keys "
            "(<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
    text += "Simply press the function key again to restore it. Several "
            "keyFrames define a ";
    text += "camera path. Paths are saved when you quit the application and "
            "restored at next start.<br><br>";
    text +=
        "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
    text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save "
            "a snapshot. ";
    text += "See the <b>Keyboard</b> tab in this window for a complete shortcut "
            "list.<br><br>";
    text += "Double clicks automates single click actions: A left button double "
            "click aligns the closer axis with the camera (if close enough). ";
    text += "A middle button double click fits the zoom of the camera and the "
            "right button re-centers the scene.<br><br>";
    text += "A left button double click while holding right button pressed "
            "defines the camera <i>Revolve Around Point</i>. ";
    text += "See the <b>Mouse</b> tab and the documentation web pages for "
            "details.<br><br>";
    text += "Press <b>Escape</b> to exit the viewer.";
    return text;
}
