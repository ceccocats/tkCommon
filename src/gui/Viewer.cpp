#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/OBJ_Loader.h"
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

    return error;
}

int Viewer::tkLoadOBJ(std::string filename, object3D_t &obj) {

    int error = 0;
    std::cout<<"loading "<<filename<<".obj    ";
    
    // correct locale dependent stof
    std::setlocale(LC_ALL, "C");

    objl::Loader loader;
    if(loader.LoadFile((filename + ".obj").c_str())) {
   
        std::cout<<" OK!!\n";
        obj.triangles.resize(loader.LoadedMeshes.size());
        obj.colors.resize(loader.LoadedMeshes.size());

        for(int o=0; o<loader.LoadedMeshes.size(); o++) {
            std::cout<<"name: "<<loader.LoadedMeshes[o].MeshName<<"  verts: "<<loader.LoadedMeshes[o].Vertices.size()<<"\n";
   
            std::vector<unsigned int> indices = loader.LoadedMeshes[o].Indices;
            std::vector<objl::Vertex> verts = loader.LoadedMeshes[o].Vertices;

            obj.colors[o].x = loader.LoadedMeshes[o].MeshMaterial.Kd.X;
            obj.colors[o].y = loader.LoadedMeshes[o].MeshMaterial.Kd.Y;
            obj.colors[o].z = loader.LoadedMeshes[o].MeshMaterial.Kd.Z;
            std::cout<<"mat: "<<loader.LoadedMeshes[o].MeshMaterial.name<<" diffuse: "<<obj.colors[o]<<"\n";
            
            obj.triangles[o] = Eigen::MatrixXf(5, indices.size());
            for(int i=0; i<indices.size(); i++) {
                int idx = indices[i];
                obj.triangles[o](0,i) = verts[idx].Position.X;
                obj.triangles[o](1,i) = verts[idx].Position.Y;
                obj.triangles[o](2,i) = verts[idx].Position.Z;
                obj.triangles[o](3,i) = verts[idx].TextureCoordinate.X;
                obj.triangles[o](4,i) = 1 - verts[idx].TextureCoordinate.Y;
            }
        }


    } else {
        error = 1;
        std::cout<<" ERROR\n";
    }
    
    error = error || tkLoadTexture((filename + ".png"), obj.tex);
    return error;
}

void Viewer::tkDrawObject3D(object3D_t *obj, float size, bool textured) {

    glPushMatrix();
    glScalef(size, size, size);

    if(textured) {
        glBindTexture(GL_TEXTURE_2D, obj->tex);
        glEnable(GL_TEXTURE_2D);
    }

    glBegin(GL_TRIANGLES);
    for(int o=0; o<obj->triangles.size(); o++) {
        if(!textured)
            glColor3f(obj->colors[o].x, obj->colors[o].y, obj->colors[o].z);

        for(int i=0; i<obj->triangles[o].cols(); i++) {
            glTexCoord2f(obj->triangles[o](3,i), obj->triangles[o](4,i)); 
            glVertex3f(obj->triangles[o](0,i),obj->triangles[o](1,i),obj->triangles[o](2,i));
        }
    }
    glEnd();

    if(textured) {
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glPopMatrix();
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
}

void Viewer::init() {

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Restore previous viewer state.
    restoreStateFromFile();
    
    // Set background color white.
    glClearColor(1.0f,1.0f,1.0f,1.0f);

    this->camera()->setZNearCoefficient(0.00001);
    this->camera()->setZClippingCoefficient(10000.0);

}

void Viewer::wheelEvent(QWheelEvent *event) {

    int delta = event->delta();
    if(delta > +100) delta = +100;
    if(delta < -100) delta = -100;

    QWheelEvent ev(event->pos(),event->globalPos(),event->pixelDelta(),event->angleDelta(),
            delta, event->orientation(), event->buttons(),event->modifiers(), event->phase(), event->source());
    event = &ev;
    QGLViewer::wheelEvent(event);
}
