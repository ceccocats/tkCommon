#include "tkCommon/gui/Viewer.h"

using namespace tk::gui;

Viewer::Viewer() {
}

Viewer::~Viewer() {
}

void 
Viewer::init() {
    // create a window and bind its context to the main thread
    pangolin::CreateWindowAndBind(windowName, 640, 480);

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_MULTISAMPLE);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}

void 
Viewer::draw() {
}

void 
Viewer::run() {
    // fetch the context and bind it to this thread
    pangolin::BindToContext(windowName);

    // we manually need to restore the properties of the context
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisZ)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    while(!pangolin::ShouldQuit()) {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);
        d_cam.Activate(s_cam);

        draw();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}

void
Viewer::setWindowName(std::string name) {
    windowName = name;
}

void 
Viewer::setBackground(tk::gui::Color_t c) {
    background = c;
}

void 
Viewer::tkSetColor(tk::gui::Color_t c) {
    glColor4ub(c.r, c.g, c.b, c.a);
}

void 
Viewer::tkApplyTf(tk::common::Tfpose tf) {
    // apply roto translation
    tk::common::Vector3<float> p = tk::common::tf2pose(tf);
    tk::common::Vector3<float> r = tk::common::tf2rot (tf);
    glTranslatef(p.x, p.y, p.z);
    glRotatef(r.x*180.0/M_PI, 1, 0, 0);
    glRotatef(r.y*180.0/M_PI, 0, 1, 0);
    glRotatef(r.z*180.0/M_PI, 0, 0, 1);
}  

void 
Viewer::tkDrawAxis(float s) {
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

void 
Viewer::tkDrawCircle(float x, float y, float z, float r, int res) {
    glBegin(GL_LINE_LOOP);
    for (int j = 0; j < res; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr = r * cosf(theta);//calculate the x component 
        float yr = r * sinf(theta);//calculate the y component 
        glVertex3f(x + xr, y + yr, z);//output vertex
    }
    glEnd(); 
}

void 
Viewer::tkDrawCloud(Eigen::MatrixXf *points) {
    glBegin(GL_POINTS);
    for (int p = 0; p < points->cols(); p++) {
        Eigen::Vector4f v = points->col(p);
        glVertex3f(v(0), v(1), v(2));
    }
    glEnd();
}

void 
Viewer::tkDrawArrow(float length, float radius, int nbSubdivisions) {
    static GLUquadric *quadric = gluNewQuadric();

    if (radius < 0.0)
        radius = 0.05 * length;

    const float head = 2.5 * (radius / length) + 0.1;
    const float coneRadiusCoef = 4.0 - 5.0 * head;

    gluCylinder(quadric, radius, radius, length * (1.0 - head / coneRadiusCoef),
                nbSubdivisions, 1);
    glTranslated(0.0, 0.0, length * (1.0 - head));
    gluCylinder(quadric, coneRadiusCoef * radius, 0.0, head * length,
                nbSubdivisions, 1);
    glTranslated(0.0, 0.0, -length * (1.0 - head));
}

void 
Viewer::tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius, int nbSubdivisions) {
    glPushMatrix();
    glRotatef(90.0, 1.0, 0.0, 0.0);
    glRotatef(90.0 + yaw*180/M_PI, 0.0, 1.0, 0.0);
    glTranslatef(pose.x, pose.y, pose.z);
    tkDrawArrow(lenght, radius, nbSubdivisions);
    glPopMatrix();
}

void 
Viewer::tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    glScalef(size.x, size.y, size.z);

    // BACK
    glBegin(GL_POLYGON);
    glVertex3f(  0.5, -0.5, 0.5 );
    glVertex3f(  0.5,  0.5, 0.5 );
    glVertex3f( -0.5,  0.5, 0.5 );
    glVertex3f( -0.5, -0.5, 0.5 );
    glEnd();
    
    // RIGHT
    glBegin(GL_POLYGON);
    glVertex3f( 0.5, -0.5, -0.5 );
    glVertex3f( 0.5,  0.5, -0.5 );
    glVertex3f( 0.5,  0.5,  0.5 );
    glVertex3f( 0.5, -0.5,  0.5 );
    glEnd();
    
    // LEFT
    glBegin(GL_POLYGON);
    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );
    glEnd();
    
    // TOP
    glBegin(GL_POLYGON);
    glVertex3f(  0.5,  0.5,  0.5 );
    glVertex3f(  0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );
    glEnd();
    
    // BOTTOM
    glBegin(GL_POLYGON);
    glVertex3f(  0.5, -0.5, -0.5 );
    glVertex3f(  0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );
    glEnd();

    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}






