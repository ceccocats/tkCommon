#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>

class MyViewer : public tk::gui::Viewer {
    private:
        GLuint hipertTex;
        object3D_t carObj;

        tk::common::Tfpose tf = tk::common::Tfpose::Identity();
        float angle;
        Eigen::MatrixXf *cloud = nullptr;

    public:
        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();

            int err = 0;
            err = err || tkLoadTexture("../data/HipertLab.png", hipertTex);
            err = err || tkLoadOBJ("../data/levante", carObj);

        }

        void draw() {
            tk::gui::Viewer::draw();

            tk::common::Vector3<float>p(0.0, 4.0, 1.0);
            tk::common::Vector3<float>s(4.0, 2.0, 2.0);
            tk::gui::Color_t col = tk::gui::color::PINK;
            tkSetColor(col);
            tkDrawCube(p, s, false);
            col.a /= 4;
            tkSetColor(col);
            tkDrawCube(p, s, true);

            tkSetColor(tk::gui::color::PINK);
            tkDrawCircle(0, 0, 0, 8.0, 100);
            
            // tornado cloud
            glPushMatrix(); {
                tkApplyTf(tf);
                if(cloud != nullptr) {
                    tkSetColor(tk::gui::color::ORANGE);
                    glPointSize(1.0f);

                    tkDrawCloud(cloud);
                }
            } glPopMatrix();

            // levante
            glPushMatrix(); {        
                tkDrawObject3D(&carObj, 1, false);
            } glPopMatrix();

            // alpha blended object must be drawn at the end
            // hipert logo as pavement
            glPushMatrix(); {
                glTranslatef(0, -4, 0);
                glColor4f(1,1,1,1);
                tkDrawTexture(hipertTex, 10);
            } glPopMatrix();

            // draw 2D     
            tkViewport2D(width, height);
            glTranslatef(0.7, -0.85, 0);
            tkDrawTexture(hipertTex, 0.5);
        }

        void setCloud(Eigen::MatrixXf *cloud) { this->cloud = cloud; }
        void setAngle(float angle) { this->angle = angle; tf = tk::common::odom2tf(0, 0, angle); }

};

MyViewer *viewer = nullptr;
bool gRun = true;


void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

void *update_th(void *data) {

    float angle = 0;

    LoopRate rate(10000, "UPDATE");
    while(gRun){
        angle += M_PI/100;
        viewer->setAngle(angle);
        rate.wait();
    }
}

int main( int argc, char** argv){

    signal(SIGINT, sig_handler);
    gRun = true;

    // TEST CLOUD
    int h_n = 100;
    int N = 360*h_n;
    double d = 5.0;
    double h_d = 0.1;
    Eigen::MatrixXf cloud = Eigen::MatrixXf(4, N);
    for(int i=0; i<N; i++) {
        int h_idx = i/360;
        double ang = double(i % 360)/180.0*M_PI;
        cloud(0,i) = cos(ang)*(d/h_idx);
        cloud(1,i) = sin(ang)*(d/h_idx) + 8;
        cloud(2,i) = h_d*(N/360) - h_d*h_idx;
        cloud(3,i) = 1;
    }

    viewer = new MyViewer();
    viewer->setWindowName("test");
    viewer->setBackground(tk::gui::color::DARK_GRAY);
    viewer->init();
    viewer->setCloud(&cloud);

    // update thread
    pthread_t       t0;
    pthread_create(&t0, NULL, update_th, NULL);

    viewer->run();

    gRun = false;
    pthread_join(t0, NULL);
    return 0;
}