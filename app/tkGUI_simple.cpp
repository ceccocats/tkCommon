#include "tkCommon/gui/Viewer.h"
#include <thread>

class MyViewer : public tk::gui::Viewer {
    public:
        tk::common::Tfpose tf = tk::common::Tfpose::Identity();
        float angle;
        Eigen::MatrixXf *cloud = nullptr;


        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();
        }

        void draw() {
            tk::gui::Viewer::draw();

            tkDrawAxis();

            tk::common::Vector3<float>p(0.0, 0.0, 1.0);
            tk::common::Vector3<float>s(3.0, 3.0, 2.0);
            tkSetColor(tk::gui::color::LIME);
            tkDrawCube(p, s, false);
            
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


            //tkDrawArrow(p, M_PI/4, 1.0);

        }

        void setCloud(Eigen::MatrixXf *cloud) { this->cloud = cloud; }
        void setAngle(float angle) { this->angle = angle; tf = tk::common::odom2tf(0, 0, angle); }

};

MyViewer *viewer = nullptr;
bool gRun = true;

int main( int argc, char** argv){

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

    // use the context in a separate rendering thread
    std::thread render_loop;
    render_loop = viewer->spawn();

    float angle = 0.0;
    LoopRate rate(10000, "UPDATE");
    while(viewer->isRunning()){
        angle += M_PI/100;
        viewer->setAngle(angle);
        rate.wait();
    }

    render_loop.join();
    return 0;
}