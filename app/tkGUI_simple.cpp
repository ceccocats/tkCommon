#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>

class MyViewer : public tk::gui::Viewer {
    private:
        object3D_t carObj;

        tk::common::Tfpose tf = tk::common::Tfpose::Identity();
        float   angle;
        float   speedometer_speed = 5.0;
        Eigen::MatrixXf *cloud = nullptr;

        bool show_demo_window = true;
    public:
        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();

            tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);

            plotManger->addLinePlot("test", tk::gui::color::WHITE, 1000, 1);
        }

        void draw() {
            tk::gui::Viewer::draw();

            // Arrow
            tkSetColor(tk::gui::color::RED);
            tkDrawArrow(tk::common::Vector3<float>{0.0, 0.0, 3.0}, angle, 2.0);

            // Cube
            tk::common::Vector3<float>p(0.0, 4.0, 1.0);
            tk::common::Vector3<float>s(4.0, 2.0, 2.0);
            tk::gui::Color_t col = tk::gui::color::PINK;
            tkSetColor(col);
            tkDrawCube(p, s, false);
            col.a /= 4;
            tkSetColor(col);
            tkDrawCube(p, s, true);

            // Circle
            tkSetColor(tk::gui::color::PINK);
            tkDrawCircle(tk::common::Vector3<float>{0.0, 0.0, 0.0}, 8.0, 100);
   
            // text
            tkSetColor(tk::gui::color::LIME);
            tkDrawText("tkGUI", tk::common::Vector3<float>{-5, 10.0, 0.0},
                                tk::common::Vector3<float>{M_PI/2, 0, 0},
                                tk::common::Vector3<float>{5.0, 5.0, 5.0});

            // tornado cloud
            glPushMatrix(); {
                tkApplyTf(tf);
                glTranslatef(8, 0, 0);

                if(cloud != nullptr) {
                    tkSetColor(tk::gui::color::ORANGE);
                    glPointSize(1.0f);

                    tkDrawCloud(cloud);

                    // Sphere
                    col = tk::gui::color::CYAN;
                    tkSetColor(col);
                    tkDrawSphere(tk::common::Vector3<float>{0.0, 0.0, 13.0}, 5.0, 16, false);
                    col.a /= 4;
                    tkSetColor(col);
                    tkDrawSphere(tk::common::Vector3<float>{0.0, 0.0, 13.0}, 5.0, 16, true);
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
                tkDrawTexture(hipertTex, 10, 10);
            } glPopMatrix();
            
            // draw 2D HUD
            tkViewport2D(width, height);

            tkDrawSpeedometer(tk::common::Vector2<float>(-0.75, -0.75), speedometer_speed, 0.2);
            
            glPushMatrix(); {
                tkSetColor(tk::gui::color::WHITE);
                glTranslatef(0.7, -0.85, 0);
                //tkDrawTexture(hipertTex, 0.5, 0.5);
            } glPopMatrix();
            

            glPushMatrix(); {
                tkSetColor(tk::gui::color::LIME);
                char fps_str[256];
                sprintf(fps_str, "FPS: %.2f", ImGui::GetIO().Framerate);
                tkDrawText(fps_str, tk::common::Vector3<float>{0.62, +0.9, 0},
                                    tk::common::Vector3<float>{0, 0, 0},
                                    tk::common::Vector3<float>{0.06, 0.06, 0.0});
            } glPopMatrix();

                       
            // draw 2d GUI 1
            ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
            if (show_demo_window)
                ImGui::ShowDemoWindow(&show_demo_window);


        }

        void setCloud(Eigen::MatrixXf *cloud) { this->cloud = cloud; }
        void setAngle(float angle) { this->angle = angle; tf = tk::common::odom2tf(0, 0, angle); }
        void setSpeed(float speed) { this->speedometer_speed = speed; }

};

MyViewer *viewer = nullptr;
bool gRun = true;


void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

void *update_th(void *data) {

    float angle = 0;
    float speed = 0;
    bool  speedUP = true;

    LoopRate rate(10000, "UPDATE");
    while(gRun){
        angle += M_PI/100;

        if (speedUP) {
            speed += 0.01;
            if (speed >= 88.0)
                speedUP = false;
        } else {
            speed -= 0.01;
            if (speed <= 0)
                speedUP = true;
        }

        viewer->setAngle(angle);
        viewer->setSpeed(speed);

        viewer->plotManger->addPoint("test", tk::common::Vector3<float>{3.0f*(float)cos(angle), 3.0f*(float)sin(angle), 0});


        rate.wait();
    }
}

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "tkGUI sample gui app");
    cmd.parse();

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
        double z = h_d*(N/360) - h_d*h_idx;
        double r = (z/5)*(z/5);
        cloud(0,i) = cos(ang)*(r);
        cloud(1,i) = sin(ang)*(r);
        cloud(2,i) = z;
        cloud(3,i) = 1;
    }

    viewer = new MyViewer();
    viewer->setWindowName("test");
    viewer->setBackground(tk::gui::color::DARK_GRAY);
    viewer->initOnThread();
    viewer->setCloud(&cloud);

    // update thread
    pthread_t       t0;
    pthread_create(&t0, NULL, update_th, NULL);

    // fake loading
    //sleep(2);

    viewer->joinThread();
    pthread_join(t0, NULL);
    return 0;
}