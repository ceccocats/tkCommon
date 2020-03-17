#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>


// Draw cube
// NOTE: Cube position is the center position
static void DrawCube(Vector3 position, float width, float height, float length, Color color)
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    rlPushMatrix();

        // NOTE: Be careful! Function order matters (rotate -> scale -> translate)
        rlTranslatef(position.x, position.y, position.z);
        //rlScalef(2.0f, 2.0f, 2.0f);
        //rlRotatef(45, 0, 1, 0);

        rlBegin(RL_TRIANGLES);
            rlColor4ub(color.r, color.g, color.b, color.a);

            // Front Face -----------------------------------------------------
            rlVertex3f(x-width/2, y-height/2, z+length/2);  // Bottom Left
            rlVertex3f(x+width/2, y-height/2, z+length/2);  // Bottom Right
            rlVertex3f(x-width/2, y+height/2, z+length/2);  // Top Left

            rlVertex3f(x+width/2, y+height/2, z+length/2);  // Top Right
            rlVertex3f(x-width/2, y+height/2, z+length/2);  // Top Left
            rlVertex3f(x+width/2, y-height/2, z+length/2);  // Bottom Right

            // Back Face ------------------------------------------------------
            rlVertex3f(x-width/2, y-height/2, z-length/2);  // Bottom Left
            rlVertex3f(x-width/2, y+height/2, z-length/2);  // Top Left
            rlVertex3f(x+width/2, y-height/2, z-length/2);  // Bottom Right

            rlVertex3f(x+width/2, y+height/2, z-length/2);  // Top Right
            rlVertex3f(x+width/2, y-height/2, z-length/2);  // Bottom Right
            rlVertex3f(x-width/2, y+height/2, z-length/2);  // Top Left

            // Top Face -------------------------------------------------------
            rlVertex3f(x-width/2, y+height/2, z-length/2);  // Top Left
            rlVertex3f(x-width/2, y+height/2, z+length/2);  // Bottom Left
            rlVertex3f(x+width/2, y+height/2, z+length/2);  // Bottom Right

            rlVertex3f(x+width/2, y+height/2, z-length/2);  // Top Right
            rlVertex3f(x-width/2, y+height/2, z-length/2);  // Top Left
            rlVertex3f(x+width/2, y+height/2, z+length/2);  // Bottom Right

            // Bottom Face ----------------------------------------------------
            rlVertex3f(x-width/2, y-height/2, z-length/2);  // Top Left
            rlVertex3f(x+width/2, y-height/2, z+length/2);  // Bottom Right
            rlVertex3f(x-width/2, y-height/2, z+length/2);  // Bottom Left

            rlVertex3f(x+width/2, y-height/2, z-length/2);  // Top Right
            rlVertex3f(x+width/2, y-height/2, z+length/2);  // Bottom Right
            rlVertex3f(x-width/2, y-height/2, z-length/2);  // Top Left

            // Right face -----------------------------------------------------
            rlVertex3f(x+width/2, y-height/2, z-length/2);  // Bottom Right
            rlVertex3f(x+width/2, y+height/2, z-length/2);  // Top Right
            rlVertex3f(x+width/2, y+height/2, z+length/2);  // Top Left

            rlVertex3f(x+width/2, y-height/2, z+length/2);  // Bottom Left
            rlVertex3f(x+width/2, y-height/2, z-length/2);  // Bottom Right
            rlVertex3f(x+width/2, y+height/2, z+length/2);  // Top Left

            // Left Face ------------------------------------------------------
            rlVertex3f(x-width/2, y-height/2, z-length/2);  // Bottom Right
            rlVertex3f(x-width/2, y+height/2, z+length/2);  // Top Left
            rlVertex3f(x-width/2, y+height/2, z-length/2);  // Top Right

            rlVertex3f(x-width/2, y-height/2, z+length/2);  // Bottom Left
            rlVertex3f(x-width/2, y+height/2, z+length/2);  // Top Left
            rlVertex3f(x-width/2, y-height/2, z-length/2);  // Bottom Right
        rlEnd();
    rlPopMatrix();
}

class MyViewer : public tk::gui::Viewer {
    private:
        tk::common::Tfpose tf = tk::common::Tfpose::Identity();
        float   angle;
        Eigen::MatrixXf *cloud = nullptr;

        bool show_demo_window = true;
    public:
        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();
        }

        void draw() {
            tk::gui::Viewer::draw();

            // Arrow
            tkSetColor(tk::gui::color::RED);
            //tkDrawArrow(tk::common::Vector3<float>{0.0, 0.0, 3.0}, angle, 2.0);

            // Cube
            tk::common::Vector3<float>p(0.0, 4.0, 1.0);
            tk::common::Vector3<float>s(4.0, 2.0, 2.0);
            tk::gui::Color_t col = tk::gui::color::PINK;
            tkSetColor(col);
            tkDrawCube(p, s, true);
        }

        void setCloud(Eigen::MatrixXf *cloud) { this->cloud = cloud; }
        void setAngle(float angle) { this->angle = angle; tf = tk::common::odom2tf(0, 0, angle); }

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
        //viewer->plotManger->addPoint("test", tk::common::Vector3<float>{3.0f*cos(angle), 3.0f*sin(angle), 0});


        rate.wait();
    }
}

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "tkGUI sample gui app");
    cmd.print();

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

    viewer->joinThread();
    pthread_join(t0, NULL);
    return 0;
}