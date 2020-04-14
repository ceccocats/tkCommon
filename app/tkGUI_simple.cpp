#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>


class Car : public tk::gui::Drawable {
public:
	tk::gui::Viewer::object3D_t carObj;

	void init(){
		tk::gui::Viewer::tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);
	}

	void draw(){
		// levante
		glPushMatrix(); {
			tk::gui::Viewer::tkDrawObject3D(&carObj, 1, false);
		} glPopMatrix();
	}
};

class Arrow : public tk::gui::Drawable {
public:

	tk::common::Vector3<float> pose = tk::common::Vector3<float>{0.0, 0.0, 3.0};
	float   angle;
	float   length;

	void draw(){
		// Arrow
		tk::gui::Viewer::tkSetColor(tk::gui::color::RED);
		tk::gui::Viewer::tkDrawArrow(pose, angle, length);
	}
};

class Cube : public tk::gui::Drawable {
public:

	tk::common::Vector3<float>pose = tk::common::Vector3<float>{0.0, 4.0, 1.0};
	tk::common::Vector3<float>size = tk::common::Vector3<float>{4.0, 2.0, 2.0};

	void draw(){

		tk::gui::Color_t col = tk::gui::color::PINK;
		tk::gui::Viewer::tkSetColor(col);
		tk::gui::Viewer::tkDrawCube(pose, size, false);
		col.a /= 4;
		tk::gui::Viewer::tkSetColor(col);
		tk::gui::Viewer::tkDrawCube(pose, size, true);
	}

};

class Circle : public tk::gui::Drawable {
public:
	tk::common::Vector3<float> pose = tk::common::Vector3<float>{0.0, 0.0, 0.0};
	float radius = 8;
	int res = 100;
	bool filled = false;
	void draw() {
		// Circle
		tk::gui::Viewer::tkSetColor(tk::gui::color::PINK);
		tk::gui::Viewer::tkDrawCircle(pose, radius, res);
	}
};

class Tornado : public tk::gui::Drawable {
public:
	tk::gui::Color_t col;
	tk::gui::PlotManager plotManger;
	Eigen::MatrixXf *cloud = nullptr;
	tk::common::Tfpose tf = tk::common::Tfpose::Identity();
	float angle;
	void draw(){
		// tornado cloud
		tf = tk::common::odom2tf(0, 0, angle);
		glPushMatrix(); {
			tk::gui::Viewer::tkApplyTf(tf);
			glTranslatef(8, 0, 0);

			if(cloud != nullptr) {
				tk::gui::Viewer::tkSetColor(tk::gui::color::ORANGE);
				glPointSize(1.0f);

				tk::gui::Viewer::tkDrawCloud(cloud);

				// Sphere
				col = tk::gui::color::CYAN;
				tk::gui::Viewer::tkSetColor(col);
				tk::gui::Viewer::tkDrawSphere(tk::common::Vector3<float>{0.0, 0.0, 13.0}, 5.0, 16, false);
				col.a /= 4;
				tk::gui::Viewer::tkSetColor(col);
				tk::gui::Viewer::tkDrawSphere(tk::common::Vector3<float>{0.0, 0.0, 13.0}, 5.0, 16, true);
			}
		} glPopMatrix();
	}
};

class TkGUIText : public tk::gui::Drawable {
public:
	void draw(){
		// text
		tk::gui::Viewer::tkSetColor(tk::gui::color::LIME);
		tk::gui::Viewer::tkDrawText("tkGUI", tk::common::Vector3<float>{-5, 10.0, 0.0},
				   tk::common::Vector3<float>{M_PI/2, 0, 0},
				   tk::common::Vector3<float>{5.0, 5.0, 5.0});
	}

	void draw2D(int width, int height, float xLim, float yLim){
		glPushMatrix(); {
			tk::gui::Viewer::tkSetColor(tk::gui::color::LIME);
			char fps_str[256];
			sprintf(fps_str, "FPS: %.2f", ImGui::GetIO().Framerate);
			tk::gui::Viewer::tkDrawText(fps_str, tk::common::Vector3<float>{0.62, +0.9, 0},
					   tk::common::Vector3<float>{0, 0, 0},
					   tk::common::Vector3<float>{0.06, 0.06, 0.0});
		} glPopMatrix();
	}
};

class Speedometer : public tk::gui::Drawable{
public:
	float speed;

	void draw2D(int width, int height, float xLim, float yLim){
		tk::gui::Viewer::tkDrawSpeedometer(tk::common::Vector2<float>(-0.75, -0.75), speed, 0.2);
	};
};

tk::gui::Viewer *viewer = nullptr;
bool gRun = true;


void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

void *update_th(void *data) {
	Car levante;
	Arrow arrow;
	Cube cube;
	Circle circle;
	Tornado tornado;
	TkGUIText text;
	Speedometer speedometer;

	levante.init();

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
	tornado.cloud = &cloud;

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

        arrow.angle = angle;
		tornado.angle = angle;
        speedometer.speed = speed;

        tornado.plotManger.addPoint("test", tk::common::Vector3<float>{3.0f*cos(angle), 3.0f*sin(angle), 0});

        viewer->add("levante", &levante);
		viewer->add("arrow", &arrow);
		viewer->add("cube", &cube);
		viewer->add("circle", &circle);
		viewer->add("tornado", &tornado);
		viewer->add("text", &text);
		viewer->add("speedometer", &speedometer);
        rate.wait();
    }
}

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "tkGUI sample gui app");
    cmd.print();

    signal(SIGINT, sig_handler);
    gRun = true;

    viewer = new tk::gui::Viewer();
    viewer->setWindowName("test");
    viewer->setBackground(tk::gui::color::DARK_GRAY);
    viewer->initOnThread();

    // update thread
    pthread_t       t0;
    pthread_create(&t0, NULL, update_th, NULL);

    // fake loading
    sleep(2);

    viewer->joinThread();
    pthread_join(t0, NULL);
    return 0;
}