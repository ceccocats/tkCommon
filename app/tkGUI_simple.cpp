#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>


class Scene : public tk::gui::Drawable {
public:
	tk::gui::Viewer::object3D_t carObj;
	Eigen::MatrixXf cloud;
	unsigned int VBO,VAO;

	void init(){
		tk::gui::Viewer::tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);
		
		// dief_vlp32.ply:  
		// wget https://cloud.hipert.unimore.it/s/W3rNtFxRGG5AC89/download
		// mv download dief_vlp32.ply

		cloud.resize(3, 10e6);
		std::ifstream is("dief_vlp32.ply");
		float x,y,z;
		int count =0;
		while (is >> x >> y >> z) {
			cloud.col(count) << x, y, z;
			count++;
		}
		cloud.conservativeResize(3, count);
		std::cout<<"cloud size: "<<count<<"\n";
	}

	void draw(tk::gui::Viewer *viewer){
		// levante
		//glPushMatrix(); {
		//	tk::gui::Viewer::tkDrawObject3D(&carObj, 1, false);
		//} glPopMatrix();

		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, cloud.size() * sizeof(float), cloud.data(), GL_DYNAMIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, cloud.cols());
		glBindVertexArray(0);
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glUseProgram(0);
		
	}

	void draw2D(tk::gui::Viewer *viewer){
	}

	
};



tk::gui::Viewer *viewer = nullptr;
bool gRun = true;


void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "tkGUI sample gui app");
    cmd.parse();

    signal(SIGINT, sig_handler);
    gRun = true;

	Scene scene;
	scene.init();

    viewer = new tk::gui::Viewer();
    viewer->setWindowName("test");
	viewer->add("scene", &scene);
    viewer->initOnThread();
    viewer->joinThread();
    return 0;
}