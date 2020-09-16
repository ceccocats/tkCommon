#include "tkCommon/gui/ViewerNew.h"
#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/components/axis/axis.h"
#include "tkCommon/gui/components/grid/grid.h"
#include "tkCommon/gui/components/mesh/mesh.h"
#include <thread>
#include <signal.h>

bool gRun = true;
tk::gui::ViewerNew* tk::gui::ViewerNew::instance = nullptr;

class Scene : public tk::gui::Drawable {
public:
	tk::gui::components::axis axis;
	tk::gui::components::grid grid;
	tk::gui::components::mesh mesh;

	tk::gui::Viewer::object3D_t carObj;
	//tk::gui::tkShader		mesh;
	static const int levanteMesh = 16;
	tk::gui::tkBufferGl<float>	carbuffer[levanteMesh];


	tk::gui::tkBufferGl<float>	test;

	void init(){
		tk::gui::Viewer::tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);

		float* vec = new float[250000 * 6];
		for(int m = 0; m < carObj.triangles.size(); m++){
			int n = carObj.triangles[m].cols();

			for(int i = 0; i < n; i++){
				vec[i * 6 + 0] = carObj.triangles[m](0,i);
				vec[i * 6 + 1] = carObj.triangles[m](1,i);
				vec[i * 6 + 2] = carObj.triangles[m](2,i);
				vec[i * 6 + 3] = carObj.triangles[m](3,i);
				vec[i * 6 + 4] = carObj.triangles[m](4,i);
				vec[i * 6 + 5] = carObj.triangles[m](5,i);
			}
			carbuffer[m].init();
			carbuffer[m].setData(vec,n*6);
		}

		axis.init();
		grid.init();
		mesh.init();
	}

	void draw(tk::gui::Viewer *viewer){

		glm::vec3 lightPos(0.0f, 0.0f, 20.0f);

		//axis
		axis.draw();

		// 3d grid
		grid.draw();

		//draw levante
		for(int m = 0; m < levanteMesh; m++){
			tk::gui::Color_t color= tk::gui::color4f(carObj.colors[m].x, carObj.colors[m].y, carObj.colors[m].z, 1);
			mesh.draw(&carbuffer[m], carObj.triangles[m].cols(), lightPos, color);
		}
	}

	void draw2D(tk::gui::Viewer *viewer){
	}
};


tk::gui::ViewerNew viewer;

void sig_handler(int signo) {
    std::cout<<"request stop\n";
	viewer.close();
}


int main( int argc, char** argv){
    signal(SIGINT, sig_handler);

    tk::common::CmdParser cmd(argv, "tkGUI new viewer");
    cmd.parse();


	viewer.init();
	
	Scene *scene = new Scene(); // with static does not work
	scene->init();
	viewer.add("scene", scene);
	viewer.run();
    return 0;
}