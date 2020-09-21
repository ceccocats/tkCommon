#include "tkCommon/gui/ViewerNew.h"
#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/components/axis/axis.h"
#include "tkCommon/gui/components/grid/grid.h"
#include "tkCommon/gui/components/mesh/mesh.h"
#include "tkCommon/gui/components/texture/texture.h"
#include "tkCommon/gui/CommonViewer.h"
#include <thread>
#include <signal.h>

bool gRun = true;
tk::gui::ViewerNew* tk::gui::ViewerNew::instance = nullptr;

class Scene : public tk::gui::Drawable {
public:
	tk::gui::components::axis axis;
	tk::gui::components::grid grid;
	tk::gui::components::mesh mesh;
	tk::gui::components::texture text;

	tk::gui::common::object3D_t carObj;
	std::vector<tk::gui::tkBufferGl<float>> levante;

	tk::gui::tkTexture<uint8_t>	texture;
	std::vector<tk::common::Vector3<float>> posTexture;
	tk::gui::tkBufferGl<float>	test;


	tk::gui::tkBufferGl<float> carbuffer[16];

	void init(){
		//tk::gui::common::loadOBJ(std::string(TKPROJ_PATH) + "data/levante.obj", carObj);

		tk::gui::common::loadOBJ("/home/alice/Downloads/Car.obj", carObj);

		levante.resize(carObj.meshes.size());
		for(int i = 0; i < carObj.meshes.size(); i++){

			levante[i].init();

			float* data; int n;
			data = carObj.meshes[i].vertexBufferPositionNormal(&n);
			levante[i].setData(data,n);
			delete[] data;

			levante[i].setIndices(carObj.meshes[i].indices.data(),carObj.meshes[i].indices.size());
		}

		axis.init();
		grid.init();
		mesh.init();
		text.init();

		int width, height, channels;
		uint8_t* img = tk::gui::common::loadImage("/home/alice/Downloads/Car.png",&width, &height, &channels);
		//uint8_t* img = tk::gui::common::loadImage(std::string(TKPROJ_PATH) + "data/tkLogo.png",&width, &height, &channels);
		texture.init(width, height, channels);
		texture.setData(img);

		float* data1; int n1;
		data1 = carObj.meshes[0].vertexBufferPositionTextcoord(&n1);

		test.init();
		test.setData(data1,n1);
		test.setIndices(carObj.meshes[0].indices.data(),carObj.meshes[0].indices.size());
		delete[] data1;



		/*float vertices[] = {
			//positions				//texture cords
			1.0f,  1.0f, 0.0f,   	1.0f, 0.0f, 
			0.5f, 1.0f, 0.0f,   	1.0f, 1.0f,
			0.5f, 0.5f, 0.0f,   	0.0f, 1.0f,
			1.0f,  0.5f, 0.0f,   	0.0f, 0.0f
		};
		unsigned int indices[] = {  
			0, 1, 3, // first triangle
			1, 2, 3  // second triangle
		};

		test.init();
		test.setData(vertices,21);
		test.setIndices(indices,6);	*/
	}

	void draw(tk::gui::Viewer *viewer){

		glm::vec3 lightPos(0.0f, 0.0f, 20.0f);

		//axis
		axis.draw();

		// 3d grid
		grid.draw();

		//draw levante
		/*for(int i = 0; i < levante.size(); i++){
			mesh.draw(&levante[i], carObj.meshes[i].indices.size(), lightPos, carObj.meshes[i].color);
		}*/


		text.draw(&texture,&test,carObj.meshes[0].indices.size(),true);
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