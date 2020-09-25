#include "tkCommon/gui/ViewerNew.h"
#include "tkCommon/gui/Buffer.h"
#include "tkCommon/gui/shader/axis.h"
#include "tkCommon/gui/shader/grid.h"
#include "tkCommon/gui/shader/mesh.h"
#include "tkCommon/gui/shader/lines.h"
#include "tkCommon/gui/shader/texture.h"
#include "tkCommon/gui/CommonViewer.h"
#include <thread>
#include <signal.h>

bool gRun = true;
tk::gui::ViewerNew* tk::gui::ViewerNew::instance = nullptr;

class Scene : public tk::gui::Drawable {
public:

	//Axis
	tk::gui::shader::axis axis;

	//Grid
	tk::gui::shader::grid grid;

	//Mesh
	tk::gui::shader::mesh 				mesh;
	std::vector<tk::gui::Buffer<float>> levante;
	tk::gui::common::object3D_t				carObj;

	//Texture
	tk::gui::shader::texture 	text;
	tk::gui::Texture<uint8_t>		texture;
	tk::gui::Buffer<float>		posText2D;
	tk::gui::Buffer<float>		posText3D;

	//Lines
	tk::gui::shader::lines lines;
	tk::gui::Buffer<float> posLines2D;
	tk::gui::Buffer<float> posLines3D;

	//Shader in texture
	tk::gui::Texture<uint8_t> rendering;
	tk::gui::Buffer<float> posrendering;

	//Light
	glm::vec3 lightPos;


	void init(){

		//Light
		lightPos = glm::vec3(0.0f, 0.0f, 20.0f);

		//Axis
		axis.init();
		///////////////


		//Grid
		grid.init();
		///////////////


		//mesh
		mesh.init();

		//Load obj file and fill buffer
		tk::gui::common::loadOBJ(std::string(TKPROJ_PATH) + "data/levante.obj", carObj);
		levante.resize(carObj.meshes.size());
		for(int i = 0; i < carObj.meshes.size(); i++){

			levante[i].init();

			float* data; int n;
			data = carObj.meshes[i].vertexBufferPositionNormal(&n);
			levante[i].setData(data,n);
			delete[] data;

			//fill indices
			levante[i].setIndexVector(carObj.meshes[i].indices.data(),carObj.meshes[i].indices.size());
		}
		///////////////


		//Texture
		text.init();

		//Load texture file and fill buffer
		int width, height, channels;
		uint8_t* img = tk::gui::common::loadImage(std::string(TKPROJ_PATH) + "data/tkLogo.png",&width, &height, &channels);
		texture.init(width, height, channels);
		texture.setData(img);

		//Texture 2D
		float vertices2D[] = {
			//positions				//texture cords
			0.8f,	1.0f, 	0.0f,   	0.0f, 0.0f, 
			0.8f, 	0.8f,	0.0f,   	0.0f, 1.0f,
			1.0f,	0.8f,	0.0f,   	1.0f, 1.0f,
			1.0f,	1.0f,	0.0f,   	1.0f, 0.0f
		};
		unsigned int indices2D[] = {  
			0, 1, 2, // first triangle
			0, 3, 2  // second triangle
		};
		posText2D.init();
		posText2D.setData(vertices2D,21);
		posText2D.setIndexVector(indices2D,6);

		//Texture 3D
		float vertices3D[] = {
			//positions				//texture cords
			6.0f,	6.0f,	0.0f,   	0.5f, 0.0f, 
			3.0f,	6.0f, 	0.0f,   	0.5f, 1.0f,
			3.0f, 	3.0f,	0.0f,   	0.0f, 1.0f,
			6.0f,	3.0f,	0.0f,   	0.0f, 0.0f,
		};
		unsigned int indices3D[] = {  
			0, 1, 2, // first triangle
			0, 3, 2, // second triangle
		};
		posText3D.init();
		posText3D.setData(vertices3D,21);
		posText3D.setIndexVector(indices3D,6);	
		///////////////


		//Lines
		lines.init();

		//Lines 3D
		posLines3D.init();
		float datalines3D[] = {

			//points						//color
			-2.5f,	2.5f,	0.0f,			1.0f,0.0f,0.0f,1.0f,
			2.5f,	2.5f,	0.0f,			0.0f,1.0f,0.0f,1.0f,
			2.5f,	-2.5f,	0.0f,			0.0f,0.0f,1.0f,1.0f,
			-2.5f,	-2.5f,	0.0f,			1.0f,1.0f,0.0f,1.0f
		};
		posLines3D.setData(datalines3D,28);

		//Lines 2D
		posLines2D.init();
		float datalines2D[] = {

			//points						//color
			1.0f,	-1.0f,	0.0f, 			1.0f,0.0f,0.0f,1.0f,
			0.7f,	-1.0f, 	0.0f,			1.0f,0.0f,0.0f,1.0f,
			0.7f, 	-0.7f,	0.0f,			0.0f,1.0f,0.0f,1.0f,
			1.0f,	-0.7f,	0.0f, 			0.0f,1.0f,0.0f,1.0f
		};
		posLines2D.setData(datalines2D,28);
		///////////////


		//Shader write in texture
		rendering.init(800,800,4);

		//Texture 2D
		float verticesCube2D[] = {
			//positions				//texture cords
			-1.0f,	1.0f,	0.0f,   	0.0f, 1.0f, 
			-0.3f,	1.0f, 	0.0f,   	1.0f, 1.0f,
			-0.3f, 	0.3f,	0.0f,   	1.0f, 0.0f,
			-1.0f,	0.3f,	0.0f,   	0.0f, 0.0f
		};
		unsigned int indicesCube2D[] = {  
			0, 1, 2, // first triangle
			0, 3, 2  // second triangle
		};
		posrendering.init();
		posrendering.setData(verticesCube2D,21);
		posrendering.setIndexVector(indicesCube2D,6);
		///////////////
	}

	void drawElements(){

		//Axis
		axis.draw();
		///////////////


		//Grid
		grid.draw();
		///////////////


		//Mesh levante
		for(int i = 0; i < levante.size(); i++){	
			mesh.draw(&levante[i], carObj.meshes[i].indices.size(), lightPos, carObj.meshes[i].color);
		}
		///////////////


		//Texture 2D
		text.draw(&texture,&posText2D,6);			//2 triangles = 6 vertex
		///////////////


		//Texture 2D
		text.draw(&texture,&posText3D,6,true);		//2 triangles = 6 vertex
		///////////////


		//Lines 3D
		lines.draw(&posLines3D,4,2,GL_LINE_LOOP);	//4 vertex vith line size 2 closing loop
		///////////////

		
		//Lines 2D
		lines.draw(&posLines2D,4,2,GL_LINE_LOOP,false);	//4 vertex vith line size 2 closing loop
		///////////////

	}

	void draw(tk::gui::Viewer *viewer){
		
		drawElements();


		//Draw all in texture
		rendering.useForRendering();
		glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		drawElements();

		rendering.unuseRendering();

		text.draw(&rendering,&posrendering,6);			//2 triangles = 6 vertex
		///////////////
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