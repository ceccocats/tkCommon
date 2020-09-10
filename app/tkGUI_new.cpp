#include "tkCommon/gui/ViewerNew.h"
#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/components/axis/axis.h"
#include <thread>
#include <signal.h>

bool gRun = true;
tk::gui::ViewerNew* tk::gui::ViewerNew::instance = nullptr;

class Scene : public tk::gui::Drawable {
public:
	tk::gui::components::axis axis;

	tk::gui::Viewer::object3D_t carObj;
	tk::gui::tkShader		mesh;
	static const int levanteMesh = 16;
	tk::gui::tkBufferGl<float>	carbuffer[levanteMesh];

	void init(){
		tk::gui::Viewer::tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);

		std::vector<tk::gui::vertexAttribs_t> att;
		att.push_back({3,18,0});
		att.push_back({3,18,3});
		att.push_back({3,18,6});
		att.push_back({3,18,9});
		att.push_back({3,18,12});
		att.push_back({3,18,15});

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
			carbuffer[m].pushVectorVertex(att);
			carbuffer[m].setVertexAttrib();
		}
		std::string vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/mesh/mesh.vert";
		std::string fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/mesh/mesh.frag";
		std::string geometry 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/mesh/mesh.geom";
		mesh.init(vertex, fragment, geometry);

		axis.init();	
	}

	void draw(tk::gui::Viewer *viewer){

		glm::vec3 lightPos(0.0f, 0.0f, 20.0f);

		//axis
		axis.draw();

		glm::mat4 modelview;
		glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview));

		mesh.use();
		mesh.setMat4("modelview", modelview);
		mesh.setVec3("lightPos",lightPos);
		for(int m = 0; m < levanteMesh; m++){
			glm::vec3 color(carObj.colors[m].x, carObj.colors[m].y, carObj.colors[m].z);
			mesh.setVec3("color",color);
			carbuffer[m].use();
			glDrawArrays(GL_POINTS, 0, carObj.triangles[m].cols());
			carbuffer[m].unuse();

		}
		mesh.unuse();
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