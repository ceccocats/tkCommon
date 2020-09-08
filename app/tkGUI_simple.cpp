#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>


class Scene : public tk::gui::Drawable {
public:
	tk::gui::Viewer::object3D_t carObj;
	Eigen::MatrixXf cloud;

	tk::gui::Shader glpointcloud;

	tk::gui::Shader glcubes;

	tk::gui::Shader glaxis;

	tk::gui::Shader glcubecloud;

	void init(){
		tk::gui::Viewer::tkLoadOBJ(std::string(TKPROJ_PATH) + "data/levante", carObj);
		
		// dief_vlp32.ply:  
		// wget https://cloud.hipert.unimore.it/s/W3rNtFxRGG5AC89/download
		// mv download dief_vlp32.ply

		std::string vertex;
		std::string geometry;
		std::string fragment;

		vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/pointcloud/pointcloud.vert";
		fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/pointcloud/pointcloud.frag";
		glpointcloud.init(vertex, fragment);

		vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubes/cubes.vert";
		fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubes/cubes.frag";
		geometry 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubes/cubes.geom";
		glcubes.init(vertex, fragment, geometry);

		vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/axis/axis.vert";
		fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/axis/axis.frag";
		geometry 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/axis/axis.geom";
		glaxis.init(vertex, fragment, geometry);

		vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubecloud/cubecloud.vert";
		fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubecloud/cubecloud.frag";
		geometry 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/cubecloud/cubecloud.geom";
		glcubecloud.init(vertex, fragment, geometry);

		cloud.resize(3, 10e6);
		std::ifstream is("dief_vlp32.ply");
		float x,y,z;
		int count =0;
		float min = 9999;
		float max = -9999;
		while (is >> x >> y >> z) {
			cloud.col(count) << x, y, z;
			count++;
			min = min > z ? z : min;
			max = max < z ? z : max;
		}
		cloud.conservativeResize(3, count);
		std::cout<<"cloud size: "<<count<<"\n";
		std::cout<<min<<".."<<max<<std::endl;
	}

	void draw(tk::gui::Viewer *viewer){

		//get modelview matrix
		glm::mat4 modelview;
		glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 

		glcubecloud.use();
		glcubecloud.setMat4("modelview", modelview);
		glcubecloud.setFloat("size",0.1f);

		unsigned int cloudVAO = 0;
		unsigned int cloudVBO = 0;
		glGenVertexArrays(1, &cloudVAO);
		glGenBuffers(1, &cloudVBO);
		glBindVertexArray(cloudVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cloudVBO);
		glBufferData(GL_ARRAY_BUFFER, cloud.size() * sizeof(float), cloud.data(), GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glBindVertexArray(0);

		glBindVertexArray(cloudVAO);
		glDrawArrays(GL_POINTS, 0, cloud.cols());
		glBindVertexArray(0);
		glcubecloud.release();

		glDeleteVertexArrays(1, &cloudVAO);
		glDeleteBuffers(1, &cloudVBO);	


		//cubes

		/*int Ncubes = 100000;
		float cubes[Ncubes*12];
		for(int i=0; i<Ncubes; i++) {
			cubes[12*i +0] = 50* float(rand()%1000) / 1000;
			cubes[12*i +1] = 50* float(rand()%1000) / 1000;
			cubes[12*i +2] = 50* float(rand()%1000) / 1000;
			cubes[12*i +3] = float(rand()%1000) / 1000;
			cubes[12*i +4] = float(rand()%1000) / 1000;
			cubes[12*i +5] = float(rand()%1000) / 1000;
			cubes[12*i +6] = float(rand()%1000) / 1000;
			cubes[12*i +7] = float(rand()%1000) / 1000;
			cubes[12*i +8] = float(rand()%1000) / 1000;
			cubes[12*i +9]  = float(rand()%1000) / 1000;
			cubes[12*i +10] = float(rand()%1000) / 1000;
			cubes[12*i +11] = float(rand()%1000) / 1000;
		}*/

		float cube[] = {
			2.0f,	// x center
			2.0f,	// y center
			2.0f,	// z center
			0.2f,		// x size
			0.2f,		// y size
			1.0f,		// z size
			(float)toRadians(45.0f),	// yaw
			(float)toRadians(45.0f),	// pitch
			(float)toRadians(90.0f),	// roll
			1.0f,	//red
			1.0f,	//green
			0.0f	//blue
		};



		unsigned int cubesVAO	= 0;
		unsigned int cubesVBO	= 0;
		glGenVertexArrays(1, &cubesVAO);
		glGenBuffers(1, &cubesVBO);


		glBindVertexArray(cubesVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cubesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube), &cube, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), 0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 12 * sizeof(float), (void*)(9 * sizeof(float)));
		glBindVertexArray(0);

		glcubes.use();
		glcubes.setMat4("modelview", modelview);

		glBindVertexArray(cubesVAO);
        glDrawArrays(GL_POINTS, 0, 1);
		glBindVertexArray(0);

		glDeleteVertexArrays(1, &cubesVAO);
		glDeleteBuffers(1, &cubesVBO);
		glcubes.release();	


		//axis
		glaxis.use();
		glaxis.setMat4("modelview", modelview);
		glaxis.setInt("width",glutGet(GLUT_SCREEN_WIDTH));
		glaxis.setInt("height",glutGet(GLUT_SCREEN_HEIGHT));
		glLineWidth(3.0);
		glDrawArrays(GL_POINTS, 0, 1);
		glLineWidth(1.0);
		glaxis.release();
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


    viewer = new tk::gui::Viewer();
	
    viewer->setWindowName("test");
    viewer->init();
	Scene scene;
	scene.init();
	viewer->add("scene", &scene);
    viewer->run();
    return 0;
}