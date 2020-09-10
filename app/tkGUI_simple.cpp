#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>

#include "tkCommon/gui/components/pointcloud3f/pointcloud3f.h"
#include "tkCommon/gui/components/pointcloud4f/pointcloud4f.h"
#include "tkCommon/gui/components/axis/axis.h"


class Scene : public tk::gui::Drawable {
public:
	tk::gui::Viewer::object3D_t carObj;
	Eigen::MatrixXf cloud;

	/*tk::gui::Shader glpointcloud;
	tk::gui::Shader glcubes;
	tk::gui::Shader glaxis;
	tk::gui::Shader glcubecloud;
	tk::gui::Shader gllightcubecloud;*/

	tk::gui::tkBufferGl<float>	pointcloud;

	tk::gui::components::pointcloud4f	pointcloud4f;

	tk::gui::components::axis			axis;

	tk::gui::tkShader		mesh;

	static const int levanteMesh = 16;
	tk::gui::tkBufferGl<float>	carbuffer[levanteMesh];


	int n = 0;

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



		pointcloud4f.init();
		axis.init();
		
		// dief_vlp32.ply:  
		// wget https://cloud.hipert.unimore.it/s/W3rNtFxRGG5AC89/download
		// mv download dief_vlp32.ply



		/*std::string vertex;
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

		vertex 		= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/lightcubecloud/lightcubecloud.vert";
		fragment 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/lightcubecloud/lightcubecloud.frag";
		geometry 	= std::string(TKPROJ_PATH) + "include/tkCommon/gui/shaders/lightcubecloud/lightcubecloud.geom";
		gllightcubecloud.init(vertex, fragment, geometry);*/

		cloud.resize(4, 10e6);
		std::ifstream is("dief_vlp32.ply");
		float x,y,z;
		int count = 0;
		while (is >> x >> y >> z) {
			cloud.col(count) << x, y, z, 0;
			count++;
		}
		cloud.conservativeResize(4, count);

		pointcloud.init();
		pointcloud.setData(cloud.data(),cloud.size()/2);

		n = cloud.cols()/2;
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
		
	

		//pointcloud
		/*pointcloud4f.draw(&pointcloud,n,tk::gui::color::LIGHT_GREEN);

		static int i = 0;
		i++;
		if(i == 60){
			n = cloud.cols();
			pointcloud.setData(cloud.data() + cloud.size()/2 ,cloud.size()/2, cloud.size()/2);

			//pointcloud.setData(cloud.data() + cloud.size()/2, cloud.size()/2);
		}
		std::cout<<i<<std::endl;




		//cubes
		static float angle = 0;
		angle += 0.01;

		float cube[] = {
			2.0f,	// x center
			2.0f,	// y center
			7.0f,	// z center
			4.2f,		// x size
			4.2f,		// y size
			4.0f,		// z size
			angle,	// roll
			angle,	// pitch
			angle,	// yaw
			1.0f,	//red
			0.5f,	//green
			0.31f,	//blue
			1.0f,	//alpha

			lightPos.x,	// x center
			lightPos.y,	// y center
			lightPos.z,	// z center
			1.0f,		// x size
			1.0f,		// y size
			1.0f,		// z size
			(float)toRadians(0.0f),	// roll
			(float)toRadians(0.0f),	// pitch
			(float)toRadians(0.0f),	// yaw
			1.0f,	//red
			1.0f,	//green
			1.0f	//blue
		};








		//get modelview matrix
		glm::mat4 modelview;
		glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 

		//gllightcubecloud.use();
		//gllightcubecloud.setMat4("modelview", modelview);
		//gllightcubecloud.setFloat("size",0.03f);
		//gllightcubecloud.setVec3("lightPos",lightPos);

		glpointcloud.use();
		glpointcloud.setMat4("modelview", modelview);

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
		glpointcloud.unuse();

		glDeleteVertexArrays(1, &cloudVAO);
		glDeleteBuffers(1, &cloudVBO);	
		//pointcloud.setData(cloud.data(),cloud.size());

		pointcloud.use();
		glpointcloud.use();
		glpointcloud.setMat4("modelview", modelview);
		glDrawArrays(GL_POINTS, 0, cloud.cols());
		pointcloud.unuse();
		glpointcloud.unuse();



		


		//cubes

		int Ncubes = 100000;
		float cubes[Ncubes*12];
		for(int i=0; i<Ncubes; i++) {
			cubes[12*i +0] = 10 + 50* float(rand()%1000) / 1000;
			cubes[12*i +1] = 10 + 50* float(rand()%1000) / 1000;
			cubes[12*i +2] = 10 + 50* float(rand()%1000) / 1000;
			cubes[12*i +3] = float(rand()%1000) / 1000;
			cubes[12*i +4] = float(rand()%1000) / 1000;
			cubes[12*i +5] = float(rand()%1000) / 1000;
			cubes[12*i +6] = float(rand()%1000) / 1000;
			cubes[12*i +7] = float(rand()%1000) / 1000;
			cubes[12*i +8] = float(rand()%1000) / 1000;
			cubes[12*i +9]  = float(rand()%1000) / 1000;
			cubes[12*i +10] = float(rand()%1000) / 1000;
			cubes[12*i +11] = float(rand()%1000) / 1000;
		}

		static float a = 0;
		a += 1;

		float cube[] = {
			2.0f,	// x center
			2.0f,	// y center
			7.0f,	// z center
			4.2f,		// x size
			4.2f,		// y size
			4.0f,		// z size
			(float)toRadians(a),	// roll
			(float)toRadians(a),	// pitch
			(float)toRadians(a),	// yaw
			1.0f,	//red
			0.5f,	//green
			0.31f,	//blue

			lightPos.x,	// x center
			lightPos.y,	// y center
			lightPos.z,	// z center
			1.0f,		// x size
			1.0f,		// y size
			1.0f,		// z size
			(float)toRadians(0.0f),	// roll
			(float)toRadians(0.0f),	// pitch
			(float)toRadians(0.0f),	// yaw
			1.0f,	//red
			1.0f,	//green
			1.0f	//blue
		};

		unsigned int cubesVAO	= 0;
		unsigned int cubesVBO	= 0;
		glGenVertexArrays(1, &cubesVAO);
		glGenBuffers(1, &cubesVBO);


		glBindVertexArray(cubesVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cubesVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cubes), &cubes, GL_DYNAMIC_DRAW);
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
		glcubes.setVec3("lightPos", lightPos);

		//tk::common::Vector3<float> pos = viewer->mouseView.getWorldPos();

		//glm::vec3 camera(pos.x,pos.y,pos.z);

		//glcubes.setVec3("viewPos", camera);

		glBindVertexArray(cubesVAO);
        glDrawArrays(GL_POINTS, 0, Ncubes);
		glBindVertexArray(0);

		glDeleteVertexArrays(1, &cubesVAO);
		glDeleteBuffers(1, &cubesVBO);
		glcubes.unuse();	

		//axis
		glaxis.use();
		glaxis.setMat4("modelview", modelview);
		glaxis.setInt("width",glutGet(GLUT_SCREEN_WIDTH));
		glaxis.setInt("height",glutGet(GLUT_SCREEN_HEIGHT));
		glLineWidth(3.0);
		glDrawArrays(GL_POINTS, 0, 1);
		glLineWidth(1.0);
		glaxis.unuse();*/
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