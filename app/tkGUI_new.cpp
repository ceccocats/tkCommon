#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/Drawables/Axis.h"
#include "tkCommon/gui/Drawables/Grid.h"
#include "tkCommon/gui/Drawables/Cloud4f.h"
#include "tkCommon/gui/Drawables/Mesh.h"
#include "tkCommon/gui/Drawables/Gps.h"
#include "tkCommon/gui/Drawables/Plot.h"


#include "tkCommon/data/CloudData.h"
#include "tkCommon/data/GPSData.h"


#include <thread>


tk::data::GPSData	gps;
tk::gui::Viewer* 	viewer = tk::gui::Viewer::getInstance();

void read_cloud(tk::data::CloudData& cloud) {

	tk::math::MatIO mat;
	mat.open("/home/alice/Documents/dataset/masa_ouster.mat");

	gps.lat = 42.182945;
	gps.lon = 12.477200;
	gps.heigth = 0.0f;
	gps.header.name = "GPS";

	cloud.header.name = "LiDAR cloud";

	cloud.features.add(tk::data::CloudData_gen::FEATURES_I);

	Eigen::MatrixXf points;
	Eigen::MatrixXf intensity;
	tk::math::MatIO::var_t var;
	for(int i=0; i<mat.size() && viewer->isRunning(); i++) {

		mat.read(mat[i], var);
		var["lidar"]["points"].get(points);

		mat.read(mat[i], var);
		var["lidar"]["f_intensity"].get(intensity);

		var.release();

		cloud.lock();

		cloud.points.copyFrom(points.data(),points.rows(),points.cols());
		cloud.features[tk::data::CloudData_gen::FEATURES_I].copyFrom(intensity.data(),intensity.rows(),intensity.cols());
		//cloud.gammaCorrectionIntensity();

		cloud.unlockWrite();

		gps.lock();
		gps.lat += 0.000001;
		gps.unlockWrite();

		usleep(100000);
	}

	mat.close();
}

int main(){

	/*tk::data::CloudData cloud;
	Eigen::MatrixXf points2;
	tk::math::MatIO mat;
	mat.open("/home/alice/cloud.mat");
	tk::math::MatIO::var_t var;
	mat.read("cloud",var);
	var["points"].get(points2);
	var["f_intensity"].get(intensity);//
	var.release();
	points2.conservativeResize(4,1000000);
	cloud.points.copyFrom(points2.data(),points2.rows(),points2.cols());
	cloud.header.name = "LiDAR cloud";
	cloud.notifyUpdate();*/

	tk::data::CloudData cloud;

	viewer->start();

	tk::gui::Plot *plt = new tk::gui::Plot("provaPlot", 10000, tk::gui::Plot::type_t::LINE, 1);
	viewer->add(new tk::gui::Grid());
	viewer->add(new tk::gui::Axis());
	viewer->add(new tk::gui::Mesh(std::string(tkCommon_PATH) + "data/levante.obj"));
	viewer->add(new tk::gui::Cloud4f(&cloud));
	viewer->add(new tk::gui::Gps(&gps));
	viewer->add(plt);

	read_cloud(cloud);
	//std::thread read_cloud_th(read_cloud);

	viewer->join();
	//read_cloud_th.join();
}



/*#include "tkCommon/gui/Viewer.h"
#include "tkCommon/data/CloudData.h"
#include "tkCommon/data/VehicleData.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/math/MatIO.h"
#include <thread>
#include <signal.h>


tk::gui::Viewer viewer;
tk::data::CloudData ldata;
tk::data::ImageData img;
tk::data::VehicleData veh;
tk::data::VehicleData veh2;


class Scene : public tk::gui::Drawable {
public:

	//Mesh
	tk::gui::shader::mesh 				mesh;
	std::vector<tk::gui::Buffer<float>> levante;
	tk::gui::common::object3D_t				carObj;

	//Texture
	tk::gui::shader::texture 	text;
	tk::gui::Texture<uint8_t>   texture;
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


	//heightmap
	tk::gui::shader::heightmap map;
	tk::math::Mat<float>	points;
	tk::math::Mat<float>	colors;

    //cubecloud
    tk::gui::shader::cubes3D cubes;
    tk::gui::Buffer<float>   buffercloud;
    int n_cubes;


	void init(){

		//Light
		lightPos = glm::vec3(0.0f, 0.0f, 20.0f);

		//mesh
		mesh.init();

		//Load obj file and fill buffer
		tk::gui::common::loadOBJ(std::string(tkCommon_PATH) + "data/levante.obj", carObj);
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
		rendering.init(800,800,4,true);

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


        Eigen::MatrixXf points2;
        Eigen::MatrixXf pr,pg,pb;

        tk::math::MatIO mat;
        mat.open("/home/alice/cloud.mat");
        tk::math::MatIO::var_t var;
        mat.read("cloud",var);
        var["points"].get(points2);
        var["f_r"].get(pr);
        var["f_g"].get(pg);
        var["f_b"].get(pb);
        var.release();
        //points2.conservativeResize(4,10000000);

        buffercloud.init();
        cubes.init();
        float* temp = new float[points2.cols() * 13];
        for(int i = 0; i < points2.cols(); i++){
            temp[i * 13 +  0] = points2(0,i); 
            temp[i * 13 +  1] = points2(1,i);
            temp[i * 13 +  2] = points2(2,i);
            temp[i * 13 +  3] = 0.1;
            temp[i * 13 +  4] = 0.1;
            temp[i * 13 +  5] = 0.1;
            temp[i * 13 +  6] = 0;
            temp[i * 13 +  7] = 0;
            temp[i * 13 +  8] = 0;
            temp[i * 13 +  9] = pr(i);
            temp[i * 13 + 10] = pg(i);
            temp[i * 13 + 11] = pb(i);
            temp[i * 13 + 12] = 1;
        }

        n_cubes = points2.cols();

        buffercloud.setData(temp,points2.cols() * 13);
        delete [] temp;

        std::cout<<points.cols()<<std::endl;

        //ldata.tf = tk::common::Tfpose::Identity();

        //ldata.points.copyFrom(points2.data(),points2.rows(),points2.cols());




		//heightmap
		map.init();

		points.resize(1,90);
		points.data_h[0] = 6;
		points.data_h[1] = 6;
		points.data_h[2] = 2;

		points.data_h[3] = 10;
		points.data_h[4] = 6;
		points.data_h[5] = 0;

		points.data_h[6] = 6;
		points.data_h[7] = 10;
		points.data_h[8] = 0;

		points.data_h[9]  = 10;
		points.data_h[10] = 10;
		points.data_h[11] = 2;

		colors.resize(1,90);

		colors.data_h[0] = 1;
		colors.data_h[1] = 0;
		colors.data_h[2] = 0;

		colors.data_h[3] = 0;
		colors.data_h[4] = 1;
		colors.data_h[5] = 0;

		colors.data_h[6] = 0;
		colors.data_h[7] = 1;
		colors.data_h[8] = 0;

		colors.data_h[9]  = 0;
		colors.data_h[10] = 0;
		colors.data_h[11] = 1;
		///////////////
	}

	void drawElements(){

		//Mesh levante
		for(int i = 0; i < levante.size(); i++){	
			//mesh.draw(&levante[i], carObj.meshes[i].indices.size(), lightPos, carObj.meshes[i].color);
		}
		///////////////


		//Texture 2D
		text.draw(&texture,&posText2D,6);			//2 triangles = 6 vertex
		///////////////


		//Texture 2D
		text.draw(&texture,&posText3D,6);		//2 triangles = 6 vertex
		///////////////


		//Lines 3D
		lines.draw(&posLines3D,4,2,GL_LINE_LOOP);	//4 vertex vith line size 2 closing loop
		///////////////

		
		//Lines 2D
		lines.draw(&posLines2D,4,2,GL_LINE_LOOP);	//4 vertex vith line size 2 closing loop
		///////////////

        cubes.draw(&buffercloud,n_cubes,lightPos);


		//map.draw(points,colors,2,2);

	}

	void draw(tk::gui::Viewer *viewer){
		
		drawElements();


		//Draw all in texture
		/*rendering.useForRendering();
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		drawElements();

		rendering.unuseRendering();*/

		/*text.draw(&rendering,&posrendering,6);			//2 triangles = 6 vertex
		///////////////
	}

	void draw2D(tk::gui::Viewer *viewer){
	}
};

void sig_handler(int signo) {
    std::cout<<"request stop\n";
	viewer.close();
}

void read_cloud() {

	tk::math::MatIO mat;
	mat.open("/media/alice/FerrariRecs1/datasets/balocco_velarray_camera.mat");

	Eigen::MatrixXf points;
	tk::math::MatIO::var_t var;
	for(int i=0; i<mat.size() && viewer.isRunning(); i++) {
		tk::common::Tfpose baseTf = tk::common::odom2tf(i*0.3, 0, 0);

		veh.tf = baseTf * tk::common::odom2tf(0, 0, M_PI);
		ldata.tf = baseTf;
		veh2.tf = tk::common::odom2tf(0, 0, M_PI_4) * baseTf * tk::common::odom2tf(0, 0, M_PI);

		mat.read(mat[i], var);
		var["lidar"]["points"].get(points);
		var.release();

		ldata.lock();
		ldata.points.copyFrom(points.data(),points.rows(),points.cols());
		ldata.unlock();

		usleep(10000);
	}

	mat.close();
}


int main( int argc, char** argv){
    signal(SIGINT, sig_handler);

    tk::common::CmdParser cmd(argv, "tkGUI new viewer");
    cmd.parse();


	viewer.init();
	
	Scene scene; // with static does not work
	scene.init();
	viewer.add("scene", &scene);
	
	/*ldata.init();
	viewer.add("lidar", &ldata);
	std::thread read_cloud_th(read_cloud);	*/
		//img.tf.linear() = img.tf.linear() * 10;
		//int w, h, c;
		//uint8_t* image = tk::gui::common::loadImage(std::string(tkCommon_PATH) + "data/tkLogo.png", &w, &h, &c);
	//img.data.copyFrom(image,1, w*h*c);

	//tk::gui::common::loadOBJ(std::string(TKPROJ_PATH) + "data/levante.obj", veh.carObj);
	//tk::gui::common::loadOBJ(std::string(TKPROJ_PATH) + "data/levante.obj", veh2.carObj);
/*
	ldata.init();
	viewer.add("lidar", &ldata);
	//viewer.add("image", &img);
	viewer.add("vehicle", &veh);
	viewer.add("vehicle2", &veh2);
	//std::thread read_cloud_th(read_cloud);	

	viewer.run();

	//read_cloud_th.join();
    return 0;
}*/