#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawables.h"

#include "tkCommon/data/CloudData.h"
#include "tkCommon/data/GpsData.h"
#include "tkCommon/data/ImuData.h"

#include "tkCommon/rt/Task.h"

tk::gui::Viewer* 	viewer = tk::gui::Viewer::getInstance();

void* th_gps(void* gpsptr){

	tk::data::GpsData* gps = (tk::data::GpsData*)gpsptr;

	gps->lat = 42.182945;
	gps->lon = 12.477200;
	gps->heigth = 0.1f;
	gps->sats   = 20;
	gps->header.name = "fake_gps";

	tk::rt::Task t;
    t.init(1000);

	while(viewer->isRunning()){
		gps->lockWrite();
		gps->lat += 0.00000001;
		gps->unlockWrite();
		t.wait();
	}
}


void* th_cloud(void* cloudptr){
	tk::data::CloudData* cloud = (tk::data::CloudData*)cloudptr;

	int h_n = 100;
	int N = 360*h_n;
	double d = 5.0;
	double h_d = 0.1;
	Eigen::MatrixXf points = Eigen::MatrixXf(4, N);

	tk::rt::Task t;
    t.init(1000);

	float rot = 0;
	cloud->header.name = "fake_cloud";
	while(viewer->isRunning()){

		for(int i=0; i<N; i++) {
			int h_idx  = i/360;
			double ang = double(i % 360)/180.0*M_PI;
			double z   = h_d*(N/360) - h_d*h_idx;
			double r   = (z/5)*(z/5);

			points(0,i) = cos(ang)*(r) + cos(rot)*(10);
			points(1,i) = sin(ang)*(r) + sin(rot)*(10);
			points(2,i) = z;
			points(3,i) = 1;
		}

		cloud->lockWrite();
		cloud->points.copyFrom(points.data(),points.rows(),points.cols());
		cloud->unlockWrite();

		rot += 0.01;
		rot = fmod(rot,M_PI*2);

		t.wait();
	}
}


void* th_plt(void* ptrplt){
	tk::gui::Plot* plt = (tk::gui::Plot*)ptrplt;
	tk::common::Tfpose pose = tk::common::Tfpose::Identity();
	float r = 0.01f;
	float a = 0.0001f;

	tk::rt::Task t;
    t.init(10000);

	while(viewer->isRunning()){
		pose.matrix()(0,3) = sin(a)*r;
		pose.matrix()(1,3) = cos(a)*r;
		r+=0.01;
		a+=0.001;
		plt->addPoint(pose);
		t.wait();
	}
}


void* th_imu(void* ptrimu){
	tk::data::ImuData* imu = (tk::data::ImuData*)ptrimu;
	imu->header.name  = "fake_imu";
	imu->header.stamp = 0;

	tk::rt::Task t;
    t.init(10000);

	float r = 0.01f;
	float d = 1.0f;

	while(viewer->isRunning()){

		imu->lockWrite();
		imu->acc.x() = sin(r)*d;
		imu->acc.y() = cos(r)*d;
		imu->acc.z() = tan(r)*d;
		imu->unlockWrite();
		r+=0.01;
		imu->header.stamp += 10000;

		t.wait();
	}
}

void key_listener(int key, int action, int source) {
	std::cout<<"KEY: "<<key<<" action: "<<action<<" source: "<<source<<"\n";
}

int main(int argc, char* argv[]){
	tk::common::CmdParser cmd(argv, "tkGUI_test");
    cmd.parse();

	//Data
	tk::data::CloudData cloud;
	tk::data::GpsData	gps;
	tk::data::ImuData   imu;

	//PlotManager
	tk::gui::Plot* 		plt;
	plt = new tk::gui::Plot("plot", 1000, tk::gui::Plot::type_t::LINE, 1);

	//Text
	tk::gui::Text *text = new tk::gui::Text("Hello");
	text->tf = tk::common::odom2tf(5, 5, 2, 0.3, 0.2, 0.5);

	viewer->start();
	viewer->addKeyCallback(key_listener);

	//Viewer insert
	viewer->add(new tk::gui::Grid());
	viewer->add(new tk::gui::Axis());
	viewer->add(new tk::gui::Mesh(std::string(tkCommon_PATH) + "data/levante.obj"));
	viewer->add(new tk::gui::Cloud4f(&cloud,"tornado"));
	viewer->add(new tk::gui::Gps(&gps));
	viewer->add(new tk::gui::Imu(&imu));
	viewer->add(plt);
	viewer->add(text);

	tk::rt::Thread th1,th2,th3,th4;

    th1.init(th_cloud, (void*)&cloud);
	th2.init(th_gps, (void*)&gps);
	th3.init(th_plt, (void*)plt);
	th4.init(th_imu, (void*)&imu);

	viewer->join();

	th1.join();
	th2.join();
	th3.join();
	th4.join();

	return 0;
}