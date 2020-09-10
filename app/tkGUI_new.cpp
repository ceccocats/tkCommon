#include "tkCommon/gui/ViewerNew.h"
#include <thread>
#include <signal.h>

bool gRun = true;
tk::gui::ViewerNew* tk::gui::ViewerNew::instance = nullptr;

class Scene : public tk::gui::Drawable {
public:
	void init(){
		
	}

	void draw(tk::gui::Viewer *viewer) {
		glColor4f(1,0,0,1); 

		// BACK
		glBegin(GL_POLYGON);
		glVertex3f(  0.5, -0.5, 0.5 );
		glVertex3f(  0.5,  0.5, 0.5 );
		glVertex3f( -0.5,  0.5, 0.5 );
		glVertex3f( -0.5, -0.5, 0.5 );
		glEnd();
		
		// RIGHT
		glBegin(GL_POLYGON);
		glVertex3f( 0.5, -0.5, -0.5 );
		glVertex3f( 0.5,  0.5, -0.5 );
		glVertex3f( 0.5,  0.5,  0.5 );
		glVertex3f( 0.5, -0.5,  0.5 );
		glEnd();
		
		// LEFT
		glBegin(GL_POLYGON);
		glVertex3f( -0.5, -0.5,  0.5 );
		glVertex3f( -0.5,  0.5,  0.5 );
		glVertex3f( -0.5,  0.5, -0.5 );
		glVertex3f( -0.5, -0.5, -0.5 );
		glEnd();
		
		// TOP
		glBegin(GL_POLYGON);
		glVertex3f(  0.5,  0.5,  0.5 );
		glVertex3f(  0.5,  0.5, -0.5 );
		glVertex3f( -0.5,  0.5, -0.5 );
		glVertex3f( -0.5,  0.5,  0.5 );
		glEnd();
		
		// BOTTOM
		glBegin(GL_POLYGON);
		glVertex3f(  0.5, -0.5, -0.5 );
		glVertex3f(  0.5, -0.5,  0.5 );
		glVertex3f( -0.5, -0.5,  0.5 );
		glVertex3f( -0.5, -0.5, -0.5 );
		glEnd();
	}

	void draw2D(tk::gui::Viewer *viewer){
	}
	
};

void sig_handler(int signo) {
    std::cout<<"request stop\n";
}


int main( int argc, char** argv){
    signal(SIGINT, sig_handler);

    tk::common::CmdParser cmd(argv, "tkGUI new viewer");
    cmd.parse();


    tk::gui::ViewerNew viewer;
	viewer.init();
	
	Scene *scene = new Scene(); // with static does not work
	scene->init();
	viewer.add("scene", scene);
	viewer.run();
    return 0;
}