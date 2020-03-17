#include "tkCommon/gui/Viewer.h"
#include <thread>
#include <signal.h>

class MyViewer : public tk::gui::Viewer {
    public:
        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();
        }

        void draw() {
            tk::gui::Viewer::draw();
        }
};

MyViewer *viewer = nullptr;
bool gRun = true;


void sig_handler(int signo) {
    std::cout<<"request stop\n";
    gRun = false;
}

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "tkGUI sample gui app");
    cmd.print();

    signal(SIGINT, sig_handler);
    gRun = true;

    viewer = new MyViewer();
    viewer->setWindowName("test");
    viewer->setBackground(tk::gui::color::DARK_GRAY);
    viewer->init();
    viewer->run();
    return 0;
}