#include <iostream>
#include <signal.h>
#include <qapplication.h>
#include "tkCommon/gui/Viewer.h"
bool gRun;
tk::gui::Viewer *viewer = nullptr;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    std::cout<<"lidar\n";
    signal(SIGINT, sig_handler);
    gRun = true;

    // viz
    QApplication application(argc, argv);
    viewer = new tk::gui::Viewer();
    viewer->setWindowTitle("tkGUI_empty");
    viewer->show();
    return application.exec();
}