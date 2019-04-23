#include <iostream>
#include <signal.h>
#include <qapplication.h>

#include "ui_Viewer2dgui.h"
class ViewerUI : public QDialog, public Ui::Dialog {
public:
  ViewerUI() { setupUi(this); }
};

bool gRun;
ViewerUI *viewer = nullptr;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, sig_handler);
    gRun = true;

    // viz
    QApplication application(argc, argv);
    viewer = new ViewerUI();
    viewer->setWindowTitle("2D gui");
    viewer->show();

    return application.exec();
}