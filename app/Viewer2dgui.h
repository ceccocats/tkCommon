#pragma once
#include "tkCommon/gui/Viewer.h"

class Viewer2dgui : public tk::gui::Viewer {
    Q_OBJECT

public:
    Viewer2dgui(QWidget *parent = nullptr) {}

public Q_SLOTS:
    void steerActv(int val) {
        std::cout<<"Steer ACTV: "<<val<<"\n";
    }

    void steerUpdate(int val) {
        std::cout<<"Steer VAL: "<<val<<"\n";
    }

    void speedActv(int val) {
        std::cout<<"Speed ACTV: "<<val<<"\n";
    }

    void speedUpdate(int val) {
        std::cout<<"Speed VAL: "<<val<<"\n";
    }
};