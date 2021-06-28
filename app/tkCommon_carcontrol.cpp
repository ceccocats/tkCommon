#include "tkCommon/CmdParser.h"
#include "tkCommon/communication/CanInterface.h"
#include "tkCommon/communication/can/VehicleCanParser.h"
#include "tkCommon/joystick/Joystick.h"
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/math/quantile.h"
#include "tkCommon/communication/car/CarControlInterface.h"
#include "tkCommon/rt/Task.h"

//viewer
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawable.h"

tk::gui::Viewer* 	viewer = tk::gui::Viewer::getInstance();
tk::communication::CanInterface canSoc;
tk::communication::CarControl carCtrl;

int main( int argc, char** argv){
    tk::common::CmdParser cmd(argv, "Car Control");
    std::string soc_file = cmd.addArg("interface", "can0", "can interface to read");
    cmd.parse();

    tkASSERT(canSoc.initSocket(soc_file));
    carCtrl.init(&canSoc);
    carCtrl.sendOdomEnable(true);
    //carCtrl.setSteerParams(0,0,0);

    tk::communication::CarControlInterface carInter;
    carInter.init(&carCtrl);

    /*
    // To set actuation
    tk::data::ActuationData act;
    act.speed = 2.0; // m/s
    act.steerAngle = 0 // rad
    carInter.setInput(act)
    */

    viewer->add(&carInter);
    viewer->start();
    
    viewer->join();
    
    carInter.close();
    carCtrl.sendOdomEnable(false);
    carCtrl.close();
    canSoc.close();
    return 0;
}