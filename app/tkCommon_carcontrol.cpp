#include "tkCommon/CmdParser.h"
#include "tkCommon/communication/CanInterface.h"
#include "tkCommon/communication/can/VehicleCanParser.h"
#include "tkCommon/joystick/Joystick.h"
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/math/quantile.h"
#include "tkCommon/communication/car/CarControl.h"
#include "tkCommon/rt/Task.h"

//viewer
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawable.h"

tk::gui::Viewer* 	viewer = tk::gui::Viewer::getInstance();

tk::communication::CanInterface canSoc;
tk::communication::CarControl carCtrl;
class Control : public tk::gui::Drawable{
    public:
        Control() {
            pid.init(0.2, 0, 0, -1.0, 1.0);
        }
        ~Control() {}

        tk::common::PID pid;

        bool active = false;
        bool setZero = false;
        float steerReqDeg = 0;
        float steerSpeed = 65536; 

        float brakeReq = 0;
        float throttleReq = 0;
        float speedReqKMH = 0;
        float speedReq = 0;

        bool speedControl = false;
        bool odomActive = true;

        void draw(tk::gui::Viewer *viewer){
            ImGui::Begin("Car control", NULL, ImGuiWindowFlags_NoScrollbar);
            if(ImGui::Checkbox("ACTIVATE", &active)) {
                std::cout<<"Set CAR state: "<<active<<"\n";
                carCtrl.enable(active);
            }
            if(ImGui::Button("SET ZERO")) {
                carCtrl.setSteerZero();
            }
            //if(ImGui::Button("RESET")) {
            //    carCtrl.resetSteerMotor();
            //}
            ImGui::SliderFloat("Steer", &steerReqDeg, -30, +30);
            ImGui::InputFloat("Steer_", &steerReqDeg, -30, +30);
            ImGui::SliderFloat("SteerSpeed", &steerSpeed, 0, 65536);
            ImGui::Text("steer pos: %d", carCtrl.steerPos);

            ImGui::NewLine();
            ImGui::Checkbox("SpeedControl", &speedControl);
            if(!speedControl) {
                ImGui::SliderFloat("Brake", &brakeReq, 0, 1.0);
                ImGui::InputFloat("Brake_", &brakeReq, 0, 1.0);
                ImGui::SliderFloat("Throttle", &throttleReq, 0, 1.0);
                ImGui::InputFloat ("Throttle_", &throttleReq, 0, 1.0);
            } else {
                ImGui::SliderFloat("Speed kmh", &speedReqKMH, -1, 40);
                speedReq = speedReqKMH/3.6;
                ImGui::ProgressBar(throttleReq, ImVec2(0.0f, 0.0f));
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                ImGui::Text("Throttle");
                ImGui::ProgressBar(brakeReq, ImVec2(0.0f, 0.0f));
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                ImGui::Text("Brake");
                
                static timeStamp_t lastTS = getTimeStamp();
                timeStamp_t thisTS = getTimeStamp();
                double dt = double(thisTS - lastTS)/1000000.0;
                lastTS = thisTS;
                double act = pid.calculate(dt, speedReq - carCtrl.odom.speed.x());

                float preBrake = 0.40;
                if(speedReq < 0)
                    preBrake = 0.7;
                if(act < 0) {
                    brakeReq = preBrake - (act);
                    throttleReq = 0;
                } else {
                    throttleReq = act;
                    brakeReq = preBrake;
                }
            }

            ImGui::NewLine();
            if(ImGui::Checkbox("ODOM ENABLE", &odomActive)) {
                std::cout<<"Set ODOM state: "<<odomActive<<"\n";
                carCtrl.sendOdomEnable(odomActive);
            }
            ImGui::Text("Speed: %lf kmh", carCtrl.odom.speed.x()*3.6);
            ImGui::End();
       }
        
};

bool gRun = true;
void signal_handler(int signal){
    gRun = false;
    std::cerr<<"\nRequest closing..\n";
}


int main( int argc, char** argv){
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    tk::common::CmdParser cmd(argv, "Car Control");
    std::string soc_file = cmd.addArg("interface", "can0", "can interface to read");
    float smoothSteer    = cmd.addFloatOpt("-lerp_steer", 0.02, "steer lerp value");
    float smoothThrottle = cmd.addFloatOpt("-lerp_throttle", 0.10, "throttle lerp value");
    bool  queryEcus      = cmd.addBoolOpt("-query", "query all connected ECUS");
    cmd.parse();


    
    tkASSERT(canSoc.initSocket(soc_file));
    carCtrl.init(&canSoc);

    if(queryEcus) {
        carCtrl.sendOdomEnable(false);
        usleep(100000);
        carCtrl.requestMotorId();
        return 0;
    }
    carCtrl.sendOdomEnable(true);
    carCtrl.sendOdomEnable(true);
    carCtrl.sendOdomEnable(true);

    Control* c = new Control();
    viewer->add(c);
    viewer->start();
    
    /*
	Joystick joy;
    tkASSERT(joy.init());
    */

    tk::rt::Task t;
    t.init(50*1e3);
    float steer = 0, throttle = 0;
    bool active = false;
    while(gRun) {
        if(c->active) {
            carCtrl.setSteerAngle(c->steerReqDeg, c->steerSpeed);
            carCtrl.setBrakePos(c->brakeReq*15000);
            carCtrl.setAccPos(c->throttleReq*100);
        }
        t.wait();
    }
    

    viewer->join();
    carCtrl.close();
    canSoc.close();
    return 0;
}