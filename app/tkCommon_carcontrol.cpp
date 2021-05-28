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

        float brakeReq = 0;
        float throttleReq = 0;
        float speedReqKMH = 0;
        float speedReq = -1;

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
                ImVec4 ca = {0.1f,0.9f,0.1f,1.0f};
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ca);
                ImGui::ProgressBar(carCtrl.getActThrottle(), ImVec2(0.0f, 0.0f));
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                ImGui::Text("Throttle");
                ca = {0.9f,0.1f,0.1f,1.0f};
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ca);
                ImGui::ProgressBar(carCtrl.getActBrake(), ImVec2(0.0f, 0.0f));
                ImGui::PopStyleColor();
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
                ImGui::Text("Brake");
                
                static timeStamp_t lastTS = getTimeStamp();
                timeStamp_t thisTS = getTimeStamp();
                double dt = double(thisTS - lastTS)/1000000.0;
                lastTS = thisTS;
                double act = pid.calculate(dt, speedReq - carCtrl.odom.speed.x());

                float preBrake = 0.40;
                if(speedReq < 0 && carCtrl.odom.speed.x() < 1)
                    preBrake = 0.6;
                if(speedReq < 0 && carCtrl.odom.speed.x() <= 0.01)
                    preBrake = 0.7;
                if(act < 0) {
                    brakeReq = preBrake - (act);
                    throttleReq = 0;
                } else {
                    throttleReq = act;
                    brakeReq = preBrake;
                }
            }
            carCtrl.setTargetSteer(steerReqDeg);
            carCtrl.setTargetBrake(brakeReq);
            carCtrl.setTargetThrottle(throttleReq);

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
    
    viewer->join();
    carCtrl.close();
    canSoc.close();
    return 0;
}