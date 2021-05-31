#include "tkCommon/communication/car/CarControl.h"
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/imgui/imgui_internal.h"

namespace tk { namespace communication {

class CarControlInterface : public tk::gui::Drawable{
    private:
        tk::communication::CarControl *carCtrl;
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

        bool manual = false;

    public:
        CarControlInterface(tk::communication::CarControl *carCtrl) {
            this->carCtrl = carCtrl;
            pid.init(0.2, 0, 0, -1.0, 1.0);
        }
        ~CarControlInterface() {}

        void setInput(float steerDeg, float speedKMH) {
            if(!manual) {
                steerReqDeg = steerDeg;
                speedReqKMH = speedKMH;
                speedReq = speedReqKMH/3.6;
            }
        }


        void draw(tk::gui::Viewer *viewer){
            ImGui::Begin("Car control", NULL, ImGuiWindowFlags_NoScrollbar);
            if(ImGui::Checkbox("ACTIVATE", &active)) {
                tkMSG("Set CAR state: "<<active);
                carCtrl->enable(active);
            }
            if(ImGui::Button("SET ZERO")) {
                tkMSG("Set Steer ZERO");
                carCtrl->setSteerZero();
            }
            if(ImGui::Button("RESET")) {
                tkMSG("Reset steer");
                carCtrl->resetSteerMotor();
            }
            if(ImGui::Checkbox("ODOM ENABLE", &odomActive)) {
                std::cout<<"Set ODOM state: "<<odomActive<<"\n";
                carCtrl->sendOdomEnable(odomActive);
            }

            // status
            ImGui::Text("Speed: %lf kmh", carCtrl->odom.speed.x()*3.6);
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            float steer_act = carCtrl->getActSteer();
            ImGui::SliderFloat("SteerAct", &steer_act, -30, +30);
            ImGui::PopItemFlag();
            ImGui::Text("Steer pos read: %d", carCtrl->steerPos);
            ImVec4 ca = {0.1f,0.9f,0.1f,1.0f};
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ca);
            ImGui::ProgressBar(carCtrl->getActThrottle(), ImVec2(0.0f, 0.0f));
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("ThrottleAct");
            ca = {0.9f,0.1f,0.1f,1.0f};
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ca);
            ImGui::ProgressBar(carCtrl->getActBrake(), ImVec2(0.0f, 0.0f));
            ImGui::PopStyleColor();
            ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
            ImGui::Text("BrakeAct");

            // input type
            ImGui::NewLine();
            

            // manual
            if(ImGui::Checkbox("Manual", &manual)) {
                // reset inputs
                steerReqDeg = 0;
                brakeReq = 0;
                throttleReq = 0;
                speedReqKMH = 0;
                speedReq = 0;
            }
            if(manual) {
                ImGui::SliderFloat("Steer", &steerReqDeg, -30, +30);
                ImGui::InputFloat("Steer_", &steerReqDeg, -30, +30);

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
                }

            }
            else {
                speedControl = true;
                ImGui::Text("Getting act from code");
            }
            ImGui::End();



            if(active) {
                if(speedControl) {
                    static timeStamp_t lastTS = getTimeStamp();
                    timeStamp_t thisTS = getTimeStamp();
                    double dt = double(thisTS - lastTS)/1000000.0;
                    lastTS = thisTS;
                    double act = pid.calculate(dt, speedReq - carCtrl->odom.speed.x());
                    float preBrake = 0.40;
                    if(speedReq < 0 && carCtrl->odom.speed.x() < 1)
                        preBrake = 0.6;
                    if(speedReq < 0 && carCtrl->odom.speed.x() <= 0.01)
                        preBrake = 0.7;
                    if(act < 0) {
                        brakeReq = preBrake - (act);
                        throttleReq = 0;
                    } else {
                        throttleReq = act;
                        brakeReq = preBrake;
                    }
                }
                carCtrl->setTargetSteer(steerReqDeg);
                carCtrl->setTargetBrake(brakeReq);
                carCtrl->setTargetThrottle(throttleReq);
            }

       }
        
};

}}