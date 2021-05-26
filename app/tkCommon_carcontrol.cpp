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
        Control() {}
        ~Control() {}

        bool active = false;
        bool setZero = false;
        float steerReqDeg = 0;
        float steerSpeed = 65536; 

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
            ImGui::End();


            /*
            ImGui::Begin("Car control", NULL, ImGuiWindowFlags_NoScrollbar);
            ImGui::Text("Steer: %f",steer);
            ImGui::Text("Throttle: %f",throttle);
            ImGui::Text("\n");
            ImGui::Text("steerReq: %f",steerReq);
            ImGui::Text("accReq: %f",accReq);
            ImGui::Text("brakeReq: %f",brakeReq);
            ImGui::Text("\n");
            ImGui::Text("enableSteer: %d (start/back)",curSteer);
            ImGui::Text("enableAcc: %d",curAcc);
            ImGui::Text("enableBrake: %d",curBrake);
            ImGui::Text("\n");
            ImGui::Text("A: steer zero");
            ImGui::Text("B: motor reset");
            ImGui::Text("X: off");
            ImGui::Text("Y: start");
            ImGui::End();

            ImGui::Begin("PIDs", NULL, ImGuiWindowFlags_NoScrollbar);
            ImGui::Text("Velocity %f", curVel);
            ImGui::SliderFloat("Request Velocity",&vel,-1.0f,30.0f,"%.1f km/h");
            ImGui::Separator();
            ImGui::Text("PID Request %f", torque);
            ImGui::SliderFloat("kp",&kpTorque,-3.0f,3.0f,"%.2f");
            ImGui::SliderFloat("ki",&kiTorque,-1.0f,1.0f,"%.4f");
            ImGui::SliderFloat("kd",&kdTorque,-1.0f,1.0f,"%.4f");
            ImGui::Separator();
            ImGui::Text("PID Request %f", brake);
            ImGui::SliderFloat("kp",&kpBrake,-3.0f,3.0f,"%.2f");
            ImGui::SliderFloat("ki",&kiBrake,-1.0f,1.0f,"%.4f");
            ImGui::SliderFloat("kd",&kdBrake,-1.0f,1.0f,"%.4f");
            ImGui::Text("Angle");
            ImGui::SliderFloat("steer",&angle,-450.0,450.0,"%.4f");

            std::string text;
            if(joystick){
                text = "use velocity";
            }else{
                text = "use joystick";
            }
            if (ImGui::Button(text.c_str())){
                joystick = !joystick;
            }
            ImGui::End();
            */
        }

        /*
        int curSteer,curAcc,curBrake;
        float steerReq,accReq,brakeReq,steer,throttle;
        
        float vel = 0.0f;
        float curVel = 0.0f;

        bool joystick = true;

        float kpTorque = 0.04f;
        float kiTorque = 0.0f;
        float kdTorque = 0.0f;

        float kpBrake = 0.04f;
        float kiBrake = 0.0f;
        float kdBrake = 0.0f;

        float torque,brake;

        float angle = 0;
        */
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
    //carCtrl.sendOdomEnable(false);

    if(queryEcus) {
        usleep(100000);
        carCtrl.requestMotorId();
        return 0;
    }


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
        /*
        // grab joystick
		tk::common::Vector2<float> stickL;
        float triggerL, triggerR;
        joy.update();
		joy.getStickLeft(stickL.x(),stickL.y());
		joy.getTriggerLeft(triggerL);
		joy.getTriggerRight(triggerR);

        // commands
        steer    = tk::math::lerp(steer, stickL.x(), smoothSteer);
        throttle = tk::math::lerp(throttle, -triggerL + triggerR, smoothThrottle);

        // update viewer
        c->steer    = steer;
        c->throttle = throttle;
        c->curSteer = carCtrl.steerPos;
        c->curAcc   = carCtrl.accPos;
        c->curBrake = carCtrl.brakePos;

        if(joy.getButtonPressed(BUTTON_BACK)) {
            std::cout<<"disable Acc\n";
            carCtrl.enable(false);
            active = false;
        } else if(joy.getButtonPressed(BUTTON_START)) {
            std::cout<<"enable Acc\n";
            carCtrl.enable(true);
            active = true;
        } else if(joy.getButtonPressed(BUTTON_A)) {
            std::cout<<"Set Steer Zero\n";
            carCtrl.setSteerZero();
        } else if(joy.getButtonPressed(BUTTON_B)) {
            std::cout<<"Reset motor\n";
            carCtrl.resetSteerMotor();
        } else if(joy.getButtonPressed(BUTTON_X)) {
            std::string cmd = "OFF";
            std::cout<<"Command: "<<cmd<<"\n";
            carCtrl.sendGenericCmd(cmd);
        } else if(joy.getButtonPressed(BUTTON_Y)) {
            std::cout<<"!!!! Engine !!!!\n";
            carCtrl.enable(true);
            usleep(1000);
            carCtrl.setBrakePos(2000);
            sleep(1);
            carCtrl.sendEngineStart();
            sleep(1);
            carCtrl.enable(false);
        }

        if(active) {

            if(c->joystick){
                carCtrl.usePid = false;

                int32_t steerReq = (-steer) * 18000;
                carCtrl.setSteerPos(steerReq);

                uint16_t accReq   = 0; 
                uint16_t brakeReq = 0;             

                if(fabs(throttle) > 0.05) {
                    accReq   = throttle > 0 ? throttle*100 : 0;
                    brakeReq = throttle < 0 ? -throttle*15000 : 0;
                }

                c->steerReq = steerReq;
                c->accReq   = accReq;
                c->brakeReq = brakeReq;

                carCtrl.setVel(-10);
                carCtrl.setAccPos(accReq); 
                carCtrl.setBrakePos(brakeReq);
            }else{
                carCtrl.usePid = true;

                //Update kp ki kd
                carCtrl.pidTorque.setGain(c->kpTorque,c->kiTorque,c->kdTorque);
                carCtrl.pidBrake.setGain(c->kpBrake,c->kiBrake,c->kdBrake);

                carCtrl.setVel(c->vel);
                c->brake = carCtrl.brakeRequest;
                c->torque = carCtrl.torqueRequest;

                carCtrl.steerAngle(c->angle);
            }
            c->curVel = carCtrl.odom.speed.x();
        }
        */

        if(c->active) {
            carCtrl.steerAngle(c->steerReqDeg, c->steerSpeed);
        }

        t.wait();
    }
    

    viewer->join();
    carCtrl.close();
    canSoc.close();
    return 0;
}