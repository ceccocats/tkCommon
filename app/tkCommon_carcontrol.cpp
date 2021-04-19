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

class Control : public tk::gui::Drawable{
    public:
        Control() {}
        ~Control() {}

        void draw(tk::gui::Viewer *viewer){
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

            ImGui::Begin("PID", NULL, ImGuiWindowFlags_NoScrollbar);
            ImGui::SliderFloat("Velocity",&vel,0.0f,30.0f,"%.1f km/h",0.1f);
            ImGui::SliderFloat("kp",&kp,-3.0f,3.0f,"%.1f",0.1f);
            ImGui::SliderFloat("ki",&ki,-3.0f,3.0f,"%.1f",0.1f);
            ImGui::SliderFloat("kd",&kd,-3.0f,3.0f,"%.1f",0.1f);

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
        }

        float steer = 0;
        float throttle = 0;

        int curSteer = 0;
        int curAcc = 0;
        int curBrake = 0;

        float steerReq = 0;        
        float accReq     = 0;
        float brakeReq = 0;

        float kp  = 1.0f;
        float ki  = 1.0f;
        float kd  = 1.0f;
        float vel = 0.0f;

        bool joystick = true;
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
    cmd.parse();

    Control* c = new Control();
    viewer->add(c);
    viewer->start();
    
	Joystick joy;
    tkASSERT(joy.init());

    tk::communication::CanInterface canSoc;
    tk::communication::CarControl carCtrl;
    
    tkASSERT(canSoc.initSocket(soc_file));
    carCtrl.init(&canSoc);
    carCtrl.sendOdomEnable(false);
    
    tk::rt::Task t;
    t.init(50*1e3);
    float steer = 0, throttle = 0;
    bool active = false;
    while(gRun) {
        // grab joystick
		tk::common::Vector2<float> stickL;
        float triggerL, triggerR;
        joy.update();
		joy.getStickLeft(stickL.x(),stickL.y());
		joy.getTriggerLeft(triggerL);
		joy.getTriggerRight(triggerR);

        // commands
        steer    = tk::math::lerp(steer, stickL.x(), smoothSteer);
        throttle = /*-triggerL + triggerR;*/ tk::math::lerp(throttle, -triggerL + triggerR, smoothThrottle);

        // update viewer
        c->steer = steer;
        c->throttle = throttle;
        //std::cout<<"Steer: "<<steer<<"\tThrottle: "<<throttle<<std::endl;

        //std::cout<<"STATUS: "<<carCtrl.steerPos<<" "<<carCtrl.accPos<<" "<<carCtrl.brakePos<<"\n";
        c->curSteer = carCtrl.steerPos;
        c->curAcc = carCtrl.accPos;
        c->curBrake = carCtrl.brakePos;
        
        //tk::data::VehicleData::odom_t o = carCtrl.odom;
        //viewer.plotManger->addPoint("odom", tk::common::odom2tf(o.x, o.y, 0));
        //std::cout<<"ODOM: "<<o.x<<" "<<o.y<<" "<<o.yaw<<" "<<o.speed<<"\n";
        //
        //// ODOM LOG
        //{
        //    static std::ofstream odom_os("odomlog_"+getTimeStampString()+".txt");
        //    odom_os<<o.t<<" "<<o.x<<" "<<o.y<<" "<<o.yaw<<" "<<o.speed<<"\n";
        //    odom_os.flush();
        //}

        if(joy.getButtonPressed(BUTTON_BACK)) {
            std::cout<<"disable Acc\n";
            carCtrl.sendAccEnable(false);
            active = false;
        } else if(joy.getButtonPressed(BUTTON_START)) {
            std::cout<<"enable Acc\n";
            carCtrl.sendAccEnable(true);
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
            carCtrl.sendAccEnable(true);
            usleep(1000);
            carCtrl.setBrakePos(2000);
            sleep(1);
            carCtrl.sendEngineStart();
            sleep(1);
            carCtrl.sendAccEnable(false);
        }

        if(active) {

            if(c->joystick){
                int32_t steerReq = (-steer) * 18000;
                carCtrl.setSteerPos(steerReq);


                uint16_t accReq   = 0; 
                uint16_t brakeReq = 0;             

                if(fabs(throttle) > 0.05) {
                    accReq   = throttle > 0 ? throttle*100 : 0;
                    brakeReq = throttle < 0 ? -throttle*15000 : 0;
                }

                //std::cout<<"Req: "<<steerReq<<" "<<accReq<<" "<<brakeReq<<"\n";
                c->steerReq = steerReq;
                c->accReq = accReq;
                c->brakeReq = brakeReq;

                carCtrl.setAccPos(accReq); 
                carCtrl.setBrakePos(brakeReq);
            }else{
                carCtrl.pid.setGain(c->kp,c->ki,c->kd);
                carCtrl.setVel(c->vel);
            }
        }
        t.wait();
    }
    
    //viewer.joinThread();
    carCtrl.close();
    canSoc.close();
    viewer->join();
    return 0;
}