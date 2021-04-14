#include "tkCommon/CmdParser.h"
#include "tkCommon/communication/CanInterface.h"
#include "tkCommon/communication/can/VehicleCanParser.h"
#include "tkCommon/joystick/Joystick.h"
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/math/quantile.h"
#include "tkCommon/communication/car/CarControl.h"

class MyViewer : public tk::gui::Viewer {
    public:
    MyViewer() {}
    ~MyViewer() {}

    void init() {
        tk::gui::Viewer::init();
    }

    void draw() {
        //tk::gui::Viewer::draw();
        tkViewport2D(width, height);

        drawBar("steer", steer, 0.2);
        drawBar("throttle", throttle, -0.2);

        std::string str = std::string("Back:  disable Acc\n")   +
                                      "Start: enable Acc\n"     + 
                                      "A:     set steer zero\n" +
                                      "B:     reset steer\n"    +
                                      "X:     disable steer\n";
        tkDrawText(str, {-xLim+0.05f, yLim-0.1f, 0.01f}, {0.0f,0.0f,0.0f}, {0.05f, 0.05f, 0.05f});


        std::string str2 = std::string("Steer Req:   ") + std::to_string(steerReq) + "\n" +
                           std::string("Steer Resp:  ") + std::to_string(curSteer) + "\n" +
                           std::string("Acc Req:   ") + std::to_string(accReq) + "\n" +
                           std::string("Acc Resp:  ") + std::to_string(curAcc) + "\n" +
                           std::string("Brk Req:   ") + std::to_string(brakeReq) + "\n" +
                           std::string("Brk Resp:  ") + std::to_string(curBrake) + "\n";
        tkDrawText(str2, {-xLim+0.05f, -0.4, 0.01f}, {0.0f,0.0f,0.0f}, {0.05f, 0.05f, 0.05f});
    }

    void drawBar(std::string name, float val, float y, float w = 0.8, float h = 0.1) {
        char buf[256];
        tkSetColor(tk::gui::color::WHITE);
        sprintf(buf, "%s: %+4.3f", name.c_str(), val);
        tkDrawText(buf, {-w,y+h,0}, {0,0,0}, {h,h,h});
        tkDrawRectangle({0,y,0}, {w*2, h, 0}, false);
        tkSetColor(tk::gui::color::RED);
        tkDrawRectangle({val*w, y, 0}, {h,h,h}, true);
    } 

    float steer = 0, throttle = 0;

    int steerReq = 0, curSteer = 0;
    int accReq = 0, brakeReq = 0, curAcc = 0, curBrake = 0;
};


bool gRun = true;
void signal_handler(int signal){
    gRun = false;
    std::cerr<<"\nRequest closing..\n";
}


int main( int argc, char** argv){
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // useful
    std::cout<<"Please set can before\n";
    std::cout<<"sudo ip link set can0 down\n";
    std::cout<<"sudo ip link set can0 type can bitrate 125000\n";
    std::cout<<"sudo ip link set can0 up\n\n";

    tk::common::CmdParser cmd(argv, "Can utils");
    std::string soc_file = cmd.addArg("interface", "can0", "can interface to read");
    std::string dbc_file = cmd.addArg("dbc", "", "DBC can to parse");
    float smoothSteer    = cmd.addFloatOpt("-lerp_steer", 0.02, "steer lerp value");
    float smoothThrottle = cmd.addFloatOpt("-lerp_throttle", 0.10, "throttle lerp value");
    cmd.parse();

    MyViewer viewer;
    viewer.setWindowName("Car control");
    viewer.initOnThread(false);
    viewer.plotManger->addCirclePlot("odom", tk::gui::color::RED, 1000000, 0.5);
    viewer.plotManger->set2d("odom",true);
    

	Joystick joy;
    tkASSERT(joy.init());

    tk::communication::CanInterface canSoc;
    tk::communication::CarControl carCtrl;
    
    tkASSERT(canSoc.initSocket(soc_file));
    carCtrl.init(&canSoc);
    carCtrl.sendOdomEnable(false);
    
    LoopRate rate(50*1e3);
    float steer = 0, throttle = 0;
    bool active = false;
    while(gRun) {
        // grab joystick
		tk::common::Vector2<float> stickL;
        float triggerL, triggerR;
        joy.update();
		joy.getStickLeft(stickL.x,stickL.y);
		joy.getTriggerLeft(triggerL);
		joy.getTriggerRight(triggerR);

        // commands
        steer    = tk::math::lerp(steer, stickL.x, smoothSteer);
        throttle = /*-triggerL + triggerR;*/ tk::math::lerp(throttle, -triggerL + triggerR, smoothThrottle);

        // update viewer
        viewer.steer = steer;
        viewer.throttle = throttle;

        std::cout<<"STATUS: "<<carCtrl.steerPos<<" "<<carCtrl.accPos<<" "<<carCtrl.brakePos<<"\n";
        viewer.curSteer = carCtrl.steerPos;
        viewer.curAcc = carCtrl.accPos;
        viewer.curBrake = carCtrl.brakePos;
        
        tk::data::VehicleData::odom_t o = carCtrl.odom;
        viewer.plotManger->addPoint("odom", tk::common::odom2tf(o.x, o.y, 0));
        std::cout<<"ODOM: "<<o.x<<" "<<o.y<<" "<<o.yaw<<" "<<o.speed<<"\n";
        
        // ODOM LOG
        {
            static std::ofstream odom_os("odomlog_"+getTimeStampString()+".txt");
            odom_os<<o.t<<" "<<o.x<<" "<<o.y<<" "<<o.yaw<<" "<<o.speed<<"\n";
            odom_os.flush();
        }

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
            carCtrl.setBrakePos(8000);
            sleep(1);
            carCtrl.sendEngineStart();
            sleep(1);
            carCtrl.sendAccEnable(false);
        }

        if(active) {
            int32_t steerReq = (-steer) * 18000;
            carCtrl.setSteerPos(steerReq);


            uint16_t accReq   = 0; 
            uint16_t brakeReq = 0;             

            if(fabs(throttle) > 0.05) {
                accReq   = throttle > 0 ? throttle*100 : 0;
                brakeReq = throttle < 0 ? -throttle*15000 : 0;
            }

            std::cout<<"Req: "<<steerReq<<" "<<accReq<<" "<<brakeReq<<"\n";
            viewer.steerReq = steerReq;
            viewer.accReq = accReq;
            viewer.brakeReq = brakeReq;

            carCtrl.setAccPos(accReq); 
            carCtrl.setBrakePos(brakeReq);
        }
        rate.wait(false);
    }
    
    viewer.joinThread();
    carCtrl.close();
    canSoc.close();
    return 0;
}