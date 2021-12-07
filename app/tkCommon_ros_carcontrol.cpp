#include <tkCommon/ros/RosWrapper.h>
#include <tkCommon/common.h>
#include <tkCommon/communication/car/CarControlInterface.h>

tk::gui::Viewer* viewer;
tk::communication::CanInterface canSoc;
tk::communication::CarControl   carCtrl;
std::mutex mtx;
tk::data::ActuationData act_data;

void
#if TKROS_VERSION == 1
actuation_cb(const ackermann_msgs::AckermannDriveStamped::ConstPtr &msg);
#elif TKROS_VERSION == 2
actuation_cb(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg);
#endif

int main(int argc, char** argv) {
    tk::common::CmdParser cmd(argv, "tkCommon_ros_carcontrol");
    std::string soc_file = cmd.addArg("interface", "can0", "can interface to read");
    std::string act_topic = cmd.addArg("topic", "/act", "");
    bool no_gui = cmd.addBoolOpt("-nogui", "Disable gui");
    cmd.parse();

    tkASSERT(canSoc.initSocket(soc_file));
    carCtrl.init(&canSoc);
    carCtrl.sendOdomEnable(true);
    //carCtrl.setSteerParams(0,0,0);

    tk::communication::CarControlInterface carInter;
    carInter.init(&carCtrl);
    if (no_gui) {
        carCtrl.resetSteerMotor();
        carInter.setActive(true);
    }

    tk::ros_wrapper::RosWrapper::instance.init();
#if TKROS_VERSION == 1
        ros::Subscriber act_sub = tk::ros_wrapper::RosWrapper::instance.n->subscribe<ackermann_msgs::AckermannDriveStamped>(act_topic, 1, actuation_cb);
#elif TKROS_VERSION == 2
        rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr act_sub = tk::ros_wrapper::RosWrapper::instance.n->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(act_topic, 1, actuation_cb);
#endif

    if (!no_gui) {
        viewer = tk::gui::Viewer::getInstance();
        viewer->start();
        viewer->add(&carInter);
    }


    tk::rt::Task t;
    t.init(10000);
    while (true) {
        // stop condition
        if (!no_gui && !viewer->isRunning())
            break;
        else if (!ros::ok())
            break;

        // check time
        if (((getTimeStamp() - act_data.header.stamp)/1e6) > 0.5f) {
            carInter.setInput(0.0f, -0.2f);
        } else {// send data
            mtx.lock();
            carInter.setInput(act_data);
            mtx.unlock();
        }
        t.wait();
    }

    if (!no_gui)
        viewer->join();

    tk::ros_wrapper::RosWrapper::instance.close();
    carInter.close();
    carCtrl.sendOdomEnable(false);
    carCtrl.close();
    canSoc.close();

    return 0;
}


void
#if TKROS_VERSION == 1
actuation_cb(const ackermann_msgs::AckermannDriveStamped::ConstPtr &msg)
#elif TKROS_VERSION == 2
actuation_cb(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg)
#endif
{
    mtx.lock();
    act_data.fromRos(*msg);
    mtx.unlock();
}