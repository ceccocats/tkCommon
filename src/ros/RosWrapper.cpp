#include "tkCommon/ros/RosWrapper.h"

namespace tk { namespace ros_wrapper {
    RosWrapper RosWrapper::instance;

    bool 
    RosWrapper::init() 
    {
    if(initted)
        return true;
    initted = true;
    
    tkDBG("ROS init\n");
    int argc = 2;
    char* argv[] = { strdup("tkRos"), strdup("lol") };
#if TKROS_VERSION == 1
    ros::init(argc, argv, "tkRos", ros::init_options::NoSigintHandler);
    
    n = new ros::NodeHandle();
#elif TKROS_VERSION == 2
    rclcpp::InitOptions init_options = rclcpp::InitOptions();
    init_options.shutdown_on_sigint = false;
    rclcpp::init(argc, argv, init_options);

    n = std::make_shared<rclcpp::Node>("Ros");
#endif
    // spin thread start
    pthread_create(&rosSpinThread, NULL, spinThread, NULL);
    return true;
}

    void RosWrapper::close() 
    {
        tkDBG("ROS close\n");
        if (initted) {
    #if TKROS_VERSION == 1
            ros::shutdown();
            delete n;
    #elif TKROS_VERSION == 2
            rclcpp::shutdown();
            delete n.get();
    #endif
            initted = false;
        }
    }

    void 
    *RosWrapper::spinThread(void*) {
    #if TKROS_VERSION == 1
        ros::spin();
    #elif TKROS_VERSION == 2
        rclcpp::spin(RosWrapper::instance.n);
    #endif
        pthread_exit(NULL);
    }
}}
