#pragma once

#if DW_VERSION_MAJOR >= 2
    #include <dw/core/Context.h>
    #include <dw/core/VersionCurrent.h>
//    #include <dw/core/NvMedia.h>
    #include <dw/sensors/Sensors.h>

    #define float16_t dwF16
#endif

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>

// macro to easily check for dw errors
#define CHECK_DW_ERROR(x)                                                                                                                                                                                \
    {                                                                                                                                                                                                    \
        dwStatus result = x;                                                                                                                                                                             \
        if (result != DW_SUCCESS)                                                                                                                                                                        \
            throw std::runtime_error(std::string("DW Error ") + dwGetStatusName(result) + std::string(" executing DW function:\n " #x) + std::string("\n at " __FILE__ ":") + std::to_string(__LINE__)); \
    }

#define CHECK_DW_ERROR_MSG(x, description)                                                                                                                                                                                                   \
    {                                                                                                                                                                                                                                        \
        dwStatus result = x;                                                                                                                                                                                                                 \
        if (result != DW_SUCCESS)                                                                                                                                                                                                            \
            throw std::runtime_error(std::string("DW Error ") + dwGetStatusName(result) + std::string(" executing DW function:\n " #x) + std::string("\n at " __FILE__ ":") + std::to_string(__LINE__) + std::string(" -> ") + description); \
    };

#define C_DW(x) CHECK_DW_ERROR(x)

namespace tk
{
namespace sensors
{
class Driveworks
{
private:
    bool initialized = false;

    int deviceGPU = 0;

public:

    static std::vector<std::string> dontCare_warning;
    
    dwContextHandle_t sdk = NULL;

    dwSALHandle_t sal = NULL;

    Driveworks();

    ~Driveworks();

    void init() {}

    void setGPUDevice(int device);

    int getGPUDevice() { return deviceGPU; }

    static void dontCare(std::string s){ dontCare_warning.push_back(s); };

    void release();
};

typedef Driveworks* DriveworksHandle_t;

} // namespace sensors
} // namespace tk

#include <dw/core/Logger.h>
#include <tkCommon/common.h>

dwLogCallback getConsoleLoggerCallback()
{
    return [](dwContextHandle_t, dwLoggerVerbosity, const char *msg) { 

        bool flag = false;
        for(int i = 0; i < tk::sensors::Driveworks::dontCare_warning.size(); i++){
            std::size_t found = std::string(msg).find(tk::sensors::Driveworks::dontCare_warning[i]);
            if (found!=std::string::npos)
                flag = true;
        }
        
        if(!flag && std::string(msg).length()!=22)
            tkWRN("Driveworks"<< std::string(msg)<<"\n");

    };
}

std::vector<std::string> tk::sensors::Driveworks::dontCare_warning;

tk::sensors::Driveworks::Driveworks()
{
    C_DW(dwLogger_initialize(getConsoleLoggerCallback()));
    C_DW(dwLogger_setLogLevel(DW_LOG_VERBOSE));

    // DriveWorks context initialization
    dwContextParameters sdkParams = {};
    C_DW(dwInitialize(&sdk, DW_VERSION, &sdkParams));

    // Sensors Abstraction Layer initialization
    C_DW(dwSAL_initialize(&sal, sdk));

    // Set variable
    initialized = true;
}

tk::sensors::Driveworks::~Driveworks() 
{
    // Check if initialized and release
    if(initialized){
        release();
    }
}

void tk::sensors::Driveworks::setGPUDevice(int device)
{
    // GPU device selection
    dwContext_selectGPUDevice(device, sdk);
}

void tk::sensors::Driveworks::release()
{
    // Sensors Abstraction layer release
    dwSAL_release(sal);

    // DriveWorks context release
    dwRelease(sdk);

    // Unset variable
    initialized = false;
}