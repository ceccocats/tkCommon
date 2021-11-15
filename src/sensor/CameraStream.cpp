#include "tkCommon/sensor/CameraStream.h"

using namespace tk::sensors;

bool 
CameraStream::loadConf(const int aIndex, const YAML::Node &aConf, const int aTriggerCamera) 
{ 
    this->mWidth = tk::common::YAMLgetConf<int>(aConf, "width", 640);
    this->mHeight = tk::common::YAMLgetConf<int>(aConf, "height", 480);
    this->mChannels = tk::common::YAMLgetConf<int>(aConf, "channels", 3);
    this->mFPS = tk::common::YAMLgetConf<int>(aConf, "fps", 30);
    this->mTriggerCamera = aTriggerCamera;
    this->mIndex = aIndex;
    if(!loadConfChild(aConf)) {
        tkERR("Error.");
        return false;
    }
    return true;
}

bool 
CameraStream::initRecorder(const std::string &aFile, const std::string &aOutputFormat) 
{ 
    tkWRN("Recorder not provided"); 
    return false; 
}

bool 
CameraStream::closeRecorder() 
{ 
    return false; 
}


bool 
CameraStream::writeFrame(tk::data::ImageData &aImage) 
{ 
    return false; 
}