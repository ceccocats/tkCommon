#pragma once

#include <tkCommon/data/ImageData.h>

#define MAX_STREAM_COUNT 32

namespace tk { namespace sensors {

	class CameraContext
	{
		public:
			CameraContext() {}
			virtual bool init (std::string mask = "", int count = 0) { return true;};
			virtual bool start(std::string mask = "", int count = 0) { return true;};
			virtual bool stop() { return true;};
			virtual bool getSelsectorMask(std::string *mask, std::vector<std::string> devices) { return true;}
	};

	class CameraStream {

	public:

		int mWidth;
		int mHeight;
		int mChannels;
		int mFPS;
		int mIndex;
		int mTriggerCamera;
		bool online = false;

		timeStamp_t timeStamp = 0;

		// To implement
		virtual bool init(const std::string &aFile) = 0;
		virtual bool loadConf(const int aIndex, const YAML::Node &aConf, const int aTriggerCamera = 0);
		virtual bool loadConfChild(const YAML::Node &aConf) = 0;
		virtual bool initRecorder(const std::string &aFile, const std::string &aOutputFormat);
		virtual bool closeRecorder();
		virtual bool close() = 0;
		virtual bool readFrame(tk::data::ImageData &aImage) = 0;
		virtual bool writeFrame(tk::data::ImageData &aImage);
	};

}}