#pragma once

#include <tkCommon/data/ImageData.h>
#include <thread>

#define MAX_STREAM_COUNT 32

namespace tk{namespace sensors{

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

		int width;
		int height;
		int channels;
		bool online = false;

		timeStamp_t timeStamp = 0;

		// To implement
		virtual bool init(int index, std::string file ) = 0;
		virtual bool initRecorder( std::string file, std::string outputFormat ) { tkWRN("Recorder not provided"); return false; }
		virtual bool closeRecorder() { return false; }
		virtual bool close() = 0;

		virtual bool readFrame( tk::data::ImageData &image ) = 0;

		virtual bool writeFrame( tk::data::ImageData &image ) { return false; };

	//protected:
//
    //    std::list<tk::data::ImageData*> rec_pool;
	};

}}