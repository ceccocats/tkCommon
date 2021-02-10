#pragma once

#include "tkCommon/gui/drawables/Drawable.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>


namespace tk { namespace gui {

    class RecordInfo : public Drawable {
    public:
        RecordInfo();
        RecordInfo(const std::string &logPath);
        ~RecordInfo();

        void onInit(tk::gui::Viewer *viewer);
        void draw(tk::gui::Viewer *viewer);
        void imGuiSettings();
        void imGuiInfos();
        void onClose();
        
        std::string toString();
    private:
        std::string path;
        std::string info;
        float       freeSpace;
        std::vector< std::pair <float,std::string> > fileList;

        int tick;
    };
}}