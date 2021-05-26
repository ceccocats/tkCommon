#include "tkCommon/gui/drawables/RecordInfo.h"

namespace tk { namespace gui {

    RecordInfo::RecordInfo()
    {
        this->path = "";
        this->info = "";
        this->tick = 0;
    }

    RecordInfo::RecordInfo(const std::string &logPath)
    {
        this->path = logPath;
        this->info = "";
        this->tick = 0;
    }
    
    RecordInfo::~RecordInfo()
    {

    }

    void 
    RecordInfo::onInit(tk::gui::Viewer *viewer)
    {

    }

    void 
    RecordInfo::draw(tk::gui::Viewer *viewer) 
    {
        tick++;
        if (tick >= 60 && path != "") {
            tick = 0;

            fileList.clear();

            // open directory
            DIR *d;
            struct stat file_stat;
            struct dirent *dir;
            struct statvfs dir_stat;
            d = opendir(path.c_str());
            if (d) {
                while (((dir = readdir(d)) != NULL)) {
                    std::string fileName = std::string(dir->d_name);

                    if (fileName != ".." && fileName != ".") {
                        std::pair<float, std::string> a;
                    
                        std::string filePath = path + "/" + fileName;
                        if (lstat(filePath.c_str(), &file_stat) == 0) {
                            a.first = (float(file_stat.st_size) / 1024.0f) / 1024.0f;
                        } 
                        a.second = fileName;

                        fileList.push_back(a);
                    }
                }

                // get freespace
                if (statvfs(path.c_str(), &dir_stat) == 0) {
                    freeSpace = (((float(dir_stat.f_bsize * dir_stat.f_bavail) / 1024.0f) / 1024.0f) / 1024.0f);
                }

                closedir(d);
            }

            // Using simple sort() function to sort
            sort(fileList.begin(), fileList.end());
            std::reverse(fileList.begin(), fileList.end());
        }
    }   

    void 
    RecordInfo::imGuiSettings()
    {

    }

    void 
    RecordInfo::imGuiInfos()
    {
        ImGui::Text("Save folder: %s", path.c_str());

        // print data
        for(int i = 0; i < fileList.size(); i++) {
            std::string format ="MB";
            float size = fileList[i].first;
            if (size >= 1024) {
                size /= 1024;
                format = "GB";
            }
            if (fileList[i].second.size() > 40) {

                fileList[i].second = fileList[i].second.substr(0, 36) + "...";
            }
            ImGui::Text("-> %-40s%s %.2f", fileList[i].second.c_str(), format.c_str(),  size);
        }

        ImVec4 color;
        if (freeSpace >= 100)
            color = ImVec4(0, 1, 0, 1);
        else if (freeSpace >= 50)
            color = ImVec4(1, 1, 0, 1);
        else
            color = ImVec4(1, 0, 0, 1);

        ImGui::TextColored(color, "Freespace: %.2f GB", freeSpace);
    }

    void 
    RecordInfo::onClose()
    {

    }
    
    std::string 
    RecordInfo::toString()
    {
        return "RecordInfo";
    }
}}