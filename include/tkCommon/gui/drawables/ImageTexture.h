#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/gui/shader/texture.h"

namespace tk{ namespace gui{

	class ImageTexture : public DataDrawable {

        private:
            tk::gui::TextureGeneric* texture = nullptr;
            int textureType = -1;

            tk::gui::Buffer<float>* pos = nullptr;
            float vertices[20];
            unsigned int indices[6];


        public:
            ImageTexture(std::string name, tk::data::SensorData* img = nullptr);
            ~ImageTexture();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
            void setPos(std::vector<tk::common::Vector3<float>> fourVertices);
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}