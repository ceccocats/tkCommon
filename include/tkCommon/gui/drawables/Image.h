#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/ImageData.h"
#include <cstdarg>

namespace tk{ namespace gui{

	class Image : public Drawable {

        private:
            std::vector<tk::gui::Texture<uint8_t>*>  textures; 
            std::vector<tk::data::ImageData*>       images;
            std::vector<bool>   updates;
            std::vector<bool>   ready;
            std::vector<uint32_t> counter;


            std::string name;

        public:
            Image(int n, std::string name){
                this->textures.resize(n);
                this->images.resize(n);
                this->updates.resize(n);
                this->ready.resize(n);
                this->counter.resize(n);
                for(int i = 0; i < n; i++){
                    this->updates[i] = false;
                    this->ready[i]   = false;
                    this->counter[i] = 0; 
                }   
                this->name = name;
            }

            Image(std::string name, int n, ...){
                va_list arguments; 

                va_start(arguments, n);   
                this->textures.resize(n);
                this->images.resize(n);
                this->updates.resize(n);
                this->ready.resize(n);
                this->counter.resize(n);

                for(int i = 0; i < n; i++){
                    this->images[i]  = va_arg(arguments, tk::data::ImageData*);
                    this->updates[i] = false;
                    this->ready[i]   = true;
                    this->counter[i] = 0;
                }
                va_end(arguments);
                this->name = name;
            }

            ~Image(){

            }

            void onInit(tk::gui::Viewer *viewer){
                for(int i = 0; i < images.size(); i++){
                    if(this->ready[i] == true){
                        textures[i] = new tk::gui::Texture<uint8_t>();
                        textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
                        textures[i]->setData(images[i]->data);
                    }

                }
            }

            void updateRef(int n, tk::data::ImageData* image){
                tkASSERT(n < images.size());
                this->images[n] = image;
                this->updates[n] = true;
            }

            void draw(tk::gui::Viewer *viewer){

                //Check data
                for(int i = 0; i< images.size(); i++){
                    if(images[i]->isChanged(counter[i]) || updates[i]){
                        //Need to init?
                        if(this->ready[i] == false){
                            this->ready[i]  = true;
                            textures[i]     = new tk::gui::Texture<uint8_t>();
                            textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
                        }
                        //Copy data
                        images[i]->lockRead();
                        textures[i]->setData(images[i]->data);
                        images[i]->unlockRead();
                    }

                    ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
                    if(this->ready[i] == true){
                        
                        float imgX = ImGui::GetWindowSize().x-20;
                        //int imgY = ImGui::GetWindowSize().y-35;
                        //float imgX = textures[i]->width;
                        float imgY = imgX / ((float)textures[i]->width / textures[i]->height);
                        ImGui::Text("%s",images[i]->header.name.c_str());
                        ImGui::Image((void*)(intptr_t)textures[i]->id(), ImVec2(imgX, imgY));
                        ImGui::Separator();
                    }
                    ImGui::End();
                }
            }

            void imGuiInfos(){
                for(int i = 0; i< images.size(); i++){
                    std::stringstream print;
                    if(ready[i] == true){
                        print<<(*images[i]);
                        //ImGui::Text("%s",images[i]->header.name.c_str());
                        ImGui::Text("%s\n\n",print.str().c_str());
                    }
                    print.clear();
                }
            }

            void onClose(){
                for(int i = 0; i< images.size(); i++)
                    textures[i]->release();
            }

            std::string toString(){
                return name;
            }
	};
}}