#pragma once
#include "tkCommon/data/ImageData.h"

namespace tk{namespace data{

        struct CameraData_t{

            ImageData_t< uint8_t >*  data = nullptr;
            unsigned int count = 0;

            bool init(unsigned int n){
                if( n > 0 ){
                    count = n;
                    data = new ImageData_t<uint8_t >[count];
                    return true;
                }
                return false;
            }

            bool init(unsigned int n, int w, int h, int ch){
                if( n > 0 ){
                    count = n;
                    data = new ImageData_t<uint8_t >[count];
                    for(unsigned int i = 0; i < count; i++){
                        data[i].init(w, h, ch);
                    }
                    return true;
                }
                return false;
            }

            CameraData_t& operator=(const CameraData_t& s){
                if(s.count != count){
                    release();
                    init(s.count);
                }
                for(unsigned int i = 0; i < count; i++){
                    data[i] = s.data[i];
                }

                return *this;
            }

            void copyData(const CameraData_t& s){
                if(s.count != count){
                    release();
                    init(s.count);
                }
                for(unsigned int i = 0; i < count; i++){
                    data[i].copyData(s.data[i]);
                }
            }

            void release(){
                if(data == nullptr){
                    return;
                }
                ImageData_t<uint8_t>* tmp = data;
                data = nullptr;
                count = 0;
                delete [] tmp;
            }

            ~CameraData_t(){
                delete [] data;
            }
        };
    }}