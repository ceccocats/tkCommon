#pragma once
#include <iostream>
#include "tkCommon/gui/Drawables/Drawables.h"

#include "tkCommon/data/GpsImuData.h"

namespace tk{ namespace gui{

    class WrapperGeneric{};

    template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
	class Wrapper<T> : public WrapperGeneric{
        private:
            tk::gui::Drawable* drawable;
	    public:
            void init();
            void updateRef(const tk::data::SensorData* data);
            void close();
	};

//void 
//updateRef(const tk::data::SensorData* data){
//    drawable->updateRef((T*)data);
//}

void 
close(){
    drawable->close();
    delete drawable;
}
}}


