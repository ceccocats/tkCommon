#pragma once
#include <iostream>
#include "tkCommon/data/GpsImuData.h"
#include "tkCommon/gui/drawables/Drawables.h"

namespace tk{ namespace gui{

    class WrapperGeneric{
        public:
            virtual void init() = 0;
            virtual void updateRef(const tk::data::SensorData* data) = 0;
    };

    template<class T>
	class Wrapper : public WrapperGeneric{
        private:
            tk::gui::Drawable* drawable;
	    public:
            void init();
            void updateRef(const tk::data::SensorData* data);
	};

//void 
//updateRef(const tk::data::SensorData* data){
//    drawable->updateRef((T*)data);
//}

}}


