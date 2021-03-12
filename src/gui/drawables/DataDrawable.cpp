#include "tkCommon/gui/drawables/DataDrawable.h"

void 
tk::gui::DataDrawable::updateRef(tk::data::SensorData* data){
    ref_mutex.lock();
    this->data = data;
    new_ref_data      = true;
    drw_has_reference = false;
    ref_mutex.unlock();
}

void 
tk::gui::DataDrawable::draw(tk::gui::Viewer *viewer) {
    if(this->data != nullptr){
        if(this->drw_has_reference){
            if(this->data->tryLockRead()){
                if(this->data->isChanged(counter)){
                    this->updateData(viewer);
                }
                this->data->unlockRead();
            }
        }else{
            if(this->ref_mutex.try_lock()){
                if(this->new_ref_data){
                    this->updateData(viewer);
                    this->new_ref_data = false;
                }
                this->ref_mutex.unlock();
                this->data->unlockRead();
            }
        }
    }
    this->drawData(viewer);
}

void 
tk::gui::DataDrawable::forceUpdate(){
    counter -= 1;
}

bool 
tk::gui::DataDrawable::synched(){
    return !this->new_ref_data;
}