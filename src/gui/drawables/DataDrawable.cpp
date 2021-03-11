#include "tkCommon/gui/drawables/DataDrawable.h"

tk::gui::DataDrawable::~DataDrawable(){
    for(int i = 0; i< ref_mutex.size(); i++){
        delete ref_mutex[i];
    }
}

void
tk::gui::DataDrawable::init(int n){
    data.resize(n);
    counter.resize(n);
    ref_mutex.resize(n);
    new_ref_data.resize(n);

    for(int i = 0; i < data.size(); i++){
        data[i] = nullptr;
        counter[i] = 0;
        ref_mutex[i] = new std::mutex();
        new_ref_data[i] = false;
    }   
}


void 
tk::gui::DataDrawable::updateRef(tk::data::SensorData* data){
    int i = data->header.sensorID;

    tkASSERT(i < this->data.size());

    this->ref_mutex[i]->lock();
    this->data[i]              = data;
    this->new_ref_data[i]      = true;
    drw_has_reference          = false;
    this->ref_mutex[i]->unlock();
}

void 
tk::gui::DataDrawable::draw(tk::gui::Viewer *viewer) {
    for(int i = 0; i < data.size(); i++){
        if(this->data[i] != nullptr){
            if(!this->drw_has_reference){
                if(this->ref_mutex[i]->try_lock()){
                    if(this->new_ref_data[i]){
                        this->updateData(i,viewer);
                        this->new_ref_data[i] = false;
                    }
                    this->ref_mutex[i]->unlock();
                    this->data[i]->unlockRead();
                }
            }else{
                if(this->data[i]->tryLockRead()){
                    if(this->data[i]->isChanged(counter[i])){
                        this->updateData(i,viewer);
                    }
                    this->data[i]->unlockRead();
                }
            }
        }
        this->drawData(viewer);
    }
}

void 
tk::gui::DataDrawable::forceUpdate(){
    for(int i = 0; i < data.size(); i++)
        counter[i] -= 1;
}

bool 
tk::gui::DataDrawable::isAsyncedCopied(int idx){
    return !this->new_ref_data[idx];
}