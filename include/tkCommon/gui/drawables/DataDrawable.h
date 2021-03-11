#pragma once
#include "tkCommon/gui/drawables/Drawable.h"


namespace tk{ namespace gui{

    template <class T>
	class DataDrawable : public Drawable {
	public:

        void draw(tk::gui::Viewer *viewer) final;

        virtual void updateRef(T* data) final;
        virtual bool isAsyncedCopied(int idx = 0) final;

	protected:

        virtual void drawData(tk::gui::Viewer *viewer) = 0; 
        virtual void updateData(tk::gui::Viewer *viewer) = 0; 
        virtual void forceUpdate() final;   

        T* data = nullptr;
        uint32_t counter = 0;

        std::stringstream print;

    private:

        std::mutex ref_mutex;
        bool drw_has_reference = true;
        bool new_ref_data      = false;
	};
}}


template <class T> 
void 
tk::gui::DataDrawable<T>::updateRef(T* data){
    this->ref_mutex.lock();
    this->data              = data;
    this->new_ref_data      = true;
    this->drw_has_reference = false;
    this->ref_mutex.unlock();
}

template <class T> 
void 
tk::gui::DataDrawable<T>::draw(tk::gui::Viewer *viewer) {
    if(this->data != nullptr){
        if(this->drw_has_reference){
            this->ref_mutex.lock();
            if(this->new_ref_data){
                this->updateData(viewer);
                this->new_ref_data = false;
            }
            this->ref_mutex.unlock();
        }else{
            if(this->data->tryLockRead()){
                if(this->data->isChanged(counter)){
                    this->updateData(viewer);
                }
                this->data->unlockRead();
            }
        }
        this->drawData(viewer);
    }
}

template <class T> 
void 
tk::gui::DataDrawable<T>::forceUpdate(){
    counter -= 1;
}

template <class T> 
bool 
tk::gui::DataDrawable<T>::isAsyncedCopied(int idx){
    return !this->new_ref_data;
}