#include "tkCommon/gui/drawables/DataDrawable.h"

using namespace tk::gui;

DataDrawable::DataDrawable()
{
    this->data                  = nullptr;
    this->counter               = 0;
    this->mDrwHasReference      = false;
    this->mDrwHasPool           = false;
    this->mFirstData            = true;
}

/*
void 
DataDrawable::updateRef(tk::data::SensorData* data)
{    
    this->mPointerMutex.lock();
    this->data              = data;
    this->mNewPointer       = true;
    this->mDrwHasReference  = false;
    this->mDrwHasPool       = false;
    this->mPointerMutex.unlock();
}
*/

void 
DataDrawable::setPool(tk::rt::DataPool *aPool) 
{
    this->mPool             = aPool;
    this->mPoolLastData     = 0;
    this->mDrwHasPool       = true;
    this->mDrwHasReference  = false;
}

void 
DataDrawable::draw(Viewer *viewer) 
{
    if (this->mDrwHasPool) {
        if(this->mPool->newData(this->mPoolLastData)) {
            int idx;

            // grab last inserted element
            this->data = (tk::data::SensorData *) this->mPool->get(idx);
            if (this->data != nullptr) {
                this->mPoolLastData = this->data->header.messageID;
                this->updateData(viewer);

                // release data
                this->mPool->releaseGet(idx);
            }
        }
    } else if (this->data != nullptr) {
        if (this->mDrwHasReference) {
            if(this->data->tryLockRead()) {
                if(mFirstData || this->data->isChanged(counter)){
                    mFirstData = false;
                    this->updateData(viewer);
                }
                this->data->unlockRead();
            }
        } 
        /*
        else {
            if (this->mNewPointer && this->mPointerMutex.try_lock()) {
                this->updateData(viewer);
                this->mNewPointer = false;
                this->mPointerMutex.unlock();
            }
        } 
        */
    }

    this->drawData(viewer);
}

void 
DataDrawable::forceUpdate() 
{
    counter -= 1;
}