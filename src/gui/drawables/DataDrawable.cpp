#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/gui/utils/deprecated_primitives.h"
using namespace tk::gui;

DataDrawable::DataDrawable()
{
    this->data                  = nullptr;
    this->counter               = 0;
    this->mDrwHasReference      = true;
    this->mDrwHasPool           = false;
    this->mFirstData            = true;
    this->mFirstDraw            = true;
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
    this->mDataEnableTf     = true;
}

void 
DataDrawable::draw(Viewer *viewer) 
{   
    if (mFirstDraw) {
        mShaderText = tk::gui::shader::text::getInstance();
        mFirstDraw = false;
    }
    if (this->mDrwHasPool) {
        if(this->mPool->newData(this->mPoolLastData)) {
            this->mPoolLastData = this->mPool->inserted;
            
            int idx;
            // grab last inserted element
            this->data = (tk::data::SensorData *) this->mPool->get(idx);
            if (this->data != nullptr) {
                
                if (data->header.messageID >= this->mPool->inserted)
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
    }
    if(data != nullptr) {
        drwModelView = drwModelView * glm::make_mat4x4(data->header.tf.matrix().data());
        
        // draw sensor tf
        if(mDataEnableTf) {
            glPushMatrix();
            auto sensor_tf = tf*data->header.tf;
            tk::gui::setColor(tk::gui::color::WHITE);
            tk::gui::prim::drawLine(tk::common::tf2pose(tf), tk::common::tf2pose(sensor_tf), 1.0);
            glMultMatrixf(sensor_tf.matrix().data());
            tk::gui::prim::drawAxis(0.2);
            glPopMatrix();

            // draw sensor name
            mShaderText->draw(drwModelView, data->header.name, 0.05, tk::gui::color::WHITE);
        }
    }
    this->drawData(viewer);
}

void 
DataDrawable::forceUpdate() 
{
    counter -= 1;
}