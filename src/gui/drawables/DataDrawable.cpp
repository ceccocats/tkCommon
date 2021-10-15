#include "tkCommon/gui/drawables/DataDrawable.h"

using namespace tk::gui;

DataDrawable::DataDrawable()
{
    this->data                  = nullptr;
    this->counter               = 0;
    this->mDrwHasReference      = true;
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
        if(enableTf) {
            // draw sensor tf
            glPushMatrix();
            glMultMatrixf((tf*data->header.tf).matrix().data());
            glLineWidth(2.5);
            glBegin(GL_LINES);
            glColor3f (1.0f,0.0f,0.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(1.0f,0.0f,0.0f);
            glColor3f (0.0f,1.0f,0.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(0.0f,1.0f,0.0f);
            glColor3f (0.0f,0.0f,1.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(0.0f,0.0f,1.0f);
            glEnd();
            glPopMatrix();
        }
    }
    this->drawData(viewer);
}

void 
DataDrawable::forceUpdate() 
{
    counter -= 1;
}