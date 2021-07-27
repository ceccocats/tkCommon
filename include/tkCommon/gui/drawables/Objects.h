#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/PerceptionData.h"
#include "tkCommon/common.h"
#include "tkCommon/geometry.h"

namespace tk{ namespace gui{

    class WorldBox : public tk::gui::Drawable {
        public:

        tk::data::ObjectBuffer *data;
        tk::data::ObjectBuffer copy;
        
        std::vector<int> classes;
        uint32_t counter = 0;
        bool update = false;
        std::vector<std::vector<int>> face_id;

        tk::gui::Color_t class_colors[10] = {
            tk::gui::color::RED        , // PEDESTRIAN
            tk::gui::color::GREEN      , // CAR
            tk::gui::color::PURPLE     , // TRUCK
            tk::gui::color::PURPLE     , // BUS
            tk::gui::color::BLUE       , // MOTOBIKE
            tk::gui::color::RED        , // CYCLE
            tk::gui::color::RED        , // RIDER
            tk::gui::color::YELLOW     , // LIGHT
            tk::gui::color::PINK       , // ROADSIGN
            tk::gui::color::PURPLE       // TRAIN
        };

        WorldBox(tk::data::ObjectBuffer& data, std::string name = "Objects"){

            this->data = &data;
            update = true;

        }
        
        void updateData(tk::data::ObjectBuffer& s){
            data = &s;
            update = true;
        }
        void draw(tk::gui::Viewer *viewer){
            if(data->isChanged(counter) || update){
                update      = false;

                data->lockRead();
                copy.clear();
                classes.clear();
                for(int i = 0; i < (*data).size(); i++){

                    copy.push_back((*data)[i]);
                    
                }
                data->unlockRead();
            }

            glPushMatrix();
            //glColor4f(1,0,0,1);
            tk::gui::Color_t c;
            glLineWidth(5.0f);
            for(int i = 0; i < copy.size(); i++){
                c = class_colors[copy[i].cl % 10];
                glDisable(GL_DEPTH_TEST);
                glPushMatrix();
                {
                    glColor4f(c.r(),c.g(),c.b(),0.5);
                    glTranslatef(copy[i].world.pose.x(), copy[i].world.pose.y(), copy[i].world.pose.z());
                    glRotatef(copy[i].world.rot.z(), 0,0,1);
                    glRotatef(copy[i].world.rot.y(), 0,1,0);
                    glRotatef(copy[i].world.rot.x(), 1,0,0);
                    tkDrawCube(tk::common::Vector3<float>(0,0,0), copy[i].world.size, true);
                }
                glPopMatrix();

                glEnable(GL_DEPTH_TEST);

            }
            glPopMatrix();
        }

        void tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled) {
            if (!filled)
                glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

            glPushMatrix();
            glTranslatef(pose.x(), pose.y(), pose.z());
            glScalef(size.x(), size.y(), size.z());

            // BACK
            glBegin(GL_POLYGON);
            glVertex3f(  0.5, -0.5, 0.5 );
            glVertex3f(  0.5,  0.5, 0.5 );
            glVertex3f( -0.5,  0.5, 0.5 );
            glVertex3f( -0.5, -0.5, 0.5 );
            glEnd();
            
            // RIGHT
            glBegin(GL_POLYGON);
            glVertex3f( 0.5, -0.5, -0.5 );
            glVertex3f( 0.5,  0.5, -0.5 );
            glVertex3f( 0.5,  0.5,  0.5 );
            glVertex3f( 0.5, -0.5,  0.5 );
            glEnd();
            
            // LEFT
            glBegin(GL_POLYGON);
            glVertex3f( -0.5, -0.5,  0.5 );
            glVertex3f( -0.5,  0.5,  0.5 );
            glVertex3f( -0.5,  0.5, -0.5 );
            glVertex3f( -0.5, -0.5, -0.5 );
            glEnd();
            
            // TOP
            glBegin(GL_POLYGON);
            glVertex3f(  0.5,  0.5,  0.5 );
            glVertex3f(  0.5,  0.5, -0.5 );
            glVertex3f( -0.5,  0.5, -0.5 );
            glVertex3f( -0.5,  0.5,  0.5 );
            glEnd();
            
            // BOTTOM
            glBegin(GL_POLYGON);
            glVertex3f(  0.5, -0.5, -0.5 );
            glVertex3f(  0.5, -0.5,  0.5 );
            glVertex3f( -0.5, -0.5,  0.5 );
            glVertex3f( -0.5, -0.5, -0.5 );
            glEnd();

            glPopMatrix();

            if (!filled)
                glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
        }
    };
}}