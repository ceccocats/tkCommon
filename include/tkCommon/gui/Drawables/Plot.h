#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/math/Vec.h"


namespace tk{ namespace gui{

	class Plot : public Drawable {
        public:
            enum type_t : int {
                LINE = 0, CIRCLES = 1, AXIS = 2, ARROW = 3
            };

        private:
            std::string name = "plot";
            type_t type;
            CircularArray<tk::common::Tfpose> points;

            float circleRay;
            int   circleRes;
            float lineW;
            float pointW;
            float poseW;

            tk::math::Vec3<float> mins, maxs;
            void updateLimits() {
                float x = points.head(0).matrix()(0, 3);
                float y = points.head(0).matrix()(1, 3);
                float z = points.head(0).matrix()(2, 3);

                if(points.size() == 1) {
                    mins = { x, y, z };
                    maxs = { x, y, z };
                }
                if(x < mins.x()) mins.x() = x;
                if(y < mins.y()) mins.y() = y;
                if(z < mins.z()) mins.z() = z;
                if(x > maxs.x()) maxs.x() = x;
                if(y > maxs.y()) maxs.y() = y;
                if(z > maxs.z()) maxs.z() = z;
            }

        public:

            Plot(std::string name, int maxDataDim, type_t type, float drawSize){
                this->name = name;
                points.setDim(maxDataDim);
                this->type = type;

                circleRay = drawSize;
                circleRes = 32;
                lineW     = drawSize;
                pointW    = drawSize;
                poseW     = drawSize;

            }

            ~Plot(){

            }

            void addPoint(tk::common::Tfpose tf) {
                points.add(tf);
                updateLimits();
            }
            void addPoint(tk::math::Vec3<float> pt) {
                addPoint(tk::common::odom2tf(pt.x(), pt.y(), pt.z(), 0,0,0));
            }

            void onInit(tk::gui::Viewer *viewer){
            
            }

            void draw(tk::gui::Viewer *viewer){

            }

            void imGuiInfos(){

            }

            void onClose(){

            }

            std::string toString(){
                return name;
            }
	};
}}