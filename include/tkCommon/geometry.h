#pragma once

#include "tkCommon/gui/Viewer.h"
#include "tkCommon/rt/Lockable.h"

namespace tk{ namespace common{

	class Polygon : public tk::rt::Lockable{
		public:
			std::vector<tk::common::Vector2<float>> points;	

			Polygon(){}
			Polygon(const tk::common::Polygon& d){
				this->points = d.points;
			}
			~Polygon(){}

			bool within(const tk::common::Vector2<float> &aPoint) const
			{
				int i, j;
				bool c = false;
				for (i = 0, j = points.size()-1; i < points.size(); j = i++) {
					if ( ((points[i].y()>aPoint.y()) != (points[j].y()>aPoint.y())) &&
					(aPoint.x() < (points[j].x()-points[i].x()) * (aPoint.y()-points[i].y()) / (points[j].y()-points[i].y()) + points[i].x()) )
						c = !c;
				}
				return c;
			}
	};

	class Prism : public Polygon{
		public:
			float base_z;
			float height;

			Prism(){}
			Prism(const tk::common::Prism& d){
				this->points = d.points;
				this->height = d.height;
				this->base_z = d.base_z;
			}
			~Prism(){}

			float distance(const tk::common::Vector2<float> &aPoint) const 
			{
				float min_r = std::numeric_limits<float>::max();
				float dx, dy, r;
				for (int i = 0; i < points.size(); ++i) {
					dx = points[i].x() - aPoint.x();
					dy = points[i].y() - aPoint.y();
					min_r = std::min(min_r, std::hypot(dx, dy));
				}
				return min_r;
			}

			Prism& operator=(const tk::common::Prism& d){
				this->points = d.points;
				this->height = d.height;
				this->base_z = d.base_z;
				return *this;
			}
	};

	class Prisms : public tk::rt::Lockable{
		public:
			tk::common::Tfpose tf = tk::common::Tfpose::Identity();
			std::vector<Prism> data;
			Prisms(){}
			~Prisms(){}

			Prisms& operator=(const tk::common::Prisms& s) {
				this->tf = s.tf;
				this->data = s.data;
				return *this;
			}
	};

	/*class Polygon : public tk::gui::Drawable{
	public:
		std::vector<tk::common::Vector2<float>> points;
		tk::gui::Color_t color;

		void draw2D(tk::gui::Viewer *viewer){
			glPushMatrix();{
				tk::gui::Viewer::tkSetColor(color);
				glBegin(GL_POLYGON);
				for(int i = 0; i < points.size(); i++){
					glVertex2f(points[i].x, points[i].y);
				}
				glEnd();
			}glPopMatrix();
		}
	};*/

	/*class Prism : public Polygon {
	public:
		float base_z;
		float height;

		void draw2D(tk::gui::Viewer *viewer){}
		void draw(tk::gui::Viewer *viewer){
			glPushMatrix();{
				glDepthMask(GL_FALSE);
				tk::gui::Viewer::tkSetColor(color);
				glBegin(GL_POLYGON);
				for(int i = points.size()-1;i >=0; i--){
					glVertex3f(points[i].x, points[i].y, base_z);
				}
				glEnd();
				for(int i = 0; i < points.size(); i++){
					glBegin(GL_POLYGON);
					glVertex3f(points[i].x, points[i].y, base_z);
					glVertex3f(points[i].x, points[i].y, base_z+height);
					glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z+height);
					glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z);
					glEnd();
				}
				glBegin(GL_POLYGON);
				for(int i = points.size()-1;i >=0; i--){
					glVertex3f(points[i].x, points[i].y, base_z + height);
				}
				glEnd();
				glDepthMask(GL_TRUE);

				tk::gui::Viewer::tkSetColor(color, 1);
				glLineWidth(2);
				glBegin(GL_LINES);
				for(int i = 0 ;i <points.size(); i++){
					glVertex3f(points[i].x, points[i].y, base_z);
					glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z);
				}
				for(int i = 0; i < points.size(); i++){
					glVertex3f(points[i].x, points[i].y, base_z);
					glVertex3f(points[i].x, points[i].y, base_z+height);
				}
				for(int i = 0 ;i <points.size(); i++){
					glVertex3f(points[i].x, points[i].y, base_z+height);
					glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z+height);
				}
				glEnd();
			}glPopMatrix();
		}
	};*/
}}