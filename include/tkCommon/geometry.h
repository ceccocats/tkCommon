#pragma once

#include "tkCommon/gui/Viewer.h"

namespace tk{ namespace common{

	class Polygon : public tk::gui::Drawable{
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
	};

	class Prism : public Polygon {
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
	};

	typedef std::vector<Prism> Prisms;

	class Polyhedron : public tk::gui::Drawable{
	public:
		tk::gui::Viewer::object3D_t obj;
		tk::gui::Color_t color;
		bool textured;

		void load(std::string path){
			if(path.substr(path.size()-4,4) == ".obj"){
				path = path.substr(0,path.size()-4);
			}
			tk::gui::Viewer::tkLoadOBJ(path, obj);
		}

		void draw(tk::gui::Viewer *viewer){
			glPushMatrix(); {
				tk::gui::Viewer::tkSetColor(color);
				tk::gui::Viewer::tkDrawObject3D(&obj, 1, textured);
			} glPopMatrix();
		}
	};



}}