#pragma once
#include <iostream>
#include "tkCommon/gui/shader/generic.h"
#include "tkCommon/gui/shader/axisPlot.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"

namespace tk{ namespace gui{

	class Viewer;

	class Drawable {
	
	public:
		bool enabled  = true;
		bool follow   = false;
		bool enableTf = false;
		tk::gui::shader::generic*	shader;
		std::string 				name = "no name";
		glm::mat4 					drwModelView;
		tk::common::Tfpose 			tf = tk::common::Tfpose::Identity();

		/** init method */
		virtual void onInit(tk::gui::Viewer *viewer) {}

		/** method used for update data in opegl from cpu and other things*/
		virtual void beforeDraw(tk::gui::Viewer *viewer) {}

		/**  method */
		virtual void draw(tk::gui::Viewer *viewer) {}

		/** draw in imgui settings window */
		virtual void imGuiSettings() {}

		/** draw in imgui infos window */
		virtual void imGuiInfos() {}
		
		/** close method */
		virtual void onClose() {}

		/** class name method */
		virtual std::string toString(){
			return name;
		}

		/** return true if the viewer should free this drawable **/
		virtual bool doFree() {
			return true;
		} 
	};

	class DrawableUnManaged : public Drawable {
		virtual bool doFree() {
			return false;
		} 
	};
}}
#include "tkCommon/gui/Viewer.h"