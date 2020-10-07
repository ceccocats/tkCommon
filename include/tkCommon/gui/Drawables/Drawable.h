#pragma once
#include <iostream>
#include "tkCommon/gui/shader/generic.h"
#include "tkCommon/gui/imgui/imgui.h"
#include "tkCommon/gui/imgui/imgui_impl_glfw.h"
#include "tkCommon/gui/imgui/imgui_impl_opengl3.h"

namespace tk{ namespace gui{

	class Viewer;

	class Drawable {
	
	public:
		bool enabled = true;
		bool follow = false;

		tk::common::Tfpose tf = tk::common::Tfpose::Identity();

		virtual void onInit(tk::gui::Viewer *viewer) {}
		virtual void draw(tk::gui::Viewer *viewer) {}
		virtual void draw2D(tk::gui::Viewer *viewer) {}
		virtual void imGuiSettings() {};
		virtual void imGuiInfos() {}
		virtual void onClose() {}

	protected:
		tk::gui::shader::generic*	shader;

		virtual void 	onChange(tk::gui::Viewer *viewer) {}

	};
}}