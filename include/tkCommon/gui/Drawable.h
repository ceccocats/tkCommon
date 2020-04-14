#pragma once

namespace tk{namespace gui{

	class Drawable{

	public:

		virtual void draw() {}

		virtual void draw2D(int width, int height, float xLim, float yLim) {}

	};

}}
