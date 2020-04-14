#pragma once

#include <vector>

namespace tk{namespace gui{

	class Drawable{

	public:

		virtual void draw() {}

		virtual void draw2D(int width, int height, float xLim, float yLim) {}

	};

	class DrawBuffer : public Drawable{

		std::vector<Drawable *> buffer;

		virtual void draw() {
			for(int i = 0; i < buffer.size(); i++){
				buffer[i]->draw();
			}
		}

		virtual void draw2D(int width, int height, float xLim, float yLim) {
			for(int i = 0; i < buffer.size(); i++){
				buffer[i]->draw2D(width, height, xLim, yLim);
			}
		}

	};

}}
