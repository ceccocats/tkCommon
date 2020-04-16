#pragma once

#include <vector>

namespace tk{namespace gui{

	class Drawable{

	public:

		virtual void draw() {}

		virtual void draw2D(int width, int height, float xLim, float yLim) {}

	};

	template<typename T, typename = std::enable_if<std::is_base_of<Drawable, T>::value>>
	class DrawBuffer : public Drawable{

	public:

		std::vector<T> buffer;

		virtual void draw() {
			for(int i = 0; i < buffer.size(); i++){
				buffer[i].draw();
			}
		}

		virtual void draw2D(int width, int height, float xLim, float yLim) {
			for(int i = 0; i < buffer.size(); i++){
				buffer[i].draw2D(width, height, xLim, yLim);
			}
		}

		void push_back(T &obj){
			buffer.push_back(obj);
		}

		T& pop_back(){
			return buffer.pop_back();
		}

		void clear(){
			buffer.clear();
		}

	};

}}
