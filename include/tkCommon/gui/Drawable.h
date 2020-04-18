#pragma once

#include <vector>
#include <mutex>

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
		std::mutex mutex;

		virtual void draw() {
			for(int i = 0; i < buffer.size(); i++){
				mutex.lock();
				buffer[i].draw();
				mutex.unlock();
			}
		}

		virtual void draw2D(int width, int height, float xLim, float yLim) {
			for(int i = 0; i < buffer.size(); i++){
				mutex.lock();
				buffer[i].draw2D(width, height, xLim, yLim);
				mutex.unlock();
			}
		}

		void push_back(T &obj){
			mutex.lock();
			buffer.push_back(obj);
			mutex.unlock();
		}

		T& pop_back(){
			return buffer.pop_back();
		}

		void clear(){
			mutex.lock();
			buffer.clear();
			mutex.unlock();
		}

	};

}}
