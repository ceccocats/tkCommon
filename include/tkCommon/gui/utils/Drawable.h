#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <map>

#include "tkCommon/gui/shader/shaders.h"

namespace tk{ namespace gui{

	class Viewer;

	class Drawable {
	
	public:
		bool enabled = true;
		bool follow = false;

		tk::common::Tfpose tf = tk::common::Tfpose::Identity();

		virtual void onInit(tk::gui::Viewer *viewer) {}
		virtual void onClose() {}
		virtual void onAdd(tk::gui::Viewer *viewer) {}
		virtual void draw(tk::gui::Viewer *viewer) {}
		virtual void draw2D(tk::gui::Viewer *viewer) {}

		bool _beforeDraw(tk::gui::Viewer *viewer) {
			datamutex.lock();

			if(drawInitted == false){
				drawInitted = true;
				this->onInit(viewer);
			}

			if(modified == true){
				modified = false;
				this->onAdd(viewer);
			}

			return drawInitted;
		}

		void lock() {
			datamutex.lock();
		}

		void unlock() {
			modified = true;
			datamutex.unlock();
		}

	protected:
		tk::gui::shader::generic*	shader;

		std::mutex 	datamutex;
		bool modified = true;

		bool drawInitted = false;
	};



	class DrawMap {
	public:
		std::vector<tk::common::Vector3<float>> centers;

		std::map<std::string, Drawable*>  map;

		void add(std::string name, Drawable* d, tk::gui::Viewer *viewer = nullptr){
			map[name] = d;
		}

		void draw2D(Viewer *viewer){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(!it->second->enabled)
					continue;
				
				if(it->second->_beforeDraw(viewer)) {
					glPushMatrix();
					glMultMatrixf(it->second->tf.matrix().data());
					it->second->draw2D(viewer);
					glPopMatrix();
				}

			}
		}

		void draw(Viewer *viewer) {
			centers.clear();

			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(!it->second->enabled)
					continue;
					
				if(it->second->follow) {
					centers.push_back(tk::common::tf2pose(it->second->tf));
				}

				if(it->second->_beforeDraw(viewer)) {
					glPushMatrix();
					glMultMatrixf(it->second->tf.matrix().data());
					it->second->draw(viewer);
					glPopMatrix();
				}
			}
		}

		void close() {
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				it->second->onClose();
			}
		}
	};



	template<typename T, typename = std::enable_if<std::is_base_of<Drawable, T>::value>>
	class DrawBuffer : public Drawable{

	public:

		std::vector<T> buffer;
		std::mutex mutex;

		DrawBuffer &operator=(const DrawBuffer &s){
			mutex.lock();
			buffer = s.buffer;
			mutex.unlock();
			return *this;
		}

		virtual void draw(tk::gui::Viewer *viewer) {
			for(int i = 0; i < buffer.size(); i++){
				mutex.lock();
				buffer[i].draw(viewer);
				mutex.unlock();
			}
		}

		virtual void draw2D(tk::gui::Viewer *viewer) {
			for(int i = 0; i < buffer.size(); i++){
				mutex.lock();
				buffer[i].draw2D(viewer);
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
