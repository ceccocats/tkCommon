#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <map>

namespace tk{namespace gui{

	class Viewer;

	class Drawable {
	private:
		std::mutex datamutex;
		bool modified = true;		

	public:
		bool enabled = true;
		tk::common::Tfpose tf;

		void lock() {
			datamutex.lock();
		}
		void unlock() {
			modified = true;
			datamutex.unlock();
		}
		virtual void onAdd(tk::gui::Viewer *viewer) {}
		virtual void draw(tk::gui::Viewer *viewer) {}
		virtual void draw2D(tk::gui::Viewer *viewer) {}
	
		void _beforeDraw(tk::gui::Viewer *viewer) {
			datamutex.lock();
			if(modified) {
				onAdd(viewer);
				modified = false;
			}
			datamutex.unlock();
		}
	};

	class DrawMap {
	public:
		std::map<std::string, Drawable*>  map;

		void add(std::string name, Drawable* d, tk::gui::Viewer *viewer = nullptr){
			map[name] = d;
		}

		void draw2D(Viewer *viewer){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(!it->second->enabled)
					continue;
				
				it->second->_beforeDraw(viewer);
				it->second->draw2D(viewer);

			}
		}

		void draw(Viewer *viewer){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(!it->second->enabled)
					continue;

				it->second->_beforeDraw(viewer);
				it->second->draw(viewer);
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
