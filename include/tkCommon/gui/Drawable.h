#pragma once

#include <iostream>
#include <vector>
#include <mutex>
#include <map>

namespace tk{namespace gui{

	class Viewer;

	class Drawable{

	public:
		std::mutex *draw_mutex = nullptr;

		Drawable & operator=(const Drawable& obj) { }

		//virtual void draw() {}

		//virtual void draw2D(int width, int height, float xLim, float yLim) {}

		virtual void draw(tk::gui::Viewer *viewer) {}

		virtual void draw2D(tk::gui::Viewer *viewer) {}

		~Drawable(){if(draw_mutex != nullptr) delete draw_mutex;}

	};

	template<typename T, typename = std::enable_if<std::is_base_of<Drawable, T>::value>>
	class DrawBuffer : public Drawable{

	public:

		std::vector<T> buffer;
		std::mutex mutex;

		DrawBuffer &operator=(const DrawBuffer &s){
			buffer = s.buffer;
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


	class DrawMap {
	public:
		std::map<std::string, Drawable*>  map;

		template<typename T, typename = std::enable_if<std::is_base_of<Drawable, T>::value>>
		void insert(std::string name, T* d ){

			auto search = map.find(name);
			if (search != map.end()) {
				Drawable *temp = map[name];
				map.erase(name);
				delete temp;
			}

			T *copy = new T();
			if(d != nullptr){
				*copy = *d;
			}
			if(copy->draw_mutex == nullptr){
				copy->draw_mutex = new std::mutex;
			}
			map.insert(std::pair<std::string, Drawable*>(name, copy));
		}

		template<typename T, typename = std::enable_if<std::is_base_of<Drawable, T>::value>>
		void update(std::string name, T* d ){

			auto search = map.find(name);
			if (search != map.end()) {
				map[name]->draw_mutex->lock();
				*((T*)map[name]) = *d;
				map[name]->draw_mutex->unlock();
			}
			else{
				insert<T>(name, d);
			}
		}

		void add(std::string name, Drawable* d){
			auto search = map.find(name);
			if (search != map.end()) {
				map[name] = d;
			} else {
				map.insert(std::pair<std::string,Drawable*>(name,d));
			}
		}

		void draw2D(Viewer *viewer){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(it->second->draw_mutex != nullptr)
					it->second->draw_mutex->lock();
				it->second->draw2D(viewer);
				if(it->second->draw_mutex != nullptr)
					it->second->draw_mutex->unlock();
			}
		}

		void draw(Viewer *viewer){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				if(it->second->draw_mutex != nullptr)
					it->second->draw_mutex->lock();
				it->second->draw(viewer);
				if(it->second->draw_mutex != nullptr)
					it->second->draw_mutex->unlock();
			}
		}

		~DrawMap(){
			for (std::map<std::string,Drawable*>::iterator it = map.begin(); it!=map.end(); ++it){
				Drawable *t = it->second;
				it->second = nullptr;
				delete t;
			}
		}
	};

}}
