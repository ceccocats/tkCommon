/*
 * dbctree.hpp
 *
 *  Created on: 04.10.2013
 *      Author: downtimes
 */
#pragma once

#include <vector>
#include <iosfwd>
#include "message.hpp"

namespace tk { namespace communication {
/**
 * This is the Top class of the dbclib and the interface to the user.
 * It enables its user to iterate over the Messages of a DBC-File
 */

class DBCIterator {

	typedef std::vector<Message> messages_t;
	//This list contains all the messages which got parsed from the DBC-File
	messages_t messageList;

public:
	typedef messages_t::const_iterator const_iterator;

	//Constructors taking either a File or a Stream of a DBC-File
	explicit DBCIterator(const std::string& filePath) {
		std::ifstream file(filePath);
		if (file) {
			init(file);
		} else {
			throw std::invalid_argument("The File could not be opened");
		}
		file.close();
	}

	explicit DBCIterator(std::istream& stream) {
		init(stream);
	}

	/*
	 * Functionality to access the Messages parsed from the File
	 * either via the iterators provided by begin() and end() or by
	 * random access operator[]
	 */
	const_iterator begin() const { return messageList.begin(); }
	const_iterator end() const { return messageList.end(); }
	messages_t::const_reference operator[](std::size_t elem) const {
		return messageList[elem];
	}

private:
	void init(std::istream& stream) {
		messageList.clear();
		std::vector<Message> messages;
		do {
			Message msg;
			stream >> msg;
			if (stream.fail()) {
				stream.clear();
				stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			} else {
				messages.push_back(msg);
			}
		} while (!stream.eof());
		messageList.insert(messageList.begin(), messages.begin(), messages.end());
	}

};

}}