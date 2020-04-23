/*
 * message.hpp
 *
 *  Created on: 04.10.2013
 *      Author: downtimes
 */
#pragma once

#include <string>
#include <vector>
#include <iosfwd>
#include <cstdint>
#include <set>

#include "signal.hpp"

namespace tk { namespace communication {

/**
 * Class representing a Message in the DBC-File. It allows its user to query
 * Data and to iterate over the Signals contained in the Message
 */class Message {

	typedef std::vector<Signal> signals_t;
	//Name of the Message
	std::string name;
	//The CAN-ID assigned to this specific Message
	std::uint32_t id;
	//The length of this message in Bytes. Allowed values are between 0 and 8
	std::size_t dlc;
	//String containing the name of the Sender of this Message if one exists in the DB
	std::string from;
	//List containing all Signals which are present in this Message
	signals_t signals;

public:
	typedef signals_t::const_iterator const_iterator;
	//Overload of operator>> to enable parsing of Messages from streams of DBC-Files
	friend std::istream& operator>>(std::istream& in, Message& msg) {
		std::string preamble;
		in >> preamble;
		//Check if we are actually reading a Message otherwise fail the stream
		if (preamble != "BO_") {
			in.setstate(std::ios_base::failbit);
			return in;
		}

		//Parse the message ID
		in >> msg.id;

		//Parse the name of the Message
		std::string name;
		in >> name;
		msg.name = name.substr(0, name.length() - 1);

		//Parse the Messages length
		in >> msg.dlc;

		//Parse the sender;
		in >> msg.from;

		in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		//As long as there is a Signal, parse the Signal
		while(in) {
			Signal sig;
			in >> sig;
			if (in) {
				msg.signals.push_back(sig);
			}
		}

		in.clear();
		return in;
	}

	//Getter functions for all the possible Data one can request from a Message
	std::string getName() const { return name; }
	std::uint32_t getId() const { return id; }
	std::size_t getDlc() const { return dlc; }
	std::string getFrom() const { return from; }
	std::set<std::string> getTo() const {
		std::set<std::string> collection;
		for (auto sig : signals) {
			auto toList = sig.getTo();
			collection.insert(toList.begin(), toList.end());
		}
		return collection;
	}

	/*
	 * Functionality to access the Signals contained in this Message
	 * either via the iterators provided by begin() and end() or by
	 * random access operator[]
	 */
	const_iterator begin() const { return signals.begin(); }
	const_iterator end() const { return signals.end(); }
	signals_t::const_reference operator[](std::size_t elem) {
		return signals[elem];
	}

};

}}