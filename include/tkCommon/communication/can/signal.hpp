/*
 * signal.hpp
 *
 *  Created on: 04.10.2013
 *      Author: downtimes
 */
#pragma once

#include <string>
#include <iosfwd>
#include <set>

namespace tk { namespace communication {


static inline std::string& trim(std::string& str, const std::string& toTrim = " ") {
	std::string::size_type pos = str.find_last_not_of(toTrim);
	if (pos == std::string::npos) {
		str.clear();
	} else {
		str.erase(pos + 1);
		str.erase(0, str.find_first_not_of(toTrim));
	}
	return str;
}

static inline std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

static inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


enum class ByteOrder {
	MOTOROLA,
	INTEL
};

enum class Sign {
	UNSIGNED,
	SIGNED
};

enum class Multiplexor {
	NONE,
	MULTIPLEXED,
	MULTIPLEXOR
};


/**
 * This class represents a Signal contained in a Message of a DBC-File.
 * One can Query all the necessary information from this class to define
 * a Signal
 */
class Signal {

	typedef std::set<std::string> toList;
	//The name of the Signal in the DBC-File
	std::string name;
	//The Byteorder of the Signal (@see: endianess)
	ByteOrder order;
	//The Startbit inside the Message of this Signal. Allowed values are 0-63
	unsigned short startBit;
	//The Length of the Signal. It can be anything between 1 and 64
	unsigned short length;
	//If the Data contained in the Signal is signed or unsigned Data
	Sign sign;
	//Depending on the information given above one can calculate the minimum of this Signal
	double minimum;
	//Depending on the inforamtion given above one can calculate the maximum of this Signal
	double maximum;
	//The Factor for calculating the physical value: phys = digits * factor + offset
	double factor;
	//The offset for calculating the physical value: phys = digits * factor + offset
	double offset;
	//String containing an associated unit.
	std::string unit;
	//Contains weather the Signal is Multiplexed and if it is, multiplexNum contains multiplex number
	Multiplexor multiplexor;
	//Contains the multiplex Number if the Signal is multiplexed
	unsigned short multiplexNum;
	//Contains to which Control Units in the CAN-Network the Signal shall be sent
	toList to;

public:
	//Overload of operator>> to allow parsing from DBC Streams
	friend std::istream& operator>>(std::istream& in, Signal& sig) {
		int c = in.peek();
		if ('B' == c) {
			in.setstate(std::ios_base::failbit);
			return in;
		}
		std::string line;
		std::getline(in, line);
		if (!line.empty() && *line.rbegin() == '\r') line.erase(line.length() - 1, 1);
		if (line.empty()) {
			in.setstate(std::ios_base::failbit);
			return in;
		}

		std::istringstream sstream(line);
		std::string preamble;
		sstream >> preamble;
		//Check if we are actually reading a Signal otherwise fail the stream
		if (preamble != "SG_") {
			sstream.setstate(std::ios_base::failbit);
			return in;
		}

		//Parse the Signal Name
		sstream >> sig.name;

		std::string multi;
		sstream >> multi;

		//This case happens if there is no Multiplexor present
		if (multi == ":") {
			sig.multiplexor = Multiplexor::NONE;
		//Case with multiplexor
		} else {
			if (multi == "M") {
				sig.multiplexor = Multiplexor::MULTIPLEXOR;
			} else {
				//The multiplexor looks like that 'm12' so we ignore the m and parse it as integer
				std::istringstream multstream(multi);
				multstream.ignore(1);
				unsigned short multiNum;
				multstream >> multiNum;
				sig.multiplexor = Multiplexor::MULTIPLEXED;
				sig.multiplexNum = multiNum;
			}
			//ignore the next thing which is a ':'
			sstream >> multi;
		}

		sstream >> sig.startBit;
		sstream.ignore(1);
		sstream >> sig.length;
		sstream.ignore(1);

		int order;
		sstream >> order;
		if (order == 0) {
			sig.order = ByteOrder::MOTOROLA;
		} else {
			sig.order = ByteOrder::INTEL;
		}

		char sign;
		sstream >> sign;
		if (sign == '+') {
			sig.sign = Sign::UNSIGNED;
		} else {
			sig.sign = Sign::SIGNED;
		}

		sstream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
		sstream >> sig.factor;
		sstream.ignore(1);
		sstream >> sig.offset;
		sstream.ignore(1);

		sstream.ignore(std::numeric_limits<std::streamsize>::max(), '[');
		sstream >> sig.minimum;
		sstream.ignore(1);
		sstream >> sig.maximum;
		sstream.ignore(1);

		std::string unit;
		sstream >> unit;
		sig.unit = trim(unit, "\"");

		std::string to;
		sstream >> to;
		std::vector<std::string> toStrings = split(to, ',');
		std::move(toStrings.begin(), toStrings.end(), std::inserter(sig.to, sig.to.begin()));

		return in;

	}

	//Getter for all the Values contained in a Signal
	std::string getName() const { return name; }
	ByteOrder getByteOrder() const { return order; }
	unsigned short getStartbit() const { return startBit; }
	unsigned short getLength() const { return length; }
	Sign getSign() const { return sign; }
	double getMinimum() const { return minimum; }
	double getMaximum() const { return maximum; }
	double getFactor() const { return factor; }
	double getOffset() const { return offset; }
	std::string getUnit() const { return unit; }
	Multiplexor getMultiplexor() const { return multiplexor; }
	unsigned short getMultiplexedNumber() const { return multiplexNum; }
	toList getTo() const { return to; }

};

}}