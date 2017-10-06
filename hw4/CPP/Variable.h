#ifndef _VARIABLE_H
#define _VARIABLE_H

#include <string>

class Variable
{
public:
	std::string name;
	std::string datatype;
	int loc;
	int type;
	int size;
	enum
	{
		SCALAR,
		ARRAY
	};
	Variable(std::string newDatatype
		, std::string newName
		, int newLoc = -1
		, int newType = SCALAR
		, int newSize = 0)
		: datatype(newDatatype)
		, name(newName)
		, loc(newLoc)
		, type(newType)
		, size(newSize)
	{}
};
#endif