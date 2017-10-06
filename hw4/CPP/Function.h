#ifndef _FUNCTION_H
#define _FUNCTION_H

#include <queue>
#include <stack>
#include "Variable.h"
class Function
{
public:
	std::string name;
	std::vector< std::pair<std::string, Variable*> > arguments;
	std::vector< std::pair<std::string, Variable*> > localVariables;
	int localVarCnt; /*Used to keep track of number of local variables. In case of arrays, the size of the array is updated in this count*/
	std::queue<std::string> funcStats;
	std::string datatype;
	std::stack< std::pair<int, int> > whileLables;
	bool isDeclaration;

	Function(std::string newDatatype, std::string newName)
		: datatype(newDatatype)
		, name(newName)
		, localVarCnt(0)
		, isDeclaration(true)
	{}

	~Function()
	{
		for (std::vector< std::pair<std::string, Variable*> >::iterator it = arguments.begin(); it != arguments.end(); ++it)
		{
			delete it->second;
		}

		for (std::vector< std::pair<std::string, Variable*> >::iterator it = localVariables.begin(); it != localVariables.end(); ++it)
		{
			delete it->second;
		}
	}

	friend std::ofstream& operator<<(std::ofstream& outFile, Function& printFunc);
};

#endif