#ifndef _SYMBOLTABLE_H
#define _SYMBOLTABLE_H
#include <string>
#include <queue>

#include "Variable.h"
#include "Function.h"
class SymbolTable
{
public:
	std::vector< std::pair<std::string, Variable*> > globalVariables;
	std::vector< std::pair<std::string, Function*> > functions;
	Function* curFunction;
	int lableCnt;
	int globalVarCnt;	/*Used to keep track of number of global variables. In case of arrays, the size of the array is updated in this count*/
	std::queue<Function*> functionQueue;

	SymbolTable()
		: lableCnt(0)
		, globalVarCnt(0)
		, curFunction(NULL)
	{}
};
#endif