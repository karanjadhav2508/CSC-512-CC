#ifndef _UTIL_H
#define _UTIL_H

#include <string>
#include <queue>
#include <fstream>
#include "SymbolTable.h"
#include "ASTNode.h"

class Util
{
public:
	static std::pair<bool, int> isSymbol(std::string::iterator pos);
	
	static std::pair<bool, int> isDoubleSymbol(std::string::iterator pos);
	
	static bool isLetter(std::string::iterator pos);
	
	static bool isDigit(std::string::iterator pos);
	
	static bool isReservedWord(std::string curString);
	
	static bool isMetaChar(std::string::iterator pos);

	static void createFuncCallStatement(std::string &idName, std::queue<std::string> &arguments, std::string &funcCallStmt);

	static std::string getOperandString(std::string id, SymbolTable *symTab, std::string *posString = NULL);

	static std::string getAryPosStmt(int startLoc, std::string *posString, int newVarPos);

	static void printIfStmt(SymbolTable *symTab, ASTNode *root, int thenLable, int endLable);
	
	static std::string to_string(const int& num);
};
#endif
