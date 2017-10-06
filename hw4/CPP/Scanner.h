#ifndef _SCANNER_H
#define _SCANNER_H

#include <fstream>
#include <string>
#include "Token.h"

class Scanner
{
private:
	std::ifstream inpPgm;
	std::string curLine;
	std::string::iterator lineIt;
	Token *curToken;

public:
	bool hasNextLine();

	bool hasNextToken();
	
	void populateToken();
	
	Token* getNextToken();

public:
	Scanner(const char* inputProgram);
};
#endif
