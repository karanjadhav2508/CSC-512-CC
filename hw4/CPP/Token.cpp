#include "Token.h"
Token::Token()
{
	id = IDTYPE_EMPTY;
	isString = false;
	isStringComplete = false;
	symType = SYMTYPE_INVALID;
}

void Token::push(char c)
{
	myToken.push_back(c);
}

void Token::setID(int newID)
{
	id = newID;
}

int Token::getID()
{
	return id;
}

void Token::setSymType(int newSymType)
{
	symType = newSymType;
}

int Token::getSymType()
{
	return symType;
}

std::string Token::getTokenName()
{
	return myToken;
}

char Token::last()
{
	return myToken.at(myToken.length() - 1);
}
