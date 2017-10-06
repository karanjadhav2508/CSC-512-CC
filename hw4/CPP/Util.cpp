#include <algorithm>
#include <string>
#include <sstream>
#include "Util.h"
#include "Token.h"
#include "Comparision.h"

std::pair<bool, int> Util::isSymbol(std::string::iterator pos)
{
	switch (*pos)
	{
	case '(':
		return std::pair<bool, int>(true, Token::SYMTYPE_LEFT_PARENTHESIS);
	case ')':
		return std::pair<bool, int>(true, Token::SYMTYPE_RIGHT_PARENTHESIS);
	case '{':
		return std::pair<bool, int>(true, Token::SYMTYPE_LEFT_BRACE);
	case '}':
		return std::pair<bool, int>(true, Token::SYMTYPE_RIGHT_BRACE);
	case '[':
		return std::pair<bool, int>(true, Token::SYMTYPE_LEFT_BRACKET);
	case ']':
		return std::pair<bool, int>(true, Token::SYMTYPE_RIGHT_BRACKET);
	case ',':
		return std::pair<bool, int>(true, Token::SYMTYPE_COMMA);
	case ';':
		return std::pair<bool, int>(true, Token::SYMTYPE_SEMICOLON);
	case '+':
		return std::pair<bool, int>(true, Token::SYMTYPE_PLUS);
	case '-':
		return std::pair<bool, int>(true, Token::SYMTYPE_MINUS);
	case '*':
		return std::pair<bool, int>(true, Token::SYMTYPE_STAR);
	case '/':
		return std::pair<bool, int>(true, Token::SYMTYPE_FORWARD_SLASH);
	case '#':				/**
							 * # is treated as symbol and later a call to isMetaChar
							 * will differentiate it from the actual set of symbols 
							 */
		return std::pair<bool, int>(true, Token::SYMTYPE_INVALID);
	case '>':
		return std::pair<bool, int>(true, Token::SYMTYPE_GT);
	case '<':
		return std::pair<bool, int>(true, Token::SYMTYPE_LT);
	case '=':
		return std::pair<bool, int>(true, Token::SYMTYPE_EQUAL);
	default:
		return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
	}
}

std::pair<bool, int> Util::isDoubleSymbol(std::string::iterator pos)
{
	switch (*pos)
	{
	case '!':
		if (*(pos + 1) != '=')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_NT_EQUAL);
	case '&':
		if (*(pos + 1) != '&')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_DOUBLE_AND);
	case '|':
		if (*(pos + 1) != '|')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_DOUBLE_OR);
	case '=':
		if (*(pos + 1) != '=')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_DOUBLE_EQUAL);
	case '<':
		if (*(pos + 1) != '=')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_LT_EQUAL);
	case '>':
		if (*(pos + 1) != '=')
		{
			return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
		}
		return std::pair<bool, int>(true, Token::SYMTYPE_GT_EQUAL);
	default:
		return std::pair<bool, int>(false, Token::SYMTYPE_INVALID);
	}

}

bool Util::isLetter(std::string::iterator pos)
{
	if ((*pos >= 'a' && *pos <= 'z')
		|| (*pos >= 'A' && *pos <= 'Z')
		|| (*pos == '_'))
	{
		return true;
	}
	return false;
}

bool Util::isDigit(std::string::iterator pos)
{
	if (*pos >= '0' && *pos <= '9')
	{
		return true;
	}
	return false;
}

bool Util::isReservedWord(std::string curString)
{
	if (!curString.compare("int")
		|| !curString.compare("void")
		|| !curString.compare("if")
		|| !curString.compare("while")
		|| !curString.compare("return")
		|| !curString.compare("read")
		|| !curString.compare("write")
		|| !curString.compare("print")
		|| !curString.compare("continue")
		|| !curString.compare("break")
		|| !curString.compare("binary")
		|| !curString.compare("decimal"))
	{
		return true;
	}

	return false;
}

bool Util::isMetaChar(std::string::iterator pos)
{
	/*std::cout << "Checking for meta\n";*/
	if (*pos == '/' && *(pos + 1) == '/')
	{
		/*std::cout << "It is double //\n";*/
		return true;
	}
	else if (*pos == '#')
	{
		return true;
	}
	return false;
}

void Util::createFuncCallStatement(std::string &idName, std::queue<std::string> &arguments, std::string &funcCallStmt)
{
	funcCallStmt.append(idName);
	funcCallStmt.push_back('(');

	if (!arguments.empty())
	{
		funcCallStmt.append(arguments.front());
		arguments.pop();
	}
	while (!arguments.empty())
	{
		funcCallStmt.push_back(',');
		funcCallStmt.append(arguments.front());
		arguments.pop();
	}
	funcCallStmt.push_back(')');
}

std::string Util::getOperandString(std::string id, SymbolTable *symTab, std::string *posString)
{
	std::string newOperand;

	std::vector< std::pair<std::string, Variable*> >::iterator locIT = std::find_if(symTab->curFunction->localVariables.begin(), symTab->curFunction->localVariables.end(), Comparision<std::string, Variable*>(id));
	if (locIT != symTab->curFunction->localVariables.end())
	{
		newOperand.append("local[");
		
		if (posString != NULL)
		{
			symTab->curFunction->funcStats.push(getAryPosStmt(locIT->second->loc, posString, symTab->curFunction->localVarCnt));
			newOperand.append("local[");
			newOperand.append(Util::to_string(symTab->curFunction->localVarCnt++));
			newOperand.push_back(']');
		}
		else
		{
			newOperand.append(Util::to_string(locIT->second->loc));
		}
		newOperand.push_back(']');
	}
	else
	{
		std::vector< std::pair<std::string, Variable*> >::iterator argIT = std::find_if(symTab->curFunction->arguments.begin(), symTab->curFunction->arguments.end(), Comparision<std::string, Variable*>(id));
		if (argIT != symTab->curFunction->arguments.end())
		{
			newOperand.append("local[");
			if (argIT->second->loc == -1)
			{
				std::string createAryRef;
				createAryRef.append("local[");
				createAryRef.append(Util::to_string(symTab->curFunction->localVarCnt));
				createAryRef.append("] = ");
				createAryRef.append(id);
				createAryRef.push_back(';');
				argIT->second->loc = symTab->curFunction->localVarCnt++;
				symTab->curFunction->funcStats.push(createAryRef);
			}

			if (posString != NULL)
			{
				symTab->curFunction->funcStats.push(getAryPosStmt(argIT->second->loc, posString, symTab->curFunction->localVarCnt));
				newOperand.append("local[");
				newOperand.append(Util::to_string(symTab->curFunction->localVarCnt++));
				newOperand.push_back(']');
			}
			else
			{
				newOperand.append(Util::to_string(argIT->second->loc));
			}
			newOperand.push_back(']');
		}
		else
		{
			std::vector< std::pair<std::string, Variable*> >::iterator glbIT = std::find_if(symTab->globalVariables.begin(), symTab->globalVariables.end(), Comparision<std::string, Variable*>(id));
			if (glbIT != symTab->globalVariables.end())
			{
				newOperand.append("global[");
				if (posString != NULL)
				{
					symTab->curFunction->funcStats.push(getAryPosStmt(glbIT->second->loc, posString, symTab->curFunction->localVarCnt));
					newOperand.append("local[");
					newOperand.append(Util::to_string(symTab->curFunction->localVarCnt++));
					newOperand.push_back(']');
				}
				else
				{
					newOperand.append(Util::to_string(glbIT->second->loc));
				}
				newOperand.push_back(']');
			}
			else
			{
				newOperand.append("varNotFound");
			}
		}
	}
	return newOperand;
}

std::string Util::getAryPosStmt(int startLoc, std::string *posString, int newVarPos)
{
	std::string createAryRef;
	createAryRef.append("local[");
	createAryRef.append(Util::to_string(newVarPos));
	createAryRef.append("] = ");
	createAryRef.append(Util::to_string(startLoc));
	createAryRef.append(" + ");
	createAryRef.append(*posString);
	createAryRef.push_back(';');
	return createAryRef;
}

void Util::printIfStmt(SymbolTable *symTab, ASTNode *root, int thenLable, int endLable)
{
	std::string ifStmt("if( ");

	if (root->type == ASTNode::OPERATION)
	{
		ifStmt.append(root->left->value);
		ifStmt.push_back(' ');
		ifStmt.append(root->value);
		ifStmt.push_back(' ');
		ifStmt.append(root->right->value);
	}
	else
	{
		ifStmt.append(root->value);
	}

	ifStmt.append(" ) goto c");
	ifStmt.append(Util::to_string(thenLable));
	ifStmt.push_back(';');


	std::string endGotoStmt("goto c");
	endGotoStmt.append(Util::to_string(endLable));
	endGotoStmt.push_back(';');

	std::string thenLableStmt("c");
	thenLableStmt.append(Util::to_string(thenLable));
	thenLableStmt.append(":;");

	symTab->curFunction->funcStats.push(ifStmt);
	symTab->curFunction->funcStats.push(endGotoStmt);
	symTab->curFunction->funcStats.push(thenLableStmt);

	return;
}

std::string Util::to_string(const int& num)
{
	std::ostringstream outString;
        outString << num;
        return outString.str();
}

