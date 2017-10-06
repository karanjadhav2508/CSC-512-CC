#include <vector>
#include "Parser.h"
#include "Util.h"
/*#include <iostream>*/
Token* Parser::nextToken()
{
	++tokenIT;

	while ((tokenIT != tokenVector->end()) 
		&& ((*tokenIT)->getID() == Token::IDTYPE_TABSPACE 
		|| (*tokenIT)->getID() == Token::IDTYPE_METASTATEMENT 
		|| (*tokenIT)->getID() == Token::IDTYPE_EMPTY))
	{
		if ((*tokenIT)->getID() == Token::IDTYPE_METASTATEMENT)
		{
			*(myGrammar->outFile) << (*tokenIT)->getTokenName() << std::endl;
		}
		++tokenIT;
	}

	if (tokenIT == tokenVector->end())
	{
		return NULL;
	}

	/*std::cout << "Token : " << (*tokenIT)->getTokenName() << std::endl;*/
	return *tokenIT;
}

Token* Parser::curToken()
{
	if (tokenIT == tokenVector->end())
	{
		return NULL;
	}

	return *tokenIT;
}

void Parser::addToken(Token *newToken)
{
	tokenVector->push_back(newToken);
}


bool Parser::parseProgram()
{
	tokenIT = tokenVector->begin();
	while ((tokenIT != tokenVector->end())
		&& ((*tokenIT)->getID() == Token::IDTYPE_TABSPACE
		|| (*tokenIT)->getID() == Token::IDTYPE_METASTATEMENT
		|| (*tokenIT)->getID() == Token::IDTYPE_EMPTY))
	{
		if ((*tokenIT)->getID() == Token::IDTYPE_METASTATEMENT)
		{
			(*myGrammar->outFile) << (*tokenIT)->getTokenName() << std::endl;
		}
		++tokenIT;
	}
	if (tokenIT != tokenVector->end())
	{
		bool temp = myGrammar->program();
		/*std::cout << std::endl;
		while (tokenIT != tokenVector->end())
		{
			std::cout << (*tokenIT)->getTokenName();
			++tokenIT;
		}*/
		return temp;
	}
	return true;
}

void Parser::incrementVariableCount()
{
		numVar++;
}

void Parser::decrementVariableCount()
{
		numVar--;
}

void Parser::incrementFunctionCount()
{
	numFunc++;
}

void Parser::incrementStmtCount()
{
	numStmt++;
}

int Parser::getNumVar()
{
	return numVar;
}

int Parser::getNumFunc()
{
	return numFunc;
}

int Parser::getNumStmt()
{
	return numStmt;
}

SymbolTable* Parser::getSymbolTable()
{
	return mySymTab;
}

Parser::Parser(Grammar *newGrammar
				, SymbolTable* newSymTab) 
				: myGrammar(newGrammar)
				, numVar(0)
				, numFunc(0)
				, numStmt(0)
				, isDeclaration(false)
				, tokenVector(new std::vector<Token*>())
				, mySymTab(newSymTab)
{
	newGrammar->instanciateParser(this);
}

Parser::~Parser()
{
	for (std::vector<Token *>::iterator it = tokenVector->begin(); it != tokenVector->end(); ++it)
	{
		delete *it;
	}

	delete tokenVector;
}

void Parser::evaluateASTTree(ASTNode *root, bool isTop, bool constAllowed)
{
	if (root == NULL)
		return;
	
	std::string newValue;

	switch (root->type)
	{
	case ASTNode::NUMBER:
		if (!constAllowed)
		{
			newValue.append("local[");
			newValue.append(Util::to_string(mySymTab->curFunction->localVarCnt++));
			newValue.push_back(']');

			std::string constAssign(newValue);
			constAssign.append(" = ");
			constAssign.append(root->value);
			constAssign.push_back(';');
			mySymTab->curFunction->funcStats.push(constAssign);

			root->value.clear();
			root->value.append(newValue);
			root->type = ASTNode::OPERAND;
		}
		break;
	case ASTNode::OPERAND:
		break;
	case ASTNode::FUNCCALL:
		if (!isTop)
		{
			newValue.append("local[");
			newValue.append(Util::to_string(mySymTab->curFunction->localVarCnt++));
			newValue.push_back(']');

			std::string funcCallStmt(newValue);
			funcCallStmt.append(" = ");
			funcCallStmt.append(root->value);
			funcCallStmt.push_back(';');
			mySymTab->curFunction->funcStats.push(funcCallStmt);

			root->value.clear();
			root->value.append(newValue);
			root->type = ASTNode::OPERAND;
		}
		break;
	case ASTNode::OPERATION:
		evaluateASTTree(root->left, false);
		evaluateASTTree(root->right, false);

		if (!isTop)
		{
			newValue.append("local[");
			newValue.append(Util::to_string(mySymTab->curFunction->localVarCnt++));
			newValue.push_back(']');

			std::string oprStmt(newValue);
			oprStmt.append(" = ");
			oprStmt.append(root->left->value);
			oprStmt.push_back(' ');
			oprStmt.append(root->value);
			oprStmt.push_back(' ');
			oprStmt.append(root->right->value);
			oprStmt.push_back(';');

			mySymTab->curFunction->funcStats.push(oprStmt);
			root->value.clear();
			root->value.append(newValue);
			root->type = ASTNode::OPERAND;
			delete root->left;
			root->left = NULL;
			delete root->right;
			root->right = NULL;
		}
	}
}
