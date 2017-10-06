/**
*	Source : Grammar.h
*	Description: Provides the grammar to be used to parse the program. Provides interface for each non-terminal in the grammar. 
*				 Non-Terminal interfaces will validate the program usign the terminals of the grammar.
*/
#ifndef _GRAMMAR_H
#define _GRAMMAR_H
#include <fstream>
#include "Parser.h"
#include "Function.h"
#include "ASTNode.h"


class Parser;

class Grammar
{
public:
	/*Interfaces for each of the non-terminal in the grammar*/
	bool program();

	bool funcList();

	bool funcListPrime(std::string dataType, std::string idName);

	bool func();

	bool funcPrime(Function *newFunction);

	bool funcDecl(Function **newFunction);

	bool funcDeclPrime(Function *newFunction);

	bool typeName(std::string &datatype);

	bool typeNamePrime(std::string &datatype);

	bool paramList(std::vector< std::pair<std::string, Variable*> > &arguments);

	bool paramListPrime(std::vector< std::pair<std::string, Variable*> > &arguments);

	bool nonEmptyListPrime(std::vector< std::pair<std::string, Variable*> > &arguments);

	bool dataDecls();

	bool dataDeclsPrime(std::string dataType, std::string idName);

	bool dataDeclsDPrime();

	bool dataFuncDecl(std::string dataType, std::string idName);

	bool idList(std::string dataType);

	bool idListPrime(std::string dataType);

	bool id(std::string dataType);

	bool idPrime(std::string dataType, std::string idName);

	bool idUse(std::string idName, std::string &assignStmt);

	bool blockStatements();

	bool statements();

	bool statement();

	bool statementPrime(std::string idName);

	bool assignment(std::string idName);

	bool funcCall(std::string idName);

	bool exprList(std::queue<std::string> &arguments);

	bool nonEmptyExprList(std::queue<std::string> &arguments);

	bool nonEmptyExprListPrime(std::queue<std::string> &arguments);

	bool ifStatement();

	bool conditionExpression(ASTNode **root);

	bool conditionExpressionPrime(ASTNode **root);

	bool conditionOp(ASTNode **root);

	bool condition(ASTNode **root);

	bool comparisionOp(ASTNode **root);

	bool whileStatment();

	bool returnStatement();

	bool returnStatementPrime();

	bool breakStatement();

	bool continueStatement();

	bool expression(ASTNode **root);

	bool expressionPrime(ASTNode **root);

	bool addOp(ASTNode **root);

	bool term(ASTNode **root);

	bool termPrime(ASTNode **root);

	bool mulOp(ASTNode **root);

	bool factor(ASTNode **root);

	bool factorPrime(std::string idName, ASTNode **root);

	void instanciateParser(Parser* newParser);

private:
	Parser *parse;

public:
		std::ofstream *outFile;

		Grammar(std::ofstream *newOutFile)
		{
			outFile = newOutFile;
		}

};
#endif

	

	

	

	

	

	

	

	

	

	

	
