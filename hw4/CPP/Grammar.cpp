#include <vector>
#include <cstring>
#include <cstdlib>
#include "Grammar.h"
#include "Token.h"
#include "ASTNode.h"
#include "Util.h"

/*#include <iostream>*/

bool Grammar::program()
{
	/**
	 * <program> --> <typename> ID <program prime> | empty
	 * <program prime> --> <data decls prime> | <func list prime>
	 */
	/*std::cout << "Program" << std::endl;*/

	std::string dataType;
	if (typeName(dataType))
	{
		if(parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
		{
			std::string idName(parse->curToken()->getTokenName());
			if (parse->nextToken()
				&& (parse->curToken()->getSymType() == Token::SYMTYPE_COMMA
				|| parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET
				|| parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON))
			{
				if (!dataDeclsPrime(dataType, idName))
				{
					return false;
				}
			}
			else
			{
				if (!funcListPrime(dataType, idName))
				{
					return false;
				}
			}
		}
		else
		{
			return false;
		}
	}
	if (!parse->curToken())
	{
		return true;
	}
	else
	{
		return false;
	}
}

bool Grammar::funcList()
{
	/**
	 * Production:	<func list> --> <func> <func list>
	 *								| empty
	 */
	/*std::cout << "funcList" << std::endl;*/
	if (parse->curToken()
		&& (!strcmp(parse->curToken()->getTokenName().c_str(), "int")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "void")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "binary")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "decimal")))
	{
		return func() && funcList();
	}

	return true;
}

bool Grammar::funcListPrime(std::string dataType, std::string idName)
{
	/**
	 * Production:	<func list prime> --> <func decl prime> <func prime> <func list>
	 */
	/*std::cout << "funcListPrime" << std::endl;*/
	Function *newFunction = new Function(dataType, idName);
	return funcDeclPrime(newFunction) && funcPrime(newFunction) && funcList();
}

bool Grammar::func()
{
	/**
	 * Production:	<func> --> <func decl> <func prime>
	*/
	/*std::cout << "func" << std::endl;*/
	Function *newFunction = NULL;
	return funcDecl(&newFunction) && funcPrime(newFunction);
}

bool Grammar::funcPrime(Function *newFunction)
{
	/**
	 * Production:	<func prime> -->semicolon 
	 *								| left_brace <data decls> <statements> right_brace 
	 */
	/*std::cout << "funcPrime" << std::endl;*/
	if (parse->curToken() && parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
	{
		*outFile << *newFunction;
		delete newFunction;
		parse->nextToken();
		return true;
	}
	else if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACE)
	{
		parse->getSymbolTable()->curFunction = newFunction;
		if(parse->nextToken()
			&& dataDecls()
			&& statements()
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_BRACE)
		{
			newFunction->isDeclaration = false;
			*outFile << *newFunction;
			parse->getSymbolTable()->functionQueue.push(newFunction);
			parse->getSymbolTable()->curFunction = NULL;
			parse->incrementFunctionCount();
			parse->nextToken();
			return true;
		}
	}

	/*std::cout << "funcPrime [false]";*/
	return false;
}

bool Grammar::funcDecl(Function **newFunction)
{
	/**
	 * Production:	<func decl> --> <typename> ID <func decl prime>
	 */
	/*std::cout << "funcDecl" << std::endl;*/
	std::string datatype;
	if (typeName(datatype)
		&& parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
	{
		*newFunction = new Function(datatype, parse->curToken()->getTokenName());
		parse->nextToken();
		return funcDeclPrime(*newFunction);
	}
	/*std::cout << "funcDecl [false]";*/
	return false;
}

bool Grammar::funcDeclPrime(Function *newFunction)
{
	/**
	 * Production:	<func decl prime> --> left_parenthesis <parameter list> right_parenthesis 
	 */
	/*std::cout << "funcDeclPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
		&& parse->nextToken()
		&& paramList(newFunction->arguments)
		&& parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS)
	{
		parse->nextToken();
		return true;
	}
	/*std::cout << "funcDeclPrime [false]";*/
	return false;
}

bool Grammar::typeName(std::string &datatype)
{
	/**
	* Production:	<type name> --> int 
	*								| void 
	*								| binary 
	*								| decimal 
	*/
	/*std::cout << "typeName" << std::endl;*/
	if (parse->curToken()
		&& (!strcmp(parse->curToken()->getTokenName().c_str(), "int")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "void")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "binary")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "decimal")))
	{
		datatype.append(parse->curToken()->getTokenName());
		parse->nextToken();
		return true;
	}
	/*std::cout << "typeName [false]";*/
	return false;
}

bool Grammar::typeNamePrime(std::string &datatype)
{
	/**
	* Production:	<type name prime> --> int
	*									| binary
	*									| decimal
	*/
	/*std::cout << "typeNamePrime" << std::endl;*/
	if (parse->curToken()
		&& (!strcmp(parse->curToken()->getTokenName().c_str(), "int")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "binary")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "decimal")))
	{
		datatype.append(parse->curToken()->getTokenName());
		parse->nextToken();
		return true;
	}
	/*std::cout << "typeNamePrime [false]";*/
	return false;
}

bool Grammar::paramList(std::vector< std::pair<std::string, Variable*> > &arguments)
{
	/**
	* Production:	<parameter list> --> void <parameter list prime>
	*									| <typename prime> ID <non-empty list prime> 
	*									| empty
	*/
	/*std::cout << "paramList" << std::endl;*/
	std::string datatype;
	if (parse->curToken()
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "void"))
	{
		parse->nextToken();
		return paramListPrime(arguments);
	}
	else if (typeNamePrime(datatype))
	{
		if (parse->curToken()
			&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
		{
			arguments.push_back(std::pair<std::string, Variable*>(parse->curToken()->getTokenName(), new Variable(datatype, parse->curToken()->getTokenName())));
			parse->nextToken();
			return nonEmptyListPrime(arguments);
		}
		return false;
	}
	return true;
}

bool Grammar::paramListPrime(std::vector< std::pair<std::string, Variable*> > &arguments)
{
	/**
	* Production:	<parameter list prime> --> ID <non-empty list prime> 
	*											| empty
	*/
	/*std::cout << "paramListPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
	{
		arguments.push_back(std::pair<std::string, Variable*>(parse->curToken()->getTokenName(), new Variable("void", parse->curToken()->getTokenName())));
		parse->nextToken();
		return nonEmptyListPrime(arguments);
	}
	/*std::cout << "paramListPrime [false]";*/
	return true;
}

bool Grammar::nonEmptyListPrime(std::vector< std::pair<std::string, Variable*> > &arguments)
{
	/**
	* Production:	<non-empty list prime> --> comma <type name> ID <non-empty list prime>
    *  											| empty
	*/
	/*std::cout << "nonEmptyListPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_COMMA)
	{
		std::string datatype;
		if (parse->nextToken()
			&& typeName(datatype)
			&& parse->curToken()
			&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
		{
			arguments.push_back(std::pair<std::string, Variable*>(parse->curToken()->getTokenName(), new Variable(datatype, parse->curToken()->getTokenName())));
			parse->nextToken();
			return nonEmptyListPrime(arguments);
		}
		return false;
	}
	return true;
}

bool Grammar::dataDecls()
{
	/**
	* Production:	<data decls> --> empty 
	*								| <type name> <id list> semicolon <data decls>
	*/
	/*std::cout << "dataDecls" << std::endl;*/
	std::string datatype;
	if (typeName(datatype))
	{
		if (idList(datatype)
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
		{
			parse->nextToken();
			return dataDecls();
		}
		/*std::cout << "dataDecls [false]";*/
		return false;
	}
	return true;
}

bool Grammar::dataDeclsPrime(std::string dataType, std::string idName)
{
	/**
	* Production:	<data decls prime> -> <id prime> <id list prime> semicolon <data decls prime prime> | empty
	*/
	/*std::cout << "dataDeclsPrime" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_COMMA
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET
		|| parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON))
	{
		if (idPrime(dataType, idName)
			&& idListPrime(dataType)
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
		{
			parse->nextToken();
			return dataDeclsDPrime();
		}
		else
		{
			parse->decrementVariableCount();
			/*std::cout << "dataDeclsPrime [false]";*/
			return false;
		}
	}
	return true;
}

bool Grammar::dataDeclsDPrime()
{
	/**
	* Production:	<data decls prime prime> --> <typename> ID <dataFuncDecl> | empty
	*/
	/*std::cout << "dataDeclsDPrime" << std::endl;*/
	std::string dataType;
	if (typeName(dataType))
	{
		if (parse->curToken()
			&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
		{
			std::string idName(parse->curToken()->getTokenName());
			parse->nextToken();
			return dataFuncDecl(dataType, idName);
		}
		/*std::cout << "dataDeclsDPrime [false]";*/
		return false;
	}
	std::string globalDecl("int global[");
	globalDecl.append(Util::to_string(parse->getSymbolTable()->globalVarCnt));
	globalDecl.append("];");
	*outFile << globalDecl << std::endl;
	return true;
}

bool Grammar::dataFuncDecl(std::string dataType, std::string idName)
{
	/**
	* Production:	<dataFuncDecl> --> <data decls prime> | <func list prime>
	*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_COMMA
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET
		|| parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON))
	{
		return dataDeclsPrime(dataType, idName);
	}
	else
	{
		std::string globalDecl("int global[");
		globalDecl.append(Util::to_string(parse->getSymbolTable()->globalVarCnt));
		globalDecl.append("];");
		*outFile << globalDecl << std::endl;
		return funcListPrime(dataType, idName);
	}
}

bool Grammar::idList(std::string dataType)
{
	/**
	* Production:	<id list> --> <id> <id list prime>
	*/
	/*std::cout << "idList" << std::endl;*/
	return id(dataType) && idListPrime(dataType);
}

bool Grammar::idListPrime(std::string dataType)
{
	/**
	* Production:	<id list prime> --> comma <id> <id list prime>
    *									| empty
	*/
	/*std::cout << "idListPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_COMMA)
	{
		parse->nextToken();
		return id(dataType) && idListPrime(dataType);
	}
	return true;
}

bool Grammar::id(std::string dataType)
{
	/**
	* Production:	<id> --> ID <id prime>
	*/
	/*std::cout << "id" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
	{
		std::string idName(parse->curToken()->getTokenName());
		parse->nextToken();
		return idPrime(dataType, idName);
	}
	/*std::cout << "id [false]";*/
	return false;
}

bool Grammar::idPrime(std::string dataType, std::string idName)
{
	/**
	* Production:	<id prime> --> left_bracket <expression> right_bracket 
	*								| empty
	*/
	/*std::cout << "idPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET)
	{
		parse->nextToken();
		ASTNode *root = NULL;
		if (expression(&root)
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_BRACKET)
		{
			parse->evaluateASTTree(root, false);
			/*Updating local/global array variable details in the symbol table*/
			if (parse->getSymbolTable()->curFunction != NULL)
			{
				parse->getSymbolTable()->curFunction->localVariables.push_back(std::pair<std::string, Variable*>(idName, new Variable(dataType, idName, parse->getSymbolTable()->curFunction->localVarCnt, Variable::ARRAY, std::atoi(root->value.c_str()))));
				parse->getSymbolTable()->curFunction->localVarCnt += std::atoi(root->value.c_str());
			}
			else
			{
				parse->getSymbolTable()->globalVariables.push_back(std::pair<std::string, Variable*>(idName, new Variable(dataType, idName, parse->getSymbolTable()->globalVarCnt, Variable::ARRAY, std::atoi(root->value.c_str()))));
				parse->getSymbolTable()->globalVarCnt += std::atoi(root->value.c_str());
			}
			parse->nextToken();
			parse->incrementVariableCount();
			delete root;
			return true;
		}
		else
		{
			/*std::cout << "idPrime [false]";*/
			return false;
		}
	}

	/*Updating local/global variable details in the symbol table*/
	if (parse->getSymbolTable()->curFunction != NULL)
	{
		parse->getSymbolTable()->curFunction->localVariables.push_back(std::pair<std::string, Variable*>(idName, new Variable(dataType, idName, parse->getSymbolTable()->curFunction->localVarCnt, Variable::SCALAR, 1)));
		parse->getSymbolTable()->curFunction->localVarCnt++;
	}
	else
	{
		parse->getSymbolTable()->globalVariables.push_back(std::pair<std::string, Variable*>(idName, new Variable(dataType, idName, parse->getSymbolTable()->globalVarCnt, Variable::SCALAR, 1)));
		parse->getSymbolTable()->globalVarCnt++;
	}

	parse->incrementVariableCount();
	return true;
}

bool Grammar::idUse(std::string idName, std::string &assignStmt)
{
	/**
	* Production:	<id use> --> left_bracket <expression> right_bracket
	*								| empty
	*/
	/*std::cout << "idPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET)
	{
		parse->nextToken();
		ASTNode *root = NULL;
		if (expression(&root)
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_BRACKET)
		{
			parse->evaluateASTTree(root, false);
			assignStmt.append(Util::getOperandString(idName, parse->getSymbolTable(), &(root->value)));
			parse->nextToken();
			delete root;
			return true;
		}
		else
		{
			/*std::cout << "idPrime [false]";*/
			return false;
		}
	}

	/*Deference variable name and store it int he assignment statement*/
	assignStmt.append(Util::getOperandString(idName, parse->getSymbolTable()));

	return true;
}

bool Grammar::blockStatements()
{
	/**
	* Production:	<block statements> --> left_brace <statements> right_brace 
	*/
	/*std::cout << "blockStatement" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACE
		&& parse->nextToken()
		&& statements()
		&& parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_BRACE)
	{
		parse->nextToken();
		return true;
	}
	/*std::cout << "blockStatements [false]";*/
	return false;
}

bool Grammar::statements()
{
	/**
	* Production:	<statements> --> <statement> <statements> 
  	*								| empty
	*/
	/*std::cout << "statements" << std::endl;*/

	if (parse->curToken() 
		&& (parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER
		|| (parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD
		&& (!strcmp(parse->curToken()->getTokenName().c_str(), "if")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "while")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "return")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "break")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "continue")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "read")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "write")
		|| !strcmp(parse->curToken()->getTokenName().c_str(), "print")))))
	{
		if (statement())
		{
			parse->incrementStmtCount();
			return statements();
		}
		else
		{
			return false;
		}
	}
	return true;
}

bool Grammar::statement()
{
	/**
	* Production:	<statement> --> ID <statement prime> 
	*								| <if statement> 
	*								| <while statement> 
	*								| <return statement> 
	*								| <break statement> 
	*								| <continue statement> 
	*								| read left_parenthesis  ID right_parenthesis semicolon 
	*								| write left_parenthesis <expression> right_parenthesis semicolon 
	*								| print left_parenthesis  STRING right_parenthesis semicolon 
	*/
	/*std::cout << "statement" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
	{
		std::string idName(parse->curToken()->getTokenName());
		parse->nextToken();
		return statementPrime(idName);
	}
	else if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD)
	{
		if (!strcmp(parse->curToken()->getTokenName().c_str(), "if"))
		{
			return ifStatement();
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "while"))
		{
			return whileStatment();
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "return"))
		{
			return returnStatement();
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "break"))
		{
			return breakStatement();
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "continue"))
		{
			return continueStatement();
		}
		if(!strcmp(parse->curToken()->getTokenName().c_str(), "read"))
		{
			std::string readStmt;
			readStmt.append("read(");
			if (parse->nextToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
				&& parse->nextToken()
				&& parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
			{
				readStmt.append(Util::getOperandString(parse->curToken()->getTokenName(), parse->getSymbolTable()));
				if (parse->nextToken()
					&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS
					&& parse->nextToken()
					&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
				{
					readStmt.append(");");
					parse->getSymbolTable()->curFunction->funcStats.push(readStmt);
					parse->nextToken();
					return true;
				}
			}
			return false;
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "write"))
		{
			std::string writeStmt;
			ASTNode *root = NULL;
			writeStmt.append("write(");
			if (parse->nextToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
				&& parse->nextToken()
				&& expression(&root)
				&& parse->curToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS
				&& parse->nextToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
			{
				parse->evaluateASTTree(root, false);
				writeStmt.append(root->value);
				writeStmt.append(");");
				parse->getSymbolTable()->curFunction->funcStats.push(writeStmt);
				parse->nextToken();
				delete root;
				return true;
			}
			return false;
		}
		else if (!strcmp(parse->curToken()->getTokenName().c_str(), "print"))
		{
			std::string printStmt;
			printStmt.append("print(");
			if (parse->nextToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
				&& parse->nextToken()
				&& parse->curToken()->getID() == Token::IDTYPE_STRING)
			{
				printStmt.append(parse->curToken()->getTokenName());
				if (parse->nextToken()
					&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS
					&& parse->nextToken()
					&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
				{
					printStmt.append(");");
					parse->getSymbolTable()->curFunction->funcStats.push(printStmt);
					parse->nextToken();
					return true;
				}
			}
			return false;
		}
	}
	/*std::cout << "statement [false]";*/
	return false;
}

bool Grammar::statementPrime(std::string idName)
{
	/**
	* Production:	<statement prime> --> <assignment> 
	*									| <func call>
	*/
	/*std::cout << "statementPrime" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET
		|| parse->curToken()->getSymType() == Token::SYMTYPE_EQUAL))
	{
		return assignment(idName);
	}
	else if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS)
	{
		return funcCall(idName);
	}
	/*std::cout << "statementPrime [false]";*/
	return false;
}

bool Grammar::assignment(std::string idName)
{
	/**
	* Production:	<assignment> --> <id use> equal_sign <expression> semicolon 
	*/
	/*std::cout << "assignment" << std::endl;*/
	//TODO: Add Code Here
}

bool Grammar::funcCall(std::string idName)
{
	/**
	* Production:	<func call> --> left_parenthesis <expr list> right_parenthesis semicolon 
	*/
	/*std::cout << "funcCall" << std::endl;*/
	std::string funcCallStmt;
	std::queue<std::string> arguments;
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
		&& parse->nextToken()
		&& exprList(arguments)
		&& parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS
		&& parse->nextToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
	{
		Util::createFuncCallStatement(idName, arguments, funcCallStmt);
		funcCallStmt.push_back(';');
		parse->getSymbolTable()->curFunction->funcStats.push(funcCallStmt);
		parse->nextToken();
		return true;
	}
	/*std::cout << "funcCall [false]";*/
	return false;
}

bool Grammar::exprList(std::queue<std::string> &arguments)
{
	/**
	* Production:	<expr list> --> <non-empty expr list> 
	*  								| empty
	*/
	/*std::cout << "exprList" << std::endl;*/
	if (parse->curToken()
		&&(parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER
		|| parse->curToken()->getID() == Token::IDTYPE_NUMBER
		|| parse->curToken()->getSymType() == Token::SYMTYPE_MINUS
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS))
	{
		return nonEmptyExprList(arguments);
	}
	return true;
}

bool Grammar::nonEmptyExprList(std::queue<std::string> &arguments)
{
	/**
	* Production:	<non-empty expr list> --> <expression> <non-empty expr list prime> 
	*/
	/*std::cout << "nonEmptyExprList" << std::endl;*/
	ASTNode *root = NULL;
	if (expression(&root))
	{
		parse->evaluateASTTree(root, false, false);
		arguments.push(root->value);
		return nonEmptyExprListPrime(arguments);
	}
	return false;
}

bool Grammar::nonEmptyExprListPrime(std::queue<std::string> &arguments)
{
	/**
	* Production:	<non-empty expr list prime> --> comma <expression> <non-empty expr list prime>
	*												| empty
	*/
	/*std::cout << "nonEmptyExprListPrime" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_COMMA)
	{ 
		parse->nextToken();
		ASTNode *root = NULL;
		if (expression(&root))
		{
			parse->evaluateASTTree(root, false, false);
			arguments.push(root->value);
			delete root;
			return nonEmptyExprListPrime(arguments);
		}
		return false;
	}
	return true;
}

bool operand(ASTNode **root) {
	if(parse->curToken()) {
		//NUMBER
		if (parse->curToken()->getID() == Token::IDTYPE_NUMBER)
		{
			*root = new ASTNode(parse->curToken()->getTokenName(), ASTNode::NUMBER);
			parse->nextToken();
			return true;
		}
		//minus_sign NUMBER
		else if (parse->curToken()->getSymType() == Token::SYMTYPE_MINUS)
		{
			if (parse->nextToken()
				&& parse->curToken()->getID() == Token::IDTYPE_NUMBER)
			{
				std::string number;
				number.push_back('-');
				number.append(parse->curToken()->getTokenName());
				*root = new ASTNode(number, ASTNode::NUMBER);
				parse->nextToken();
				return true;
			}
			//no token after minus_sign or if present, not a NUMBER
			return false;
		}
		
	}
	//no current token
	return false;
}

bool Grammar::ifStatement()
{
	/**
	* Production:	<if statement> --> if left_parenthesis <condition expression> right_parenthesis <block statements> 
	*/
	/*std::cout << "ifStatement" << std::endl;*/
	//TODO: Add Code Here
	ASTNode *root = NULL;
	//check till: if left_paranthesis(including next token)
	if(parse->curToken() 
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD 
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "if") 
		&& parse->nextToken()  
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
		&& parse->nextToken() 
		&&operand(&root) 
		&& parse->curToken() 
		&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS)
	{
		//rest of it later...	
	}
		
}

bool Grammar::conditionExpression(ASTNode **root)
{
	/**
	* Production:	<condition expression> -->  <condition> <condition expression prime>
	*/
	/*std::cout << "conditionExpression" << std::endl;*/
	return condition(root) && conditionExpressionPrime(root);
}

bool Grammar::conditionExpressionPrime(ASTNode **root)
{
	/**
	* Production:	<condition expression prime> -->  <condition op> <condition> <condition expression prime>
   	*												| empty
	*/
	/*std::cout << "conditionExpressionPrime" << std::endl;*/
	if (conditionOp(root))
	{
		return condition(&((*root)->right)) && conditionExpressionPrime(root);
	}
	return true;
}

bool Grammar::conditionOp(ASTNode **root)
{
	/**
	* Production:	<condition op> --> double_and_sign 
    *								| double_or_sign
	*/
	/*std::cout << "conditionOp" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_DOUBLE_AND
		|| parse->curToken()->getSymType() == Token::SYMTYPE_DOUBLE_OR))
	{
		ASTNode *newNode = new ASTNode(parse->curToken()->getTokenName(), ASTNode::OPERATION);
		newNode->left = *root;
		*root = newNode;
		parse->nextToken();
		return true;
	}
	/*std::cout << "conditionOp [false]";*/
	return false;
}

bool Grammar::condition(ASTNode **root)
{
	/**
	* Production:	<condition> --> <expression> <comparison op> <expression> 
	*/
	/*std::cout << "condition" << std::endl;*/
	return expression(root) && comparisionOp(root) && expression(&((*root)->right));
}

bool Grammar::comparisionOp(ASTNode **root)
{
	/**
	* Production:	<comparison op> --> == 
    *									| != 
    *									| > 
    *									| >= 
    *									| < 
    *									| <=
	*/
	/*std::cout << "comparisionOp" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_DOUBLE_EQUAL
		|| parse->curToken()->getSymType() == Token::SYMTYPE_NT_EQUAL
		|| parse->curToken()->getSymType() == Token::SYMTYPE_GT
		|| parse->curToken()->getSymType() == Token::SYMTYPE_GT_EQUAL
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LT
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LT_EQUAL))
	{
		ASTNode *newRoot = new ASTNode(parse->curToken()->getTokenName(), ASTNode::OPERATION);
		newRoot->left = *root;
		*root = newRoot;
		parse->nextToken();
		return true;
	}
	/*std::cout << "comparisionOp [false]";*/
	return false;
}

bool Grammar::whileStatment()
{
	/**
	* Production:	<while statement> --> while left_parenthesis <condition expression> right_parenthesis <block statements> 
	*/
	/*std::cout << "whileStatment" << std::endl;*/
	ASTNode *root = NULL;
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "while")
		&& parse->nextToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
		&& parse->nextToken()
		&& conditionExpression(&root)
		&& parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS)
	{
		parse->evaluateASTTree(root);
		int loopLable = parse->getSymbolTable()->lableCnt++;
		int thenLable = parse->getSymbolTable()->lableCnt++;
		int endLable = parse->getSymbolTable()->lableCnt++;
		
		std::string loopLableStmt("c");
		loopLableStmt.append(Util::to_string(loopLable));
		loopLableStmt.append(":;");
		parse->getSymbolTable()->curFunction->funcStats.push(loopLableStmt);

		Util::printIfStmt(parse->getSymbolTable(), root, thenLable, endLable);

		delete root;
		parse->getSymbolTable()->curFunction->whileLables.push(std::pair<int, int>(loopLable, endLable));
		parse->nextToken();
		if (blockStatements())
		{
			std::string loopGotoStmt("goto c");
			loopGotoStmt.append(Util::to_string(loopLable));
			loopGotoStmt.push_back(';');

			std::string endLableStmt("c");
			endLableStmt.append(Util::to_string(endLable));
			endLableStmt.append(":;");

			parse->getSymbolTable()->curFunction->funcStats.push(loopGotoStmt);
			parse->getSymbolTable()->curFunction->funcStats.push(endLableStmt);
			parse->getSymbolTable()->curFunction->whileLables.pop();
			return true;
		}
		parse->getSymbolTable()->curFunction->whileLables.pop();
	}
	/*std::cout << "whileStatment [false]";*/
	return false;
}

bool Grammar::returnStatement()
{
	/**
	* Production:	<return statement> --> return <return statement prime>
	*/
	/*std::cout << "returnStatement" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "return"))
	{
		parse->nextToken();
		return returnStatementPrime();
	}
	/*std::cout << "returnStatement [false]";*/
	return false;
}

bool Grammar::returnStatementPrime()
{
	/**
	* Production:	<return statement prime> --> <expression> semicolon 
    *											| semicolon 
	*/
	/*std::cout << "returnStatementPrime" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER
		|| parse->curToken()->getID() == Token::IDTYPE_NUMBER
		|| parse->curToken()->getSymType() == Token::SYMTYPE_MINUS
		|| parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS))
	{
		ASTNode *root = NULL;
			if (expression(&root)
				&& parse->curToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
			{
				std::string retStmt("return ");
				parse->evaluateASTTree(root, false, false);
				retStmt.append(root->value);
				retStmt.push_back(';');
				parse->getSymbolTable()->curFunction->funcStats.push(retStmt);
				delete root;
				parse->nextToken();
				return true;
			}
			return false;
	}
	else if (parse->curToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
	{
		std::string retStmt("return;");
		parse->getSymbolTable()->curFunction->funcStats.push(retStmt);
		parse->nextToken();
		return true;
	}
	/*std::cout << "returnStatementPrime [false]";*/
	return false;
}

bool Grammar::breakStatement()
{
	/**
	* Production:	<break statement> ---> break semicolon 
	*/
	/*std::cout << "breakStatement" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "break")
		&& parse->nextToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
	{
		std::string retStmt("goto c");
		retStmt.append(Util::to_string(parse->getSymbolTable()->curFunction->whileLables.top().second));
		retStmt.push_back(';');
		parse->getSymbolTable()->curFunction->funcStats.push(retStmt);

		parse->nextToken();
		return true;
	}
	/*std::cout << "breakStatement [false]";*/
	return false;
}

bool Grammar::continueStatement()
{
	/**
	* Production:	<continue statement> ---> continue semicolon
	*/
	/*std::cout << "continueStatement" << std::endl;*/
	if (parse->curToken()
		&& parse->curToken()->getID() == Token::IDTYPE_RESERVEDWORD
		&& !strcmp(parse->curToken()->getTokenName().c_str(), "continue")
		&& parse->nextToken()
		&& parse->curToken()->getSymType() == Token::SYMTYPE_SEMICOLON)
	{
		std::string retStmt("goto c");
		retStmt.append(Util::to_string(parse->getSymbolTable()->curFunction->whileLables.top().first));
		retStmt.push_back(';');
		parse->getSymbolTable()->curFunction->funcStats.push(retStmt);

		parse->nextToken();
		return true;
	}
	/*std::cout << "continueStatement [false]";*/
	return false;
}

bool Grammar::expression(ASTNode **root)
{
	/**
	* Production:	<expression> --> <term> <expression prime> 
	*/
	/*std::cout << "expression" << std::endl;*/
	return term(root) && expressionPrime(root);
}

bool Grammar::expressionPrime(ASTNode **root)
{
	/**
	* Production:	<expression prime> --> <addop> <term> <expression prime>
	*				| empty
	*/
	/*std::cout << "expressionPrime" << std::endl;*/
	if (addOp(root))
	{
		return term(&((*root)->right)) && expressionPrime(root);
	}
	return true;
}

bool Grammar::addOp(ASTNode **root)
{
	/**
	* Production:	<addop> --> plus_sign 
    *							| minus_sign 
	*/
	/*std::cout << "addOp" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_PLUS
		|| parse->curToken()->getSymType() == Token::SYMTYPE_MINUS))
	{
		ASTNode *newRoot = new ASTNode(parse->curToken()->getTokenName(), ASTNode::OPERATION);
		newRoot->left = *root;
		*root = newRoot;
		parse->nextToken();
		return true;
	}
	/*std::cout << "addOp [false]";*/
	return false;
}

bool Grammar::term(ASTNode **root)
{
	/**
	* Production:	<term> --> <factor> <term prime> 
	*/
	/*std::cout << "term" << std::endl;*/
	return factor(root) && termPrime(root);
}

bool Grammar::termPrime(ASTNode **root)
{
	/**
	* Production:	<term> --> <mulop> <factor> <term prime> | empty
	*/
	/*std::cout << "termPrime" << std::endl;*/
	if (mulOp(root))
	{
		return factor(&((*root)->right)) && termPrime(root);
	}
	return true;
}

bool Grammar::mulOp(ASTNode **root)
{
	/**
	* Production:	<mulop> --> star_sign 
    *							| forward_slash
	*/
	/*std::cout << "mulOp" << std::endl;*/
	if (parse->curToken()
		&& (parse->curToken()->getSymType() == Token::SYMTYPE_STAR
		|| parse->curToken()->getSymType() == Token::SYMTYPE_FORWARD_SLASH))
	{
		ASTNode *newRoot = new ASTNode(parse->curToken()->getTokenName(), ASTNode::OPERATION);
		newRoot->left = *root;
		*root = newRoot;
		parse->nextToken();
		return true;
	}
	/*std::cout << "mulOp [false]";*/
	return false;
}

bool Grammar::factor(ASTNode **root)
{
	/**
	* Production:	<factor> --> ID <factor prime> 
    *							| NUMBER 
    *							| minus_sign NUMBER 
    *							| left_parenthesis<expression> right_parenthesis
	*/
	/*std::cout << "factor" << std::endl;*/
	if (parse->curToken())
	{
		if (parse->curToken()->getID() == Token::IDTYPE_IDENTIFIER)
		{
			std::string idName = parse->curToken()->getTokenName();
			parse->nextToken();
			return factorPrime(idName, root);
		}
		else if (parse->curToken()->getID() == Token::IDTYPE_NUMBER)
		{
			*root = new ASTNode(parse->curToken()->getTokenName(), ASTNode::NUMBER);
			parse->nextToken();
			return true;
		}
		else if (parse->curToken()->getSymType() == Token::SYMTYPE_MINUS)
		{
			if (parse->nextToken()
				&& parse->curToken()->getID() == Token::IDTYPE_NUMBER)
			{
				std::string number;
				number.push_back('-');
				number.append(parse->curToken()->getTokenName());
				*root = new ASTNode(number, ASTNode::NUMBER);
				parse->nextToken();
				return true;
			}
			return false;
		}
		else if (parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS
			&& parse->nextToken()
			&& expression(root)
			&& parse->curToken()
			&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS)
		{
			parse->nextToken();
			return true;
		}
	}
	/*std::cout << "factor [false]";*/
	return false;
}

bool Grammar::factorPrime(std::string idName, ASTNode **root)
{
	/**
	* Production:	<factor prime> --> left_bracket <expression> right_bracket 
    *									| left_parenthesis <expr list> right_parenthesis 
    *									| empty
	*/
	/*std::cout << "factorPrime" << std::endl;*/

		if (parse->curToken() && parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_BRACKET)
		{
			parse->nextToken();
			ASTNode *aryPos = NULL;
			if (expression(&aryPos)
				&& parse->curToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_BRACKET)
			{
				parse->evaluateASTTree(aryPos, false);
				*root = new ASTNode(Util::getOperandString(idName, parse->getSymbolTable(), &(aryPos->value)), ASTNode::OPERAND);
				delete aryPos;
				parse->nextToken();
				return true;
			}
			return false;
		}
		else if (parse->curToken() && parse->curToken()->getSymType() == Token::SYMTYPE_LEFT_PARENTHESIS)
		{
			parse->nextToken();
			std::queue<std::string> arguments;
			if (exprList(arguments)
				&& parse->curToken()
				&& parse->curToken()->getSymType() == Token::SYMTYPE_RIGHT_PARENTHESIS)
			{
				std::string funcCallStmt;
				Util::createFuncCallStatement(idName, arguments, funcCallStmt);
				*root = new ASTNode(funcCallStmt, ASTNode::FUNCCALL);
				parse->nextToken();
				return true;
			}
			return false;
		}
		else
		{
			*root = new ASTNode(Util::getOperandString(idName, parse->getSymbolTable()), ASTNode::OPERAND);
			return true;
		}
}

void Grammar::instanciateParser(Parser* newParser)
{
	parse = newParser;
}
