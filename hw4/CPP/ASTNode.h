#ifndef _AST_NODE
#define _AST_NODE

#include <string>

class ASTNode
{
public:
	std::string value;
	int type;

	enum
	{
		OPERAND,
		NUMBER,
		OPERATION,
		FUNCCALL
	};
	ASTNode *left;
	ASTNode *right;

	ASTNode(std::string newValue
		,int newType)
		: value(newValue)
		, type(newType)
	{
		left = NULL;
		right = NULL;
	}

	~ASTNode()
	{
		if (left != NULL)
		{
			delete left;
		}
		if (right != NULL)
		{
			delete right;
		}
	}
};
#endif