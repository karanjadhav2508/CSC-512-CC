#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "Scanner.h"
#include "Parser.h"

void genOutFName(char *input, std::string &output);

int main(int argc, char* argv[])
{
	if(argc != 2)
	{
		std::cout << "Invalid number of arguments." << std::endl << "Format:\t./Main <InputFileName>" << std::endl;
		return EXIT_FAILURE;
	}

	Scanner scan(argv[1]);

	std::string outFileName;
	genOutFName(argv[1], outFileName);
	std::ofstream outFile(outFileName.c_str(), std::ifstream::out);
	
	Grammar *cGrammar = new Grammar(&outFile);
	SymbolTable *newSymbolTable = new SymbolTable();
	Parser parse(cGrammar, newSymbolTable);

	while (scan.hasNextToken())
	{
		Token *t = scan.getNextToken();
		if (t->getID() == Token::IDTYPE_INVALID)
		{
			delete t;
			std::cout << "Error" << std::endl;
			return EXIT_FAILURE;
		}
		parse.addToken(t);
	}

	if (!parse.parseProgram())
	{
		std::cout << "Error" << std::endl;
		return EXIT_FAILURE;
	}
}

void genOutFName(char *input, std::string &output)
{
	bool printed = false;
	while (*input != '\0')
	{
		if (*input == '.' && !printed)
		{
			output.push_back('_');
			output.push_back('g');
			output.push_back('e');
			output.push_back('n');
			output.push_back('.');
			printed = true;
			/*std::cout << " I [" << input << "] output [" << output << "]\n";*/
		}
		else
		{
			output.push_back(*input);
			/*std::cout << " I [" << input << "] output [" << output << "]\n";*/
		}
		input++;
	}
}

std::ofstream& operator<<(std::ofstream& outFile, Function& printFunc)
{
	outFile << std::endl << printFunc.datatype << " " << printFunc.name << " (";

	if (printFunc.arguments.size() == 0)
	{
		outFile << "void";
	}
	else
	{
		for (std::vector< std::pair<std::string, Variable *> >::iterator it = printFunc.arguments.begin(); it != printFunc.arguments.end();)
		{
			outFile << it->second->datatype << " " << it->second->name;
			if (++it != printFunc.arguments.end())
			{
				outFile << ", ";
			}
		}
	}
	outFile << ")";

	if (printFunc.isDeclaration)
	{
		outFile << ";" << std::endl;
	}
	else
	{
		outFile << std::endl << "{" << std::endl;
		if (printFunc.localVarCnt != 0)
		{
			outFile << "\t" << "int local[" << printFunc.localVarCnt << "];" << std::endl;
		}

		while (!printFunc.funcStats.empty())
		{
			outFile << "\t" << printFunc.funcStats.front() << std::endl;
			printFunc.funcStats.pop();
		}
		outFile << "}" << std::endl;
	}
	return outFile;
}

