How to compile the program?
Use make command to compile the program.
=====================================================================================================================================================================================

How to run the program?
Run the executable named Main with the input filename as an argument.
Eg: ./Main <input file name>
=====================================================================================================================================================================================

Output:
Generates the output file as <input file name with out .c>_gen.c
Prints Error if the input grammar is not accpeted by the Scanner or Parser.
=====================================================================================================================================================================================

Description:
------------
Main.cpp:
Main.cpp is the interface between the user and the scanner, parser program. It makes use of the scanner API to identify each token and push it into the parser list of tokens. Parser API is used to parse the given input program using the grammar API. The grammar API provides interface for the non-terminals in the grammar, these APIs will validate the program using the terminals.

Token.cpp:
Token.cpp contains the class for a token. It holds the properties of a token and provides methods to access the token and populate it. The properties of the token identifies the type of token and if it is a symbol, it contains the symbol type.

Util.cpp:
Util.cpp provides methods to check the type of a charater.

Scanner.cpp:
Scanner.cpp provides methods to check whether there is next token, get the next token, and populate a token. The method to populate a token contains the logic to identify a token and populate the Token objects with the token details.

Parser.cpp:
Parser.cpp reads the list of tokens provided by the scanner. It is a LL(1) parser which will read the current token and decides which production to chose based on the first terminal in a production. 
Parser.cpp provides interface which enables the Grammar to read the current token and to move to the next one. If no token is available in the list, it returns NULL.

Grammar.cpp:
Grammar.cpp has interface for each of the non-terminal in the grammar. It makes use of the Parser.cpp's token interface to read the tokens and decide which production to choose.
The non-terminal interface implements all the productions related to a non-terminal and also implements the code generation logic to generate code which complies with the output program grammar.

ASTNode.h:
Contains the structure an AST node. A node's values can populated using the constructor.

Comparision.h:
Contains a helper class to compare any two keys in a vector of <key, value> pairs.

Function.h:
Contains the structure of a Function. This is use to store a function details and statements in a function.

SymbolTable.h:
Contains the structure of the Symbol Table. Used globally by the grammar to store and retrive the details about the intermediate representation.

Variable.h:
Contains the structure of a varibale. This is use to store a variable details and retrive it back.

Implementation:
1. If a token is an identifier then the token value is compared against the list of reserved words to mark the token as a reserved word. 
2. # and // are treated as symbols and at later stage the character is compared against # or // to mark the token as MetaStatement. 
3. If any statement starts with " then the following characters are considered as a part of a string until another " is reached. If no ending " is found in the same line then the token is marked as invalid.
4. Any token starting with a digit is treated as a number and it is populated until no new digits are found.
5. Any token starting with an alphabet or underscore is treated as an identifier and it is populated until no new alphaber or underscore or digit is found.
6. Any token starting with # or // is treated as MetaStatement and it is populated until the end of line is reached.
7. All non-terminals are implements as methods in the Grammar class.
8. If a terminal is encountered in a production then the current token is validated againts the terminals and it is consumed i.e., the grammar class moves to the next token in the parser's token list. 
9. A production is choosen from the list of productions in non-terminal using the first terminal in each production. 
	If none of the productons' first terminal maches the current token and there is no empty production is avaialble in the non-terminal then the NT's method will return false.
	If none of the productons' first terminal maches the current token and there is an empty production available in the non-terminal then the NT's method will return true since the follow part is checked the calling fuction or calling non-terminal.
10. The first production in the Grammar is program. 
	Program's production will check whether the input program has the token in the format <dataDecls> <func list> or <func list> or empty. If any of the three cases is valid then it returns true.
11. The Grammar APIs for each production implements logic to generate code for each of the terminal encountered. And also build AST tree for each expression or condition expression ecountered. After building a AST tree for an expression, the tree is flattened into a series of output code which complies with the output program grammar.
12. Grammar APIs and Parser APIs makes use of the SymbolTable, Function, Variable, and ASTNode structures to represent the input program in memory.

=====================================================================================================================================================================================
LL(1) Grammar:
--------------

<program> --> <typename> ID <program prime> | empty
<program prime> --> <data decls prime> 
         | <func list prime>
<func list> --> empty 
		| <func> <func list> 
<func list prime> --> <func decl prime> <func prime> <func list>
<func> --> <func decl> <func prime>
<func prime> --> semicolon 
		| left_brace <data decls> <statements> right_brace 
<func decl> --> <type name> ID <func decl prime>
<func decl prime> --> left_parenthesis <parameter list> right_parenthesis 
<type name> --> int | void | binary | decimal 
<type name prime> --> int | binary | decimal 
<parameter list> --> void <parameter list prime>
		     | <typename prime> ID <non-empty list prime> 
		     | empty
<parameter list prime> --> ID <non-empty list prime> 
			   | empty
<non-empty list prime> --> comma <type name> ID <non-empty list prime>
			   | empty
<data decls> --> empty 
		| <type name> <id list> semicolon <data decls>
<data decls prime> -> <id prime> <id list prime> semicolon <data decls prime prime> | empty
<data decls prime prime> --> <typename> ID <dataFuncDecl> | empty
<dataFuncDecl> --> <data decls prime> | <func list prime>
<id list> --> <id> <id list prime>
<id list prime> --> comma <id> <id list prime> 
		    | empty
<id> --> ID <id prime>
<id prime> --> left_bracket <expression> right_bracket
	       | empty
<id use> --> left_bracket <expression> right_bracket
		| empty
<block statements> --> left_brace <statements> right_brace 
<statements> --> <statement> <statements> | empty
<statement> --> ID <statement prime> | <if statement> | <while statement> | <return statement> | <break statement> | <continue statement> | read left_parenthesis  ID right_parenthesis semicolon | write left_parenthesis <expression> right_parenthesis semicolon | print left_parenthesis  STRING right_parenthesis semicolon
<statement prime> --> <assignment> | <func call>
<assignment> --> <id use> equal_sign <expression> semicolon
<func call> --> left_parenthesis <expr list> right_parenthesis semicolon
<expr list> --> <non-empty expr list> | empty
<non-empty expr list> --> <expression> <non-empty expr list prime>
<non-empty expr list prime> --> comma <expression> <non-empty expr list prime>
				| empty
<if statement> --> if left_parenthesis <condition expression> right_parenthesis <block statements> 
<condition expression> -->  <condition> <condition expression prime> 
<condition expression prime> --> <condition op> <condition> <condition expression prime>
				 | empty
<condition op> --> double_and_sign | double_or_sign 
<condition> --> <expression> <comparison op> <expression> 
<comparison op> --> == | != | > | >= | < | <=
<while statement> --> while left_parenthesis <condition expression> right_parenthesis <block statements> 
<return statement> --> return <return statement prime >
<return statement prime> --> <expression> semicolon | semicolon  
<break statement> ---> break semicolon 
<continue statement> ---> continue semicolon

<expression> --> <term> <expression prime> 
<expression prime> --> <addop> <term> <expression prime>
			| empty
<addop> --> plus_sign | minus_sign 
<term> --> <factor> <term prime>
<term prime> --> <mulop> <factor> <term prime>
		 | empty
<mulop> --> star_sign | forward_slash 
<factor> --> ID <factor prime> | NUMBER | minus_sign NUMBER | left_parenthesis <expression> right_parenthesis
<factor prime> --> left_bracket <expression> right_bracket | left_parenthesis <expr list> right_parenthesis | empty
