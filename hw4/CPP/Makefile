# Our image is missing the link between the installed g++ and command requiring us to call the command by its full 
# name g++-4.8 other systems use just the g++ part without the version number

cc=/usr/bin/g++-4.8
all: Main Scanner Token Util Parser Grammar
	$(cc) -o Main Main.o Scanner.o Token.o Util.o Parser.o Grammar.o
Main: Main.cpp
	$(cc) -c Main.cpp
Scanner: Scanner.cpp
	$(cc) -c Scanner.cpp
Token: Token.cpp
	$(cc) -c Token.cpp
Util: Util.cpp
	$(cc) -c Util.cpp
Parser: Parser.cpp
	$(cc) -c Parser.cpp
Grammar: Grammar.cpp
	$(cc) -c Grammar.cpp
clean:
	-rm -f Main *.o
