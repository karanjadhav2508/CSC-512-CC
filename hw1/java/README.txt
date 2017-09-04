ReadME
by Danny Reinheimer 


Compile Instructions:
To compiler the scanner run the following java command 
javac Pair.java RegularExpressions.java Scan.java Scanner.java Token.java TokenNames.java

to execute the scanner run the following command in the terminal:
java Scanner <inputFileName>

An example of the scanner execution is:
java Scanner foo.c

The output will be the generated code with the identifiers starting with cs512
the output file name will be the input file name appended with _gen.c so foo.c will be foo_gen.c

The implementation for this scanner is as follows:

I created my own set of regular expressions for each of the different types of tokens.  I take in a file and read it one character at a time and I see if that character matches any regular expressions and if so i get the coresponding token.  I keep building a larger string and matching tokens until there is no tokens that match and at that point I take the last matched token and the string minus the last character and return the token value pair to be writen to the new generated file.  If we reach the end of the line we check to see if we have a token that includes the end of line character and if not throw an error.  If we subtracted the last token then we start are next token search with the subtracted character.  

When printing the output we check to see if the token is an identifier.  If it is
and the string does not equal main then we add the string cs512 to the beginning of the identifer.

One thing I had to do was manully check to see if the input string we were processing contained the value // since the code would assume it is two symbol's with value / and
not a metaStatement.  

I also check to see if we have reached the end of the file by looking for the character value /u001a.  this tell me that I should take the found token before it and return the value of the string minus the end of file value.

The only unique data type used in my code is a Pair data type, which is just a key-value pair allowing me to return to value in one return call.
