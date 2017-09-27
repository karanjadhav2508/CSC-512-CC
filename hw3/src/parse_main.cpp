#include "parser.h"
#include "parse_main.h"
#include "scanner.h"
#include "token.h"

#include <fstream>
#include <iostream>
#include <vector>

auto main( int argc, char **argv ) -> int {

  //-----------------------------------------------------------------------------
  // Expect exactly two arguments, this program's name and the input text file.
  //-----------------------------------------------------------------------------

  if( argc != 2 ) {
    std::cout << "Error:  Expecting input file name as the sole argument." << std::endl;
    return(1);
  }

  std::string input_filename( argv[1] );
  string_vector program_text;

  //-----------------------------------------------------------------------------
  // Read the input file into a string_vector called program_text.  File open
  // errors will be reported inside the function.
  //-----------------------------------------------------------------------------
    
  if( !read_file_to_string_vector( input_filename, program_text ) ) {
    return(1);
  }

  //-----------------------------------------------------------------------------
  // Instantiate the scanner with the input text and tokenize it. Exit on error.
  //-----------------------------------------------------------------------------
  
  Scanner scanner( program_text );

  std::string error_message;
  if( !scanner.tokenize(error_message) ) {
    if( error_message.length() > 0 ) {
      std::cout << "error :  " << error_message << std::endl;
    }
    return(2);    
  }


  //-----------------------------------------------------------------------------
  // The scanner gets pushed into the parser.  The parser controls the main loop
  // other the scanner to analyze the text.
  //-----------------------------------------------------------------------------

  Parser parser;
  
  bool pass = parser.parse( scanner );

  if( pass ) {
    std::cout << "pass "
	      << "variable " << parser.get_variable_count() << " "
	      << "function " << parser.get_function_count() << " "
	      << "statement " << parser.get_statement_count() << std::endl;
  } else {
    std::cout << "error : parser error" << std::endl;
  }

  
  return(0);
}

//-----------------------------------------------------------------------------
// Open a file stored in the string 'filename' for reading.  Store the
// result in a string_vector as the program text.  Report I/O failures
// and return false if one happens.
//-----------------------------------------------------------------------------

bool read_file_to_string_vector( const std::string & filename,
				 string_vector & file_text ) {

  std::fstream input_file;
  input_file.open( filename, std::ios::in );
  
  if( input_file.is_open() ) {
    
    std::string line;
    
    while( std::getline( input_file, line ) ) {
      file_text.push_back(line);
    }
    
    input_file.close();

    return(true);
    
  } else {

    std::cout << "Failed to open file '" << filename << "'." << std::endl;
    return(false);
    
  }
    
}


