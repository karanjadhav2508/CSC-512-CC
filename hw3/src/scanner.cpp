#include "scanner.h"
#include "token.h"

#include <iostream>
#include <set>
#include <string>

//-----------------------------------------------------------------------------
// Construct a set with the reserved words for quick lookup.  This declaration
// is static and private for the scanner and is therefore encapsulated.
//-----------------------------------------------------------------------------

const std::set<std::string> Scanner::reserved_words =
  {
    "int",
    "void",
    "if",
    "while",
    "return",
    "read",
    "write",
    "print",
    "continue",
    "break",
    "binary",
    "decimal"
  };

//-----------------------------------------------------------------------------
// Simple symbols which are defined as single character based tokens.  This
// declaration is static and private for the scanner and is therefore
// encapsulated.
//-----------------------------------------------------------------------------

const std::set<char> Scanner::simple_symbols =
  {
    '(',
    ')',
    '{',
    '}',
    '[',
    ']',
    ',',
    ';',
    '+',
    '-',
    '*',
    '/'
  };

//-----------------------------------------------------------------------------
// The standard parameterized constructor.
//-----------------------------------------------------------------------------

Scanner::Scanner( const string_vector & text ) : filetext{text}, tokens{}, token_index{0} {
}


//-----------------------------------------------------------------------------
// No resources to clean-up, yet I specify it because I avoid auto-generated
// class functions.
//-----------------------------------------------------------------------------

Scanner::~Scanner() {
}


bool Scanner::has_more_tokens( void ) {

  return( token_index < tokens.size() );
  
}

//-----------------------------------------------------------------------------
// Tokens are stored sequentially in a token_vector.  Advance to the next
// token and return it.
// Report an overflow error, which is a programming mistake (of the scanner)
// and not a mistake of the scan itself.
//-----------------------------------------------------------------------------

const Token Scanner::get_next_token( void ) {

  ++token_index;

  if( (token_index-1) >= tokens.size() ) {
    Token error_token (TokenType::ERROR, "Programming error:  Token stack overflow", 0 );
    return( error_token );
  }
  
  return( tokens[token_index-1] );
  
}

//-----------------------------------------------------------------------------
// The method which does the heavy lifting.  The language has a small subset
// of tokens which intersect, namely, reserved_word and id.  A few symbols
// require special handling:  !=, &&, ||.  These symbols must have the
// second character along with the first to make the symbol valid.
// The other consideration are:  = == > >= < <=
// They must be maximally matched, from left to right.  However, no combination
// of those symbols can produce invalid syntax in the scanner (the parser is
// another matter).
//
// Hierarchy of scan:
//   Consume all whitespace until a non-white space is found.
//   Examine the character for meta-statement, string, reserved_word, id,
//   and finaly number.
//   If none of the above match then the character represents an illegal
//   character.
//-----------------------------------------------------------------------------

bool Scanner::tokenize( std::string & error_message ) {

  unsigned line_number = 0;
  
  for( auto & line : filetext ) {

    ++line_number;
    
    unsigned pos = 0;
    unsigned stop = 0;

    while( pos < line.length() ) {

      consume_whitespace( line, pos );
      if( pos >= line.length() ) { break; }

      if( is_meta_statement( line, pos ) ) {
	tokens.emplace_back( TokenType::META_STATEMENT, line.substr(pos), line_number );
	break;
      }

      bool legal_string = false;
      
      if( is_string( line, pos, stop, legal_string ) ) {
	
	if( legal_string ) {
	  
	  tokens.emplace_back( TokenType::STRING, line.substr(pos, (stop-pos+1)), line_number );
	  pos = stop+1;
	  continue;
	  
	} else {
	  
	  // No completed double quotes.

	  error_message = "Runaway string on line " + std::to_string(line_number) + ".";
	  
	  return(false);
	  
	}
      }

      if( is_reserved_word_or_identifier( line, pos, stop ) ) {
	std::string word = line.substr(pos,(stop-pos+1));
	if( is_reserved_word( word ) ) {
	  tokens.emplace_back( TokenType::RESERVED_WORD, word, line_number );
	} else {
	  tokens.emplace_back( TokenType::IDENTIFIER, word, line_number );
	}
	pos = stop+1;
	continue;
      }
      
      if( is_symbol( line, pos, stop ) ) {
	tokens.emplace_back( TokenType::SYMBOL, line.substr( pos, (stop-pos+1) ), line_number );
	pos = stop+1;
	continue;
      }
      
      if( is_number( line, pos, stop ) ) {
	tokens.emplace_back( TokenType::NUMBER, line.substr( pos, (stop-pos+1) ), line_number );
	pos = stop+1;
	continue;
      }

      // Illegal if we reach this point.
      
      error_message = "Illegal character '" + line.substr(pos,1) + "' found on line " +
	std::to_string( line_number ) + ".";

      return(false);
      
    }
    
  }

  tokens.emplace_back( TokenType::EOF_TOK, "", 0 );
  
  return(true);
}

#ifdef DEBUG

void Scanner::debug_display_token( const Token & token ) {

  std::cout << "DEBUG>>>  Token type:  " << token.get_token_type_display() << std::endl;
  std::cout << "DEBUG>>>  Token name:  " << token.get_token_name_display() << std::endl;

}

#endif
  
void Scanner::consume_whitespace( const std::string & line, unsigned & pos ) {

  while( pos < line.length() ) {
    
    char c = line[pos];
    if( is_whitespace(c) ) {
      ++pos;
    } else {
      return;
    }
    
  }

  return;

}

bool Scanner::is_whitespace( char c ) {

  return( (c == ' ') || (c == '\t') );
  
}

bool Scanner::is_meta_statement( const std::string & text, unsigned start ) {

  auto pos = start;
  
  if( pos >= text.length() ) {
    return(false);
  }

  // Check for # for macros.

  if( text[pos] == '#' ) {
    return(true);
  }

  if( pos+1 >= text.length() ) {
    return(false);
  }

  // Check for // for comments.
  
  return( (text[pos] == '/') && (text[pos+1] == '/') );
  
}
bool Scanner::is_string( const std::string & text,
			 unsigned start,
			 unsigned & stop,
			 bool & legal_string ) {

  unsigned pos = start;
  
  if( pos >= text.length() ) {
    return(false);
  }

  legal_string = true;
  
  char c = text[pos];
  if( c == '"' ) {
    ++pos;
    legal_string = false;
    while( pos < text.length() ) {
      c = text[pos];
      if( c == '"' ) {
	legal_string = true;
	stop = pos;
	return(true);
      }
      ++pos;
    } 

    return(true);
  }

  return(false);
  
}

bool Scanner::is_reserved_word_or_identifier( const std::string & text, unsigned start, unsigned & stop ) {

  auto pos = start;
  
  if( pos >= text.length() ) {
    return(false);
  }
  
  // Start

  // <identifier> --> <letter> (<letter> | <digit>)*

  if( !is_letter(text[pos]) ) {
    return(false);
  }

  ++pos;
  
  while( pos < text.length() ) {
    if( is_letter( text[pos] ) || is_digit( text[pos] ) ) {
      ++pos;
    } else {
      stop = pos-1;
      return( true );
    }
  }
    
  stop = pos-1;
  return(true);
  
}

bool Scanner::is_reserved_word( const std::string & text ) {

  auto rw = reserved_words.find( text );
  if( rw != reserved_words.end() ) {
    return(true);
  }

  return(false);
  
}

bool Scanner::is_letter( char c ) {

  // Not relying on ASCII to save me.

  switch( c ) {
   
  case '_' : { return(true); }
    
  case 'a' : { return(true); }
  case 'b' : { return(true); }
  case 'c' : { return(true); }
  case 'd' : { return(true); }
  case 'e' : { return(true); }
  case 'f' : { return(true); }
  case 'g' : { return(true); }
  case 'h' : { return(true); }
  case 'i' : { return(true); }
  case 'j' : { return(true); }
  case 'k' : { return(true); }
  case 'l' : { return(true); }
  case 'm' : { return(true); }
  case 'n' : { return(true); }
  case 'o' : { return(true); }
  case 'p' : { return(true); }
  case 'q' : { return(true); }
  case 'r' : { return(true); }
  case 's' : { return(true); }
  case 't' : { return(true); }
  case 'u' : { return(true); }
  case 'v' : { return(true); }
  case 'w' : { return(true); }
  case 'x' : { return(true); }
  case 'y' : { return(true); }
  case 'z' : { return(true); }
   
  case 'A' : { return(true); }
  case 'B' : { return(true); }
  case 'C' : { return(true); }
  case 'D' : { return(true); }
  case 'E' : { return(true); }
  case 'F' : { return(true); }
  case 'G' : { return(true); }
  case 'H' : { return(true); }
  case 'I' : { return(true); }
  case 'J' : { return(true); }
  case 'K' : { return(true); }
  case 'L' : { return(true); }
  case 'M' : { return(true); }
  case 'N' : { return(true); }
  case 'O' : { return(true); }
  case 'P' : { return(true); }
  case 'Q' : { return(true); }
  case 'R' : { return(true); }
  case 'S' : { return(true); }
  case 'T' : { return(true); }
  case 'U' : { return(true); }
  case 'V' : { return(true); }
  case 'W' : { return(true); }
  case 'X' : { return(true); }
  case 'Y' : { return(true); }
  case 'Z' : { return(true); }
    
  default: { return(false); }
    
  }
  
}

bool Scanner::is_digit( char c ) {

  switch( c ) {
  case '0' : { return(true); }
  case '1' : { return(true); }
  case '2' : { return(true); }
  case '3' : { return(true); }
  case '4' : { return(true); }
  case '5' : { return(true); }
  case '6' : { return(true); }
  case '7' : { return(true); }
  case '8' : { return(true); }
  case '9' : { return(true); }
  default : {return(false); }

  }
  
}

bool Scanner::is_symbol( const std::string & text, unsigned & start, unsigned & stop ) {

  // Do simple symbols first.

  unsigned pos = start;
  
  // ( ) { } [ ] , ; + - * / == != > >= < <= = && ||

  if( pos >= text.length() ) {
    return(false);
  }
   
  char c = text[pos];
  auto ssf = simple_symbols.find(c);
  if( ssf != simple_symbols.end() ) {
    stop = pos;
    return(true);
  }

  // Last char on line check.

  if( pos+1 >= text.length() ) {

    if( (c == '>') || (c == '<') || (c == '=') ) {
      stop = pos;
      return(true);
    }
    
    return(false);
    
  } else {

    char c2 = text[pos+1];

    if( (c == '=') ) {
      if( c2 == '=' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(true);
      }
    }
    
    if( (c == '!') ) {
      if( c2 == '=' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(false);
      }
    }
    
    if( (c == '&') ) {
      if( c2 == '&' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(false);
      }
    }
    
    if( (c == '|') ) {
      if( c2 == '|' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(false);
      }
    }
    
    if( (c == '>') ) {
      if( c2 == '=' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(true);
      }
    }
    
    if( (c == '<') ) {
      if( c2 == '=' ) {
	stop = pos+1;
	return(true);
      } else {
	stop = pos;
	return(true);
      }
    }
    
  }
  
  return(false);
  
}

bool Scanner::is_number( const std::string & text, unsigned & start, unsigned & stop ) {

  auto pos = start;
  
  if( pos >= text.length() ) {
    return(false);
  }
  
  if( !is_digit(text[pos]) ) {
    return(false);
  }

  ++pos;
  
  while( pos < text.length() ) {
    if( is_digit(text[pos]) ) {
      ++pos;
    } else {
      stop = pos-1;
      return( true );
    }
  }
    
  stop = pos-1;
  return(true);
  
}
