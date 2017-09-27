#include "token.h"

//-----------------------------------------------------------------------------
// This class is simple and self-explanatory.  For the initial assignment the
// symbol value is recorded as the string character.  For the parser assignment
// I plan to enumerate all symbols into their own enum.  For example:
//
// ( => SYMBOL_LEFT_PAREN
// ) => SYMBOL_RIGHT_PAREN
// ... etc.
//
// Line number is recorded with the token for error printing.
//
// Error tokens will be added later if parse recovery becomes a requirement.
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Overloaded constructors.  The parser generates tokens in the first plus table.
// It doesn't need line_number information or text information for TokenTypes that
// actual text is unused (by the parser in the first+ table).
//-----------------------------------------------------------------------------


Token::Token( TokenType t_type, const std::string & text, unsigned line_number )
  : token_type{t_type}, name{text}, line_number{line_number} {
}

Token::Token( TokenType t_type, const std::string & text )
  : token_type{t_type}, name{text}, line_number{0} {
}

Token::Token( TokenType t_type )
  : token_type{t_type}, name{""}, line_number{0} {
}

Token::~Token() {
}

Token::Token( const Token & token ) {
  
  this->token_type = token.token_type;
  this->name = token.name;
  this->line_number = token.line_number;
  
}

Token::Token( Token && token ) :
  token_type{token.token_type},
  name{ std::move( token.name ) },
  line_number{line_number} {
}

const Token & Token::operator=( const Token & source ) {

  if( this != &source ) {
    this->token_type = source.token_type;
    this->name = source.name;
    this->line_number = source.line_number;
  }

  return(*this);
  
}

const Token & Token::operator=( Token && source ) {

  if( this != &source ) {
    this->token_type = source.token_type;
    this->name = std::move( source.name );
    this->line_number = source.line_number;
  }

  return(*this);
  
}

TokenType Token::get_token_type( void ) {

  return( token_type );
  
}

const std::string & Token::get_token_name( void ) {

  return( name );
  
}

unsigned Token::get_line_number( void ) {

  return(line_number);
  
}

//----------------------------------------------------------------------
// A debug routine that converts enumerations to strings for display.
//----------------------------------------------------------------------

#ifdef DEBUG

const std::string Token::get_token_type_display( void ) const {

  switch( token_type ) {
    
  case( TokenType::IDENTIFIER )     : { return( "IDENTIFIER" ); }
  case( TokenType::NUMBER )         : { return( "NUMBER" ); }
  case( TokenType::RESERVED_WORD )  : { return( "RESERVED_WORD" ); }
  case( TokenType::SYMBOL )         : { return( "SYMBOL" ); }
  case( TokenType::STRING )         : { return( "STRING" ); }
  case( TokenType::META_STATEMENT ) : { return( "META_STATEMENT" ); }
  case( TokenType::ERROR )          : { return( "ERROR"); }
  case( TokenType::EOF_TOK )        : { return( "EOF_TOK"); }
  case( TokenType::INITIAL )        : { return( "INITIAL"); }
  default                           : { return( "<<<PROGRAMMING ERROR>>>" ); }
    
  }
  
}

const std::string Token::get_token_name_display( void ) const {

  return( name );
  
}

#endif
