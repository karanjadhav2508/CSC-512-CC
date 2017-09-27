#pragma once

#include <string>

enum class TokenType
  { IDENTIFIER, NUMBER, RESERVED_WORD, SYMBOL, STRING, META_STATEMENT, ERROR, EOF_TOK, INITIAL };

class Token {

public:

  Token() = delete;
  Token( TokenType type, const std::string & text, unsigned line_number );
  Token( TokenType type, const std::string & text );
  Token( TokenType type);
  
  virtual ~Token();

  Token( const Token & token );
  Token( Token && token );

  const Token & operator=( const Token & token );
  const Token & operator=( Token && token );

  TokenType get_token_type( void );
  const std::string & get_token_name( void );
  unsigned get_line_number( void );

#ifdef DEBUG

  const std::string get_token_type_display(void) const;
  const std::string get_token_name_display(void) const;
  
#endif
  
protected:
private:

  TokenType   token_type;
  std::string name;
  unsigned    line_number;
  
};
