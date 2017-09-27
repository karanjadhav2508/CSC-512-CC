#pragma once

#include "token.h"

#include <set>
#include <string>
#include <vector>

typedef std::vector<std::string> string_vector;
typedef std::vector<Token> token_vector;

class Scanner {

 public:

  Scanner() = delete;
  Scanner( const string_vector & text );
  Scanner( const Scanner & source ) = delete;
  Scanner( const Scanner && source ) = delete;

  virtual ~Scanner();

  const Scanner & operator=(const Scanner & source) = delete;
  const Scanner & operator=(const Scanner && source) = delete;

  bool has_more_tokens(void);
  const Token get_next_token(void);
  bool tokenize( std::string & error_message );

#ifdef DEBUG
  
  void debug_display_token(const Token & token);

#endif
  
 protected:
 private:

  string_vector filetext;
  token_vector  tokens;
  unsigned      token_index;

  void consume_whitespace( const std::string & line, unsigned & pos );

  bool is_whitespace( char c );
  bool is_meta_statement( const std::string & text, unsigned start );
  bool is_string( const std::string & text,
		  unsigned start,
		  unsigned & stop,
		  bool & legal_string );
  bool is_reserved_word_or_identifier( const std::string & text, unsigned start, unsigned & stop );
  bool is_reserved_word( const std::string & text );
  bool is_letter( char c );
  bool is_digit( char c );
  bool is_symbol( const std::string & text, unsigned & start, unsigned & stop );
  bool is_number( const std::string & text, unsigned & start, unsigned & stop );
  
  static const std::set<std::string> reserved_words;
  static const std::set<char> simple_symbols;

};
