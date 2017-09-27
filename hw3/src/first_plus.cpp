#include "first_plus.h"
#include "token.h" 

#include <string>

FIRST_PLUS_SET first_plus;

//---------------------------------------------------------------------
// The First Plus set for all the productions.  Underneath this is a
// std::map of enums that point to token information.  The parser
// simply reads from the global map to do all First Plus checks.
// This routine initializes the global map.
//---------------------------------------------------------------------

void initialize_first_plus_sets( void ) { 

  first_plus[FirstPlus::program_start_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::program_start_p1] = {
    { TokenType::EOF_TOK, "" }
  };

  first_plus[FirstPlus::program_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::type_name_p0] = {
    { TokenType::RESERVED_WORD, "int" }
  };

  first_plus[FirstPlus::type_name_p1] = {
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::type_name_p2] = {
    { TokenType::RESERVED_WORD, "binary" }
  };

  first_plus[FirstPlus::type_name_p3] = {
    { TokenType::RESERVED_WORD, "decimal" }
  };

  first_plus[FirstPlus::program_0_p0] = {
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, "[" },
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::program_0_p1] = {
    { TokenType::SYMBOL, "(" }
  };

  first_plus[FirstPlus::id_0_p0] = {
    { TokenType::SYMBOL, "[" }
  };

  first_plus[FirstPlus::id_0_p1] = {
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::id_list_0_p0] = {
    { TokenType::SYMBOL, "," }
  };

  first_plus[FirstPlus::id_list_0_p1] = {
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::program_1_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::program_1_p1] = {
    { TokenType::EOF_TOK, "" }
  };

  first_plus[FirstPlus::func_0_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_0_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::func_path_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_path_p1] = {
    { TokenType::EOF_TOK, "" }
  };

  first_plus[FirstPlus::expression_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::id_p0] = {
    { TokenType::IDENTIFIER, "" }
  };

  first_plus[FirstPlus::func_or_data_p0] = {
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, "[" },
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::func_or_data_p1] = {
    { TokenType::SYMBOL, "(" }
  };

  first_plus[FirstPlus::parameter_list_p0] = {
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::parameter_list_p1] = {
    { TokenType::RESERVED_WORD, "int" }
  };

  first_plus[FirstPlus::parameter_list_p2] = {
    { TokenType::RESERVED_WORD, "decimal" }
  };

  first_plus[FirstPlus::parameter_list_p3] = {
    { TokenType::RESERVED_WORD, "binary" }
  };

  first_plus[FirstPlus::func_1_p0] = {
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::func_1_p1] = {
    { TokenType::SYMBOL, "{" }
  };

  first_plus[FirstPlus::func_4_p0] = {
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::func_4_p1] = {
    { TokenType::SYMBOL, "{" }
  };

  first_plus[FirstPlus::func_list_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::factor_p0] = {
    { TokenType::IDENTIFIER, "" }
  };

  first_plus[FirstPlus::factor_p1] = {
    { TokenType::NUMBER, "" }
  };

  first_plus[FirstPlus::factor_p2] = {
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::factor_p3] = {
    { TokenType::SYMBOL, "(" }
  };

  first_plus[FirstPlus::term_0_p0] = {
    { TokenType::SYMBOL, "/" },
    { TokenType::SYMBOL, "*" }
  };

  first_plus[FirstPlus::term_0_p1] = {
    { TokenType::SYMBOL, "!=" },
    { TokenType::SYMBOL, "<" },
    { TokenType::SYMBOL, "<=" },
    { TokenType::SYMBOL, "==" },
    { TokenType::SYMBOL, ">" },
    { TokenType::SYMBOL, ">=" },
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, "&&" },
    { TokenType::SYMBOL, "||" },
    { TokenType::SYMBOL, "-" },
    { TokenType::SYMBOL, "+" },
    { TokenType::SYMBOL, "]" },
    { TokenType::SYMBOL, ")" },
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::expression_0_p0] = {
    { TokenType::SYMBOL, "-" },
    { TokenType::SYMBOL, "+" }
  };

  first_plus[FirstPlus::expression_0_p1] = {
    { TokenType::SYMBOL, "!=" },
    { TokenType::SYMBOL, "<" },
    { TokenType::SYMBOL, "<=" },
    { TokenType::SYMBOL, "==" },
    { TokenType::SYMBOL, ">" },
    { TokenType::SYMBOL, ">=" },
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, "&&" },
    { TokenType::SYMBOL, "||" },
    { TokenType::SYMBOL, "]" },
    { TokenType::SYMBOL, ")" },
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::func_list_0_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_list_0_p1] = {
    { TokenType::EOF_TOK, "" }
  };

  first_plus[FirstPlus::parameter_list_0_p0] = {
    { TokenType::IDENTIFIER, "" }
  };

  first_plus[FirstPlus::parameter_list_0_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::non_empty_list_0_p0] = {
    { TokenType::SYMBOL, "," }
  };

  first_plus[FirstPlus::non_empty_list_0_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::func_2_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_2_p1] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::func_2_p2] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::func_5_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_5_p1] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::func_5_p2] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::func_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::factor_0_p0] = {
    { TokenType::SYMBOL, "[" }
  };

  first_plus[FirstPlus::factor_0_p1] = {
    { TokenType::SYMBOL, "(" }
  };

  first_plus[FirstPlus::factor_0_p2] = {
    { TokenType::SYMBOL, "!=" },
    { TokenType::SYMBOL, "<" },
    { TokenType::SYMBOL, "<=" },
    { TokenType::SYMBOL, "==" },
    { TokenType::SYMBOL, ">" },
    { TokenType::SYMBOL, ">=" },
    { TokenType::SYMBOL, "," },
    { TokenType::SYMBOL, "&&" },
    { TokenType::SYMBOL, "||" },
    { TokenType::SYMBOL, "/" },
    { TokenType::SYMBOL, "-" },
    { TokenType::SYMBOL, "+" },
    { TokenType::SYMBOL, "]" },
    { TokenType::SYMBOL, ")" },
    { TokenType::SYMBOL, ";" },
    { TokenType::SYMBOL, "*" }
  };

  first_plus[FirstPlus::mulop_p0] = {
    { TokenType::SYMBOL, "*" }
  };

  first_plus[FirstPlus::mulop_p1] = {
    { TokenType::SYMBOL, "/" }
  };

  first_plus[FirstPlus::addop_p0] = {
    { TokenType::SYMBOL, "+" }
  };

  first_plus[FirstPlus::addop_p1] = {
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::term_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::data_decls_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::func_3_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::func_3_p1] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::statements_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::func_6_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::func_6_p1] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::factor_1_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::factor_1_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::id_list_p0] = {
    { TokenType::IDENTIFIER, "" }
  };

  first_plus[FirstPlus::data_decls_0_p0] = {
    { TokenType::RESERVED_WORD, "binary" },
    { TokenType::RESERVED_WORD, "decimal" },
    { TokenType::RESERVED_WORD, "int" },
    { TokenType::RESERVED_WORD, "void" }
  };

  first_plus[FirstPlus::data_decls_0_p1] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::SYMBOL, "}" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::statement_p0] = {
    { TokenType::IDENTIFIER, "" }
  };

  first_plus[FirstPlus::statement_p1] = {
    { TokenType::RESERVED_WORD, "if" }
  };

  first_plus[FirstPlus::statement_p2] = {
    { TokenType::RESERVED_WORD, "while" }
  };

  first_plus[FirstPlus::statement_p3] = {
    { TokenType::RESERVED_WORD, "return" }
  };

  first_plus[FirstPlus::statement_p4] = {
    { TokenType::RESERVED_WORD, "break" }
  };

  first_plus[FirstPlus::statement_p5] = {
    { TokenType::RESERVED_WORD, "continue" }
  };

  first_plus[FirstPlus::statement_p6] = {
    { TokenType::RESERVED_WORD, "read" }
  };

  first_plus[FirstPlus::statement_p7] = {
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::statement_p8] = {
    { TokenType::RESERVED_WORD, "print" }
  };

  first_plus[FirstPlus::statements_0_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::statements_0_p1] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::expr_list_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::statement_0_p0] = {
    { TokenType::SYMBOL, "=" }
  };

  first_plus[FirstPlus::statement_0_p1] = {
    { TokenType::SYMBOL, "[" }
  };

  first_plus[FirstPlus::statement_0_p2] = {
    { TokenType::SYMBOL, "(" }
  };

  first_plus[FirstPlus::condition_expression_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::block_statements_p0] = {
    { TokenType::SYMBOL, "{" }
  };

  first_plus[FirstPlus::statement_2_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::statement_2_p1] = {
    { TokenType::SYMBOL, ";" }
  };

  first_plus[FirstPlus::non_empty_expr_list_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::statement_1_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::statement_1_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::condition_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::NUMBER, "" },
    { TokenType::SYMBOL, "(" },
    { TokenType::SYMBOL, "-" }
  };

  first_plus[FirstPlus::condition_expression_0_p0] = {
    { TokenType::SYMBOL, "&&" },
    { TokenType::SYMBOL, "||" }
  };

  first_plus[FirstPlus::condition_expression_0_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::block_statements_0_p0] = {
    { TokenType::IDENTIFIER, "" },
    { TokenType::RESERVED_WORD, "break" },
    { TokenType::RESERVED_WORD, "continue" },
    { TokenType::RESERVED_WORD, "if" },
    { TokenType::RESERVED_WORD, "print" },
    { TokenType::RESERVED_WORD, "read" },
    { TokenType::RESERVED_WORD, "return" },
    { TokenType::RESERVED_WORD, "while" },
    { TokenType::RESERVED_WORD, "write" }
  };

  first_plus[FirstPlus::block_statements_0_p1] = {
    { TokenType::SYMBOL, "}" }
  };

  first_plus[FirstPlus::non_empty_expr_list_0_p0] = {
    { TokenType::SYMBOL, "," }
  };

  first_plus[FirstPlus::non_empty_expr_list_0_p1] = {
    { TokenType::SYMBOL, ")" }
  };

  first_plus[FirstPlus::comparison_op_p0] = {
    { TokenType::SYMBOL, "==" }
  };

  first_plus[FirstPlus::comparison_op_p1] = {
    { TokenType::SYMBOL, "!=" }
  };

  first_plus[FirstPlus::comparison_op_p2] = {
    { TokenType::SYMBOL, ">" }
  };

  first_plus[FirstPlus::comparison_op_p3] = {
    { TokenType::SYMBOL, ">=" }
  };

  first_plus[FirstPlus::comparison_op_p4] = {
    { TokenType::SYMBOL, "<" }
  };

  first_plus[FirstPlus::comparison_op_p5] = {
    { TokenType::SYMBOL, "<=" }
  };

  first_plus[FirstPlus::condition_op_p0] = {
    { TokenType::SYMBOL, "&&" }
  };

  first_plus[FirstPlus::condition_op_p1] = {
    { TokenType::SYMBOL, "||" }
  };

}

//---------------------------------------------------------------------
// Given a token and a FirstPlus name enum, this routine compares to
// see if the Token is inside the FirstPlus set.  It returns true
// if there is match and false otherwise.
//---------------------------------------------------------------------

bool check_first_plus_set( Token & token, FirstPlus name ) {

  for( auto & t : first_plus[name] ) {

    TokenType ttype = t.get_token_type();
    std::string tname = t.get_token_name();
    
    if( ttype == token.get_token_type() ) { 

      //---------------------------------
      // These need only match by type.
      //---------------------------------
      
      if( (ttype == TokenType::IDENTIFIER) || (ttype == TokenType::NUMBER)
	  || (ttype == TokenType::STRING) || (ttype == TokenType::EOF_TOK) ) {

	return(true);

      }

      //-----------------------------------------------------------------      
      // Symbols and reserved words must match on type and token name.
      //-----------------------------------------------------------------
      
      if( (ttype == TokenType::RESERVED_WORD) && (tname == token.get_token_name()) ||
	  (ttype == TokenType::SYMBOL) && (tname == token.get_token_name()) ) {

	return(true);

      }
      
    }
    
  }

  return(false);

}

