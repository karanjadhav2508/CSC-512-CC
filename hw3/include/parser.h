#pragma once

#include "first_plus.h"
#include "scanner.h"
#include "token.h"

#include <map>
#include <string>
#include <vector>

static bool first_plus_sets_initialized = false;

class Parser {

 public:

  Parser();
  virtual ~Parser();

  Parser( const Parser & src ) = delete;
  Parser( const Parser && src ) = delete;

  const Parser operator=( const Parser & other ) = delete;
  const Parser operator=( const Parser && other ) = delete;

  bool parse( Scanner & scanner );
  unsigned get_variable_count(void)   { return( variable_count ); }
  unsigned get_function_count(void)   { return( function_count ); }
  unsigned get_statement_count(void)  { return( statement_count ); }
  
 protected:
 private:

  //-----------------------------------------------------------
  // Used to control one time initialization of first+ table.
  //-----------------------------------------------------------
  
  static bool first_sets_initialized;
  
  //-----------------------------------------------------------
  // Indicates if the parser detected a failing condition.
  //-----------------------------------------------------------
  
  bool fail_state;

  //-----------------------------------------------------------
  // Holds the counts requested in the assignment.
  //-----------------------------------------------------------
  
  unsigned variable_count;
  unsigned function_count;
  unsigned statement_count;

  //-----------------------------------------------------------
  // The scanner from projet 1, passed in as a reference.
  //-----------------------------------------------------------
  
  Scanner *scanner;

  //-----------------------------------------------------------
  // It holds the current top of the token stack.
  //-----------------------------------------------------------
  
  Token current_word;

  //-----------------------------------------------------------
  // Convenience function for getting the next token.
  //-----------------------------------------------------------
  
  bool get_next_word( void );

  //-----------------------------------------------------------
  // Grammar routines for LL(1) grammar.
  //-----------------------------------------------------------
  
  bool program_start(void);
  bool program(void);
  bool type_name(void);
  bool program_0(void);
  bool id_0(void);
  bool id_list_0(void);
  bool program_1(void);
  bool func_0(void);
  bool func_path(void);
  bool expression(void);
  bool id(void);
  bool func_or_data(void);
  bool parameter_list(void);
  bool func_1(void);
  bool func_4(void);
  bool func_list(void);
  bool factor(void);
  bool term_0(void);
  bool expression_0(void);
  bool func_list_0(void);
  bool parameter_list_0(void);
  bool non_empty_list_0(void);
  bool func_2(void);
  bool func_5(void);
  bool func(void);
  bool factor_0(void);
  bool mulop(void);
  bool addop(void);
  bool term(void);
  bool data_decls(void);
  bool func_3(void);
  bool statements(void);
  bool func_6(void);
  bool factor_1(void);
  bool id_list(void);
  bool data_decls_0(void);
  bool statement(void);
  bool statements_0(void);
  bool expr_list(void);
  bool statement_0(void);
  bool condition_expression(void);
  bool block_statements(void);
  bool statement_2(void);
  bool non_empty_expr_list(void);
  bool statement_1(void);
  bool condition(void);
  bool condition_expression_0(void);
  bool block_statements_0(void);
  bool non_empty_expr_list_0(void);
  bool comparison_op(void);
  bool condition_op(void);

};
