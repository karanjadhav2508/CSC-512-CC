#pragma once
#include "token.h" 

#include <map>
#include <string>
#include <vector>

//-------------------------------------------------------------
// The first plus table enumerated for all first plus checks.
//-------------------------------------------------------------

enum class FirstPlus {
  program_start_p0,
  program_start_p1,
  program_p0,
  type_name_p0,
  type_name_p1,
  type_name_p2,
  type_name_p3,
  program_0_p0,
  program_0_p1,
  id_0_p0,
  id_0_p1,
  id_list_0_p0,
  id_list_0_p1,
  program_1_p0,
  program_1_p1,
  func_0_p0,
  func_0_p1,
  func_path_p0,
  func_path_p1,
  expression_p0,
  id_p0,
  func_or_data_p0,
  func_or_data_p1,
  parameter_list_p0,
  parameter_list_p1,
  parameter_list_p2,
  parameter_list_p3,
  func_1_p0,
  func_1_p1,
  func_4_p0,
  func_4_p1,
  func_list_p0,
  factor_p0,
  factor_p1,
  factor_p2,
  factor_p3,
  term_0_p0,
  term_0_p1,
  expression_0_p0,
  expression_0_p1,
  func_list_0_p0,
  func_list_0_p1,
  parameter_list_0_p0,
  parameter_list_0_p1,
  non_empty_list_0_p0,
  non_empty_list_0_p1,
  func_2_p0,
  func_2_p1,
  func_2_p2,
  func_5_p0,
  func_5_p1,
  func_5_p2,
  func_p0,
  factor_0_p0,
  factor_0_p1,
  factor_0_p2,
  mulop_p0,
  mulop_p1,
  addop_p0,
  addop_p1,
  term_p0,
  data_decls_p0,
  func_3_p0,
  func_3_p1,
  statements_p0,
  func_6_p0,
  func_6_p1,
  factor_1_p0,
  factor_1_p1,
  id_list_p0,
  data_decls_0_p0,
  data_decls_0_p1,
  statement_p0,
  statement_p1,
  statement_p2,
  statement_p3,
  statement_p4,
  statement_p5,
  statement_p6,
  statement_p7,
  statement_p8,
  statements_0_p0,
  statements_0_p1,
  expr_list_p0,
  statement_0_p0,
  statement_0_p1,
  statement_0_p2,
  condition_expression_p0,
  block_statements_p0,
  statement_2_p0,
  statement_2_p1,
  non_empty_expr_list_p0,
  statement_1_p0,
  statement_1_p1,
  condition_p0,
  condition_expression_0_p0,
  condition_expression_0_p1,
  block_statements_0_p0,
  block_statements_0_p1,
  non_empty_expr_list_0_p0,
  non_empty_expr_list_0_p1,
  comparison_op_p0,
  comparison_op_p1,
  comparison_op_p2,
  comparison_op_p3,
  comparison_op_p4,
  comparison_op_p5,
  condition_op_p0,
  condition_op_p1
};

//-------------------------------------------------------------
// Change difficult syntax to ease on the eyes.
//-------------------------------------------------------------

typedef std::map<FirstPlus,std::vector<Token>> FIRST_PLUS_SET;

//-------------------------------------------------------------
// Establish the first plus sets.  This should be called
// only one time across all Parse objects.  The Parse class
// enforces this rule.
//-------------------------------------------------------------

void initialize_first_plus_sets( void );

//-------------------------------------------------------------
// Compare a parse token and see if it's in the first plus set.
// Use the enumeration to identify the correct first plus table.
//-------------------------------------------------------------

bool check_first_plus_set( Token & token, FirstPlus name );

