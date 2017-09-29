#include "parser.h"

#include "first_plus.h"
#include "scanner.h"
#include "token.h"

#include <iostream>
#include <string>

//----------------------------------------------------------------------------------------------
// The Parser object.  Aside from the usual constructor / destructor pair, there is parse(),
// get_next_word(), 3 getters for the counts, and the rules for the grammer, starting with
// program_start().  For now, the first_plus_sets lives outside this class.  I do this for
// clarity.  With all of the subroutine calls of the actual grammar, it started to clutter
// the fail.  Later, after grading and some usage, I plan to refactor into two private impls,
// one per the first plus table and the other for the recursive routines.
//
// Failure is maintained in a global state.  Rather than break out of any recursive calls, I
// set the fail state and let the stack unwind naturally.  This anticipates the need for
// error correction / useful parse diagnostics.
//
// Finally, variable "Token current_word" holds the top of the Token stack.  All routines look
// to it when making decisions based on tokens.
//----------------------------------------------------------------------------------------------


Parser::Parser() :
  fail_state{false}, variable_count{0}, function_count{0}, statement_count{0},
  scanner{nullptr},
  current_word{ TokenType::INITIAL, std::string{""}, 0 }
{

  //----------------------------------------------------------------------------------------------
  // The first plus sets are statically allocated.  I could have placed them into
  // a giant initializer list and moved the initializtion to a compiler step, but
  // it becomes unreadable without variable assignment syntax on a deep nested
  // structure.  When the parser is fully vetted, I'll refactor so that this step
  // is not needed.
  //----------------------------------------------------------------------------------------------
  
  if( !first_plus_sets_initialized ) {

    initialize_first_plus_sets();
    first_plus_sets_initialized = true;

  }
}

Parser::~Parser() {
}

//----------------------------------------------------------------------------------------------
// Use the scanner and internally assign it within the Parse framework.  Now, the parser
// controls the main scan loop.  Descend into 'program_start'.  Return status of the parse.
//----------------------------------------------------------------------------------------------

bool Parser::parse( Scanner & scanner ) {

  this->scanner = &scanner;

  
#ifdef DEBUG

  this->scanner->debug_display_token( current_word );

#endif
  
  return( program_start() );

}

//----------------------------------------------------------------------------------------------
// This is a convenience routine which is pretty straightforward.  Any time a new token is
// needed this routine will fetch it.  It must skip over all meta statements and it must check
// to see if the stack contains more items.  If no more tokens are on the stack it must signal
// 'false' in the return call, which will lead to a parse error.
//----------------------------------------------------------------------------------------------

bool Parser::get_next_word( void ) {

  // Skip over meta-statements.
  
  while( scanner->has_more_tokens() ) {
    
#ifdef DEBUG

  this->scanner->debug_display_token( current_word );

#endif
  
    current_word = scanner->get_next_token();
    
    if( current_word.get_token_type() != TokenType::META_STATEMENT ) {
      
      return(true);

    }

  }

#ifdef DEBUG

  std::cout << "ran out of tokens" << std::endl;
  
#endif
  
  return( false );

}

//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//
// Parse entrance: (sketch of the structure of these routines follows --
//
// Example grammar with the 3 types of productions seen by this program:
//
//   A  --> B c D         FIRST_PLUS( m )
//        | a             FIRST_PLUS( a )
//        | EPSILON       FIRST_PLUS( EPSILON r )
//
// bool A( void ) {
//
//   if( check_first_plus_set( current_word, FirstPlus::A_p0 ) ) { // Examine table to see to go down RHS.
//   
//                     //------------------------------------------------------------
//     if( B() ) {     // Match to the first production, call non-terminal B().
//                     //------------------------------------------------------------
// 
//       //------------------------------------------------------------
//       // See if the current token on the stack matches csymbol.
//       //------------------------------------------------------------
//       
//       if( (current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'csymbol') ) {
//       
//          //------------------------------------------------------------
//          // Token is used, grab the next.
// 	    //------------------------------------------------------------
// 	 
//          if( get_next_word() ) {
// 	 
//             //------------------------------------------------------------
//             // Descend to non-terminal D
// 	       //------------------------------------------------------------
// 	    
//             if( D() ) {
// 	    
//               //------------------------------------------------------------
//               // We've matched the first production at this point.
// 	      //------------------------------------------------------------
// 	      
//               return(true);
// 
//             }
// 
//          }
// 
//       }
// 
//       //------------------------------------------------------------
//       // The second production just needs to check for terminal 'a'.
//       //------------------------------------------------------------
//       
//    } else if( check_first_plus_set( current_word, FirstPlus::A_p1 ) ) {  // Examine table to see to go down RHS.
// 
//      if((current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'asymbol')) {
// 
//       //------------------------------------------------------------
//       // Fetch the next word.
//       //------------------------------------------------------------
//       
//       if( get_next_word() ) {
// 
//         //------------------------------------------------------------
//         // The second production matches.
//         //------------------------------------------------------------
// 
//         return(true);
// 
//       }
// 
//    } else if( check_first_plus_set( current_word, FirstPlus::A_p2 ) ) {  // Examine table to see to go down RHS.
// 
//       if((current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'rsymbol')) {
// 
//         //------------------------------------------------------------
//         // The EPSILON case must check to see if the token on the stack is valid.
// 	   // It must not consume the token by fetching another word. The third
// 	   // production matches.
//         //------------------------------------------------------------
// 
//         return(true);
// 
//    }
// 
//         //------------------------------------------------------------
//         // Could not match any of the 3 productions so A() is not matched and returns false.
//         //------------------------------------------------------------
// 
//    fail_state = true;   // Global fail state (within Parse object).
// 
//    return(false);
//    
// }
// 
//
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------

bool Parser::program_start( void ) {

  //  <program_start>             --> <program> eof
  //                                | eof

#ifdef DEBUG

  std::cout << "<program_start>" << std::endl;
  
#endif
  
  if( get_next_word() ) {

    if( check_first_plus_set( current_word, FirstPlus::program_start_p0 ) ) {
 
      if( program() ) {
	if( current_word.get_token_type() == TokenType::EOF_TOK ) {
	  return(!fail_state);
	}
      }

    } else if( check_first_plus_set( current_word,  FirstPlus::program_start_p1 ) ) {
    
      if( current_word.get_token_type() == TokenType::EOF_TOK ) {
	
	return(!fail_state);
	
      }

    }

  }

  fail_state = true;

  return(false);
  
}

bool Parser::program(void) {

  // <program>                    --> <type_name> ID <program_0>                     FIRST_PLUS = { binary decimal int void }

  if( check_first_plus_set( current_word, FirstPlus::program_p0 ) ) {

    if ( type_name() ) {

      if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

        if( get_next_word() ) {

          if ( program_0() ) {

            return( true );

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::type_name(void) {

  // <type_name>                  --> int                     FIRST_PLUS = { int }
  //                                | void                     FIRST_PLUS = { void }
  //                                | binary                     FIRST_PLUS = { binary }
  //                                | decimal                     FIRST_PLUS = { decimal }

  if( check_first_plus_set( current_word, FirstPlus::type_name_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "int")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::type_name_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "void")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::type_name_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "binary")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::type_name_p3 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "decimal")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}

// TODO: Implement this method
bool Parser::program_0(void) {

  // <program_0>                  --> <id_0> <id_list_0> semicolon <program_1>                     FIRST_PLUS = { comma left_bracket semicolon }
  //                                | left_parenthesis <func_0> <func_path>                     FIRST_PLUS = { left_parenthesis }

  // Add Code Here

  #ifdef DEBUG

  std::cout << "<program_0>" << std::endl;

  #endif

  if(check_first_plus_set(current_word, FirstPlus::program_0_p0)) {
    if(id_0()) {
      ++variable_count;
      if(id_list_0()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
          ++statement_count;
          std::cout << "<program_0>" << statement_count << std::endl;
          if(get_next_word()) {
            if(program_1()) {
              return(true);
            }
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::program_0_p1)) {
    if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
      if(get_next_word()) {
        if(func_0()) {
          if(func_path()) {
            return(true);
          }
        }
      }
    }
  }

  fail_state = true;
  return(false);
}


bool Parser::id_0(void) {

  // <id_0>                       --> left_bracket <expression> right_bracket                     FIRST_PLUS = { left_bracket }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON comma semicolon }

  if( check_first_plus_set( current_word, FirstPlus::id_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "[")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "]")  ) {

            if( get_next_word() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::id_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::id_list_0(void) {

  // <id_list_0>                  --> comma <id> <id_list_0>                     FIRST_PLUS = { comma }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON semicolon }

  if( check_first_plus_set( current_word, FirstPlus::id_list_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ",")  ) {

      if( get_next_word() ) {

        if ( id() ) {

	  ++variable_count;
      
          if ( id_list_0() ) {

            return( true );

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::id_list_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::program_1(void) {

  // <program_1>                  --> <type_name> ID <func_or_data>                     FIRST_PLUS = { binary decimal int void }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON eof }

  if( check_first_plus_set( current_word, FirstPlus::program_1_p0 ) ) {

    if ( type_name() ) {

      if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

        if( get_next_word() ) {

          if ( func_or_data() ) {

            return( true );

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::program_1_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::func_0(void) {

  // <func_0>                     --> <parameter_list> right_parenthesis <func_1>                     FIRST_PLUS = { binary decimal int void }
  //                                | right_parenthesis <func_4>                     FIRST_PLUS = { right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::func_0_p0 ) ) {

    if ( parameter_list() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

        if( get_next_word() ) {

          if ( func_1() ) {

            return( true );

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_0_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

      if( get_next_word() ) {

        if ( func_4() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_path(void) {

  // <func_path>                  --> <func_list>                     FIRST_PLUS = { binary decimal int void }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON eof }

  if( check_first_plus_set( current_word, FirstPlus::func_path_p0 ) ) {

    if ( func_list() ) {

      return( true );

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_path_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::expression(void) {

  // <expression>                 --> <factor> <term_0> <expression_0>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::expression_p0 ) ) {

    if ( factor() ) {

      if ( term_0() ) {

        if ( expression_0() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::id(void) {

  // <id>                         --> ID <id_0>                     FIRST_PLUS = { ID }

  if( check_first_plus_set( current_word, FirstPlus::id_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

      if( get_next_word() ) {

        if ( id_0() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_or_data(void) {

  // <func_or_data>               --> <id_0> <id_list_0> semicolon <program_1>                     FIRST_PLUS = { comma left_bracket semicolon }
  //                                | left_parenthesis <func_0> <func_list_0>                     FIRST_PLUS = { left_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::func_or_data_p0 ) ) {

    if ( id_0() ) {

      ++variable_count;
      
      if ( id_list_0() ) {

        if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

          if( get_next_word() ) {

            if ( program_1() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_or_data_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "(")  ) {

      if( get_next_word() ) {

        if ( func_0() ) {

          if ( func_list_0() ) {

            return( true );

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::parameter_list(void) {

  // <parameter_list>             --> void <parameter_list_0>                     FIRST_PLUS = { void }
  //                                | int ID <non_empty_list_0>                     FIRST_PLUS = { int }
  //                                | decimal ID <non_empty_list_0>                     FIRST_PLUS = { decimal }
  //                                | binary ID <non_empty_list_0>                     FIRST_PLUS = { binary }

  if( check_first_plus_set( current_word, FirstPlus::parameter_list_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "void")  ) {

      if( get_next_word() ) {

        if ( parameter_list_0() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::parameter_list_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "int")  ) {

      if( get_next_word() ) {

        if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

          if( get_next_word() ) {

            if ( non_empty_list_0() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::parameter_list_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "decimal")  ) {

      if( get_next_word() ) {

        if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

          if( get_next_word() ) {

            if ( non_empty_list_0() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::parameter_list_p3 ) ) {

    if ( (current_word.get_token_type() == TokenType::RESERVED_WORD) && (current_word.get_token_name() == "binary")  ) {

      if( get_next_word() ) {

        if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

          if( get_next_word() ) {

            if ( non_empty_list_0() ) {

              return( true );

            }

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_1(void) {

  // <func_1>                     --> semicolon                     FIRST_PLUS = { semicolon }
  //                                | left_brace <func_2>                     FIRST_PLUS = { left_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_1_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_1_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "{")  ) {

      if( get_next_word() ) {

        if ( func_2() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_4(void) {

  // <func_4>                     --> semicolon                     FIRST_PLUS = { semicolon }
  //                                | left_brace <func_5>                     FIRST_PLUS = { left_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_4_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_4_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "{")  ) {

      if( get_next_word() ) {

        if ( func_5() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_list(void) {

  // <func_list>                  --> <func> <func_list_0>                     FIRST_PLUS = { binary decimal int void }

  if( check_first_plus_set( current_word, FirstPlus::func_list_p0 ) ) {

    if ( func() ) {

      if ( func_list_0() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::factor(void) {

  // <factor>                     --> ID <factor_0>                     FIRST_PLUS = { ID }
  //                                | NUMBER                     FIRST_PLUS = { NUMBER }
  //                                | minus_sign NUMBER                     FIRST_PLUS = { minus_sign }
  //                                | left_parenthesis <expression> right_parenthesis                     FIRST_PLUS = { left_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::factor_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

      if( get_next_word() ) {

        if ( factor_0() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::NUMBER)  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "-")  ) {

      if( get_next_word() ) {

        if ( (current_word.get_token_type() == TokenType::NUMBER)  ) {

          if( get_next_word() ) {

            return( true );

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_p3 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "(")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

            if( get_next_word() ) {

              return( true );

            }

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::term_0(void) {

  // <term_0>                     --> <mulop> <factor> <term_0>                     FIRST_PLUS = { forward_slash star_sign }
  //                                | EPSILON                     FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign minus_sign plus_sign right_bracket right_parenthesis semicolon }

  if( check_first_plus_set( current_word, FirstPlus::term_0_p0 ) ) {

    if ( mulop() ) {

      if ( factor() ) {

        if ( term_0() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::term_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::expression_0(void) {

  // <expression_0>               --> <addop> <term> <expression_0>                     FIRST_PLUS = { minus_sign plus_sign }
  //                                | EPSILON                     FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign right_bracket right_parenthesis semicolon }

  if( check_first_plus_set( current_word, FirstPlus::expression_0_p0 ) ) {

    if ( addop() ) {

      if ( term() ) {

        if ( expression_0() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::expression_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::func_list_0(void) {

  // <func_list_0>                --> <func_list>                     FIRST_PLUS = { binary decimal int void }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON eof }

  if( check_first_plus_set( current_word, FirstPlus::func_list_0_p0 ) ) {

    if ( func_list() ) {

      return( true );

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_list_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::parameter_list_0(void) {

  // <parameter_list_0>           --> ID <non_empty_list_0>                     FIRST_PLUS = { ID }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::parameter_list_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

      if( get_next_word() ) {

        if ( non_empty_list_0() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::parameter_list_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::non_empty_list_0(void) {

  // <non_empty_list_0>           --> comma <type_name> ID <non_empty_list_0>                     FIRST_PLUS = { comma }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::non_empty_list_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ",")  ) {

      if( get_next_word() ) {

        if ( type_name() ) {

          if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

            if( get_next_word() ) {

              if ( non_empty_list_0() ) {

                return( true );

              }

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::non_empty_list_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::func_2(void) {

  // <func_2>                     --> <data_decls> <func_3>                     FIRST_PLUS = { binary decimal int void }
  //                                | <statements> right_brace                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | right_brace                     FIRST_PLUS = { right_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_2_p0 ) ) {

    if ( data_decls() ) {

      if ( func_3() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_2_p1 ) ) {

    if ( statements() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

	++function_count;
	
        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_2_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {
      
      ++function_count;

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_5(void) {

  // <func_5>                     --> <data_decls> <func_6>                     FIRST_PLUS = { binary decimal int void }
  //                                | <statements> right_brace                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | right_brace                     FIRST_PLUS = { right_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_5_p0 ) ) {

    if ( data_decls() ) {

      if ( func_6() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_5_p1 ) ) {

    if ( statements() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {
	
	++function_count;

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_5_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

      ++function_count;
	
      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func(void) {

  // <func>                       --> <type_name> ID left_parenthesis <func_0>                     FIRST_PLUS = { binary decimal int void }

  if( check_first_plus_set( current_word, FirstPlus::func_p0 ) ) {

    if ( type_name() ) {

      if ( (current_word.get_token_type() == TokenType::IDENTIFIER)  ) {

        if( get_next_word() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "(")  ) {

            if( get_next_word() ) {

              if ( func_0() ) {

                return( true );

              }

            }

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::factor_0(void) {

  // <factor_0>                   --> left_bracket <expression> right_bracket                     FIRST_PLUS = { left_bracket }
  //                                | left_parenthesis <factor_1>                     FIRST_PLUS = { left_parenthesis }
  //                                | EPSILON                     FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign forward_slash minus_sign plus_sign right_bracket right_parenthesis semicolon star_sign }

  if( check_first_plus_set( current_word, FirstPlus::factor_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "[")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "]")  ) {

            if( get_next_word() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_0_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "(")  ) {

      if( get_next_word() ) {

        if ( factor_1() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_0_p2 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::mulop(void) {

  // <mulop>                      --> star_sign                     FIRST_PLUS = { star_sign }
  //                                | forward_slash                     FIRST_PLUS = { forward_slash }

  if( check_first_plus_set( current_word, FirstPlus::mulop_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "*")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::mulop_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "/")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::addop(void) {

  // <addop>                      --> plus_sign                     FIRST_PLUS = { plus_sign }
  //                                | minus_sign                     FIRST_PLUS = { minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::addop_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "+")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::addop_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "-")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::term(void) {

  // <term>                       --> <factor> <term_0>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::term_p0 ) ) {

    if ( factor() ) {

      if ( term_0() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::data_decls(void) {

  // <data_decls>                 --> <type_name> <id_list> semicolon <data_decls_0>                     FIRST_PLUS = { binary decimal int void }

  if( check_first_plus_set( current_word, FirstPlus::data_decls_p0 ) ) {

    if ( type_name() ) {

      if ( id_list() ) {

        if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

          if( get_next_word() ) {

            if ( data_decls_0() ) {

              return( true );

            }

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_3(void) {

  // <func_3>                     --> <statements> right_brace                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | right_brace                     FIRST_PLUS = { right_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_3_p0 ) ) {

    if ( statements() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {
	
	++function_count;

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_3_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::statements(void) {

  // <statements>                 --> <statement> <statements_0>                     FIRST_PLUS = { ID break continue if print read return while write }

  if( check_first_plus_set( current_word, FirstPlus::statements_p0 ) ) {

    if ( statement() ) {

      if ( statements_0() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::func_6(void) {

  // <func_6>                     --> <statements> right_brace                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | right_brace                     FIRST_PLUS = { right_brace }

  if( check_first_plus_set( current_word, FirstPlus::func_6_p0 ) ) {

    if ( statements() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {
	
	++function_count;

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::func_6_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::factor_1(void) {

  // <factor_1>                   --> <expr_list> right_parenthesis                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }
  //                                | right_parenthesis                     FIRST_PLUS = { right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::factor_1_p0 ) ) {

    if ( expr_list() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::factor_1_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::id_list(void) {

  // <id_list>                    --> <id> <id_list_0>                     FIRST_PLUS = { ID }

  if( check_first_plus_set( current_word, FirstPlus::id_list_p0 ) ) {

    ++variable_count;
      
    if ( id() ) {

      if ( id_list_0() ) {
	
        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::data_decls_0(void) {

  // <data_decls_0>               --> <data_decls>                     FIRST_PLUS = { binary decimal int void }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON ID break continue if print read return right_brace while write }

  if( check_first_plus_set( current_word, FirstPlus::data_decls_0_p0 ) ) {

    if ( data_decls() ) {

      return( true );

    }

  } else if( check_first_plus_set( current_word, FirstPlus::data_decls_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}

// TODO: Implement this method
bool Parser::statement(void) {

  // <statement>                  --> ID <statement_0>                     FIRST_PLUS = { ID }
  //                                | if left_parenthesis <condition_expression> right_parenthesis <block_statements>                     FIRST_PLUS = { if }
  //                                | while left_parenthesis <condition_expression> right_parenthesis <block_statements>                     FIRST_PLUS = { while }
  //                                | return <statement_2>                     FIRST_PLUS = { return }
  //                                | break semicolon                     FIRST_PLUS = { break }
  //                                | continue semicolon                     FIRST_PLUS = { continue }
  //                                | read left_parenthesis ID right_parenthesis semicolon                     FIRST_PLUS = { read }
  //                                | write left_parenthesis <expression> right_parenthesis semicolon                     FIRST_PLUS = { write }
  //                                | print left_parenthesis STRING right_parenthesis semicolon                     FIRST_PLUS = { print }


  // Add Code here

  #ifdef DEBUG

  std::cout << "<statement>" << std::endl;

  #endif

  if(check_first_plus_set(current_word, FirstPlus::statement_p0)) {
    if(current_word.get_token_type()==TokenType::IDENTIFIER) {
      if(get_next_word()) {
        if(statement_0()) {
          ++statement_count;
          std::cout << "<statement_p0>" << statement_count << std::endl;
          return(true);
        }
      }    
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p1)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="if")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
          if(get_next_word()) {
            if(condition_expression()) {
              if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==")")) {
                if(get_next_word()) {
                  if(block_statements()) {
                    return(true);
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p2)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="while")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
          if(get_next_word()) {
            if(condition_expression()) {
              if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==")")) {
                if(get_next_word()) {
                  if(block_statements()) {
                    return(true);
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p3)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="return")) {
      if(get_next_word()) {
        if(statement_2()) {
          ++statement_count;
          std::cout << "<statement_p3>" << statement_count << std::endl;
          return(true);
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p4)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="break")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
	  ++statement_count;
          if(get_next_word()) {
            return(true);
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p5)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="continue")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
          ++statement_count;
          std::cout << "<statement_p5>" << statement_count << std::endl;
          if(get_next_word()) {
            return(true);
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p6)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="read")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
          if(get_next_word()) {
            if(current_word.get_token_type()==TokenType::IDENTIFIER) {
              if(get_next_word()) {
                if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==")")) {
                  if(get_next_word()) {
                    if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
                      ++statement_count;
                      std::cout << "<statement_p6>" << statement_count << std::endl;
                      if(get_next_word()) {
                        return(true);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p7)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="write")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
          if(get_next_word()) {
            if(expression()) {
              if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==")")) {
                if(get_next_word()) {
                  if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
                    ++statement_count;
                    std::cout << "<statement_p7>" << statement_count << std::endl;
                    if(get_next_word()) {
                      return(true);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  } else if(check_first_plus_set(current_word, FirstPlus::statement_p8)) {
    if((current_word.get_token_type()==TokenType::RESERVED_WORD) && (current_word.get_token_name()=="print")) {
      if(get_next_word()) {
        if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()=="(")) {
          if(get_next_word()) {
            if(current_word.get_token_type()==TokenType::STRING) {
              if(get_next_word()) {
                if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==")")) {
                  if(get_next_word()) {
                    if((current_word.get_token_type()==TokenType::SYMBOL) && (current_word.get_token_name()==";")) {
                      ++statement_count;
                      std::cout << "<statement_p8>" << statement_count << std::endl;
                      if(get_next_word()) {
                        return(true);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  fail_state = true;
  return(false);
}


bool Parser::statements_0(void) {

  // <statements_0>               --> <statements>                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON right_brace }

  if( check_first_plus_set( current_word, FirstPlus::statements_0_p0 ) ) {

    if ( statements() ) {

      return( true );

    }

  } else if( check_first_plus_set( current_word, FirstPlus::statements_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::expr_list(void) {

  // <expr_list>                  --> <non_empty_expr_list>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::expr_list_p0 ) ) {

    if ( non_empty_expr_list() ) {

      return( true );

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::statement_0(void) {

  // <statement_0>                --> equal_sign <expression> semicolon                     FIRST_PLUS = { equal_sign }
  //                                | left_bracket <expression> right_bracket equal_sign <expression> semicolon                     FIRST_PLUS = { left_bracket }
  //                                | left_parenthesis <statement_1>                     FIRST_PLUS = { left_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::statement_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "=")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

            if( get_next_word() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::statement_0_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "[")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "]")  ) {

            if( get_next_word() ) {

              if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "=")  ) {

                if( get_next_word() ) {

                  if ( expression() ) {

                    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

                      if( get_next_word() ) {

                        return( true );

                      }

                    }

                  }

                }

              }

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::statement_0_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "(")  ) {

      if( get_next_word() ) {

        if ( statement_1() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::condition_expression(void) {

  // <condition_expression>       --> <condition> <condition_expression_0>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::condition_expression_p0 ) ) {

    if ( condition() ) {

      if ( condition_expression_0() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::block_statements(void) {

  // <block_statements>           --> left_brace <block_statements_0>                     FIRST_PLUS = { left_brace }

  if( check_first_plus_set( current_word, FirstPlus::block_statements_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "{")  ) {

      if( get_next_word() ) {

        if ( block_statements_0() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::statement_2(void) {

  // <statement_2>                --> <expression> semicolon                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }
  //                                | semicolon                     FIRST_PLUS = { semicolon }

  if( check_first_plus_set( current_word, FirstPlus::statement_2_p0 ) ) {

    if ( expression() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::statement_2_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::non_empty_expr_list(void) {

  // <non_empty_expr_list>        --> <expression> <non_empty_expr_list_0>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::non_empty_expr_list_p0 ) ) {

    if ( expression() ) {

      if ( non_empty_expr_list_0() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::statement_1(void) {

  // <statement_1>                --> <expr_list> right_parenthesis semicolon                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }
  //                                | right_parenthesis semicolon                     FIRST_PLUS = { right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::statement_1_p0 ) ) {

    if ( expr_list() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

        if( get_next_word() ) {

          if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

            if( get_next_word() ) {

              return( true );

            }

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::statement_1_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ")")  ) {

      if( get_next_word() ) {

        if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ";")  ) {

          if( get_next_word() ) {

            return( true );

          }

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::condition(void) {

  // <condition>                  --> <expression> <comparison_op> <expression>                     FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }

  if( check_first_plus_set( current_word, FirstPlus::condition_p0 ) ) {

    if ( expression() ) {

      if ( comparison_op() ) {

        if ( expression() ) {

          return( true );

        }

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::condition_expression_0(void) {

  // <condition_expression_0>     --> <condition_op> <condition>                     FIRST_PLUS = { double_and_sign double_or_sign }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::condition_expression_0_p0 ) ) {

    if ( condition_op() ) {

      if ( condition() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::condition_expression_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::block_statements_0(void) {

  // <block_statements_0>         --> <statements> right_brace                     FIRST_PLUS = { ID break continue if print read return while write }
  //                                | right_brace                     FIRST_PLUS = { right_brace }

  if( check_first_plus_set( current_word, FirstPlus::block_statements_0_p0 ) ) {

    if ( statements() ) {

      if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

        if( get_next_word() ) {

          return( true );

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::block_statements_0_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "}")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::non_empty_expr_list_0(void) {

  // <non_empty_expr_list_0>      --> comma <expression> <non_empty_expr_list_0>                     FIRST_PLUS = { comma }
  //                                | EPSILON                     FIRST_PLUS = { EPSILON right_parenthesis }

  if( check_first_plus_set( current_word, FirstPlus::non_empty_expr_list_0_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ",")  ) {

      if( get_next_word() ) {

        if ( expression() ) {

          if ( non_empty_expr_list_0() ) {

            return( true );

          }

        }

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::non_empty_expr_list_0_p1 ) ) {

      // EPSILON consumes no tokens.

    return( true );

  }

  fail_state = true;

  return(false);

}


bool Parser::comparison_op(void) {

  // <comparison_op>              --> ==                     FIRST_PLUS = { == }
  //                                | !=                     FIRST_PLUS = { != }
  //                                | >                     FIRST_PLUS = { > }
  //                                | >=                     FIRST_PLUS = { >= }
  //                                | <                     FIRST_PLUS = { < }
  //                                | <=                     FIRST_PLUS = { <= }

  if( check_first_plus_set( current_word, FirstPlus::comparison_op_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "==")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::comparison_op_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "!=")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::comparison_op_p2 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ">")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::comparison_op_p3 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == ">=")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::comparison_op_p4 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "<")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::comparison_op_p5 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "<=")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


bool Parser::condition_op(void) {

  // <condition_op>               --> double_and_sign                     FIRST_PLUS = { double_and_sign }
  //                                | double_or_sign                     FIRST_PLUS = { double_or_sign }

  if( check_first_plus_set( current_word, FirstPlus::condition_op_p0 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "&&")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  } else if( check_first_plus_set( current_word, FirstPlus::condition_op_p1 ) ) {

    if ( (current_word.get_token_type() == TokenType::SYMBOL) && (current_word.get_token_name() == "||")  ) {

      if( get_next_word() ) {

        return( true );

      }

    }

  }

  fail_state = true;

  return(false);

}


