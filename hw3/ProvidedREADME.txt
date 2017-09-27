Project 2:  Parser
Tim Ozimek
CSC512

================================================
Outline of this file:
================================================

   1) Instructions to compile and run the parser.
   2) Brief overview of the solution, including algorithms used.
   3) The LL(1) grammar including FIRST+ sets, non-terminal counts, and
      production counts.
   4) Addendum of programs I generated to construct the grammar via a
      Perl program.

================================================
1) Instructions to compile and run the parser.
================================================

To compile:

   cd project2
   make

To run:

   bin/parser <input_file>

================================================
2) Overview
================================================

I modified and reused my project 1 scanner and connected it with the parser
in project 2.

For the scanner, I removed the output file of tokens as it was not listed as
a requirement for this assignment.  However, it will still flag errors to
standard output.  Further, I added an EOF token to my enumerated token types and
placed it on the bottom of the token stack before parsing begins.  The file
is read into a string_vector and fed to the scanner.  The scanner houses the
token stack.  Tokens contain type and name information that the parser uses.
The scanner object is passed to the parser object.

The parser is a recursive descent LL(1) parser.  Remarkably, it has a small
interface footprint:

   -- constructor / destructor

   -- parse -- simply called the first production rule.

   -- get_variable_count, get_function_count, get_statement_count

Privately:

   -- get_next_word

   -- A series of functions that match one-to-one with the non-terminals in the
      grammar.

The routines are largely self explanatory.  For each production a check is first
made against the FIRST_PLUS set.  If there's a matching token the right hand side
of the production is traversed.  Otherwise routine moves on to the next production
with a common left hand side.  If none of the productions match, including EPSILON,
a fail_state variable is set within the Parse object and the parse stack unwinds.
I felt the unwinding would later be useful to display errors.

All routines are LL(1) compliant.  At no time does parse look ahead to more than
one token, except when to bypass META_STATEMENTS, which is mentioned as a requirement.

Function get_next_word deserves a little note.  Because the token list may contain
meta statements they must be ignored.  Also, if the token stack is empty it must be
reported as an error.  Whenever the parser calls get_next_word, it is expecting at
least one more token in a parse.

The FIRST PLUS set and check routine are in first_plus.cpp/h and contain a table
entry for all productions.

The functions that implement the non-terminals of the grammar follow a very basic
procedure.  Aside from discarding meta-statements they take the following action:

Assume we have a production (uppercase non-terminal, lowercase terminals).

   A  --> B c D         FIRST_PLUS( m )
        | a             FIRST_PLUS( a )
        | EPSILON       FIRST_PLUS( EPSILON r )

This would be coded as (discussion continues on other side of separator):

=======================================================================================================

bool A( void ) {


  if( check_first_plus_set( current_word, FirstPlus::A_p0 ) ) { // Examine table to see to go down RHS.
  
                    //------------------------------------------------------------
    if( B() ) {     // Match to the first production, call non-terminal B().
                    //------------------------------------------------------------

      //------------------------------------------------------------
      // See if the current token on the stack matches csymbol.
      //------------------------------------------------------------
      
      if( (current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'csymbol') ) {
      
         //------------------------------------------------------------
         // Token is used, grab the next.
	 //------------------------------------------------------------
	 
         if( get_next_word() ) {
	 
            //------------------------------------------------------------
            // Descend to non-terminal D
	    //------------------------------------------------------------
	    
            if( D() ) {
	    
              //------------------------------------------------------------
              // We've matched the first production at this point.
	      //------------------------------------------------------------
	      
              return(true);

            }

         }

      }

      //------------------------------------------------------------
      // The second production just needs to check for terminal 'a'.
      //------------------------------------------------------------
      
   } else if( check_first_plus_set( current_word, FirstPlus::A_p1 ) ) {  // Examine table to see to go down RHS.

     if((current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'asymbol')) {

      //------------------------------------------------------------
      // Fetch the next word.
      //------------------------------------------------------------
      
      if( get_next_word() ) {

        //------------------------------------------------------------
        // The second production matches.
        //------------------------------------------------------------

        return(true);

      }

   } else if( check_first_plus_set( current_word, FirstPlus::A_p2 ) ) {  // Examine table to see to go down RHS.

      if((current_word.get_token_type() == SYMBOL) && (current_word.get_token_name() == 'rsymbol')) {

        //------------------------------------------------------------
        // The EPSILON case must check to see if the token on the stack is valid.
	// It must not consume the token by fetching another word. The third
	// production matches.
        //------------------------------------------------------------

        return(true);

   }

        //------------------------------------------------------------
        // Could not match any of the 3 productions so A() is not matched and returns false.
        //------------------------------------------------------------

   fail_state = true;   // Global fail state (within Parse object).

   return(false);
   
}

=======================================================================================================

These 3 basic rules are all that's needed in any of the productions.  It is very mechanical.  In fact,
I wrote a Perl program to parse my grammar file and produce all of these rules as well as the entire
first_plus.cpp / first_plus.h.  I had being doing it by hand over too many failed attempts that I decided
to automate the process.  I can even make an FSM later and compress the entire functions to one loop that
just reads rules on a table.  But I don't have enough time. Besides, the code is easier to debug outside
of an FSM as it is.

================================================
3) Grammar
================================================

The non terminal names originally had an a space delimiting names between <>.  I replaced spaces with
'_' to ease my eyes.

The format is the usual that the book and Dr. Shen have used.  Each production will also have the
FIRST_PLUS set following it.

The start state is <program_start>.

   ======================================
   
   Statistics:

      109 total productions     
      51 non-terminals         
      
   ======================================

      
<program_start>              --> <program> eof                     [FIRST_PLUS = { binary decimal int void }]
                               | eof                               [FIRST_PLUS = { eof }]

<program>                    --> <type_name> ID <program_0>        [FIRST_PLUS = { binary decimal int void }]

<type_name>                  --> int                               [FIRST_PLUS = { int }]
                               | void                              [FIRST_PLUS = { void }]
                               | binary                            [FIRST_PLUS = { binary }]
                               | decimal                           [FIRST_PLUS = { decimal }]

<program_0>                  --> <id_0> <id_list_0> semicolon <program_1>    [FIRST_PLUS = { comma left_bracket semicolon }]
                               | left_parenthesis <func_0> <func_path>       [FIRST_PLUS = { left_parenthesis }]

<id_0>                       --> left_bracket <expression> right_bracket     [FIRST_PLUS = { left_bracket }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON comma semicolon }]

<id_list_0>                  --> comma <id> <id_list_0>                      [FIRST_PLUS = { comma }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON semicolon }]

<program_1>                  --> <type_name> ID <func_or_data>               [FIRST_PLUS = { binary decimal int void }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON eof }]

<func_0>                     --> <parameter_list> right_parenthesis <func_1>    [FIRST_PLUS = { binary decimal int void }]
                               | right_parenthesis <func_4>                     [FIRST_PLUS = { right_parenthesis }]

<func_path>                  --> <func_list>                                 [FIRST_PLUS = { binary decimal int void }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON eof }]

<expression>                 --> <factor> <term_0> <expression_0>            [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<id>                         --> ID <id_0>                                   [FIRST_PLUS = { ID }]

<func_or_data>               --> <id_0> <id_list_0> semicolon <program_1>    [FIRST_PLUS = { comma left_bracket semicolon }]
                               | left_parenthesis <func_0> <func_list_0>     [FIRST_PLUS = { left_parenthesis }]

<parameter_list>             --> void <parameter_list_0>                     [FIRST_PLUS = { void }]
                               | int ID <non_empty_list_0>                   [FIRST_PLUS = { int }]
                               | decimal ID <non_empty_list_0>               [FIRST_PLUS = { decimal }]
                               | binary ID <non_empty_list_0>                [FIRST_PLUS = { binary }]

<func_1>                     --> semicolon                                   [FIRST_PLUS = { semicolon }]
                               | left_brace <func_2>                         [FIRST_PLUS = { left_brace }]

<func_4>                     --> semicolon                                   [FIRST_PLUS = { semicolon }]
                               | left_brace <func_5>                         [FIRST_PLUS = { left_brace }]

<func_list>                  --> <func> <func_list_0>                        [FIRST_PLUS = { binary decimal int void }]

<factor>                     --> ID <factor_0>                               [FIRST_PLUS = { ID }]
                               | NUMBER                                      [FIRST_PLUS = { NUMBER }]
                               | minus_sign NUMBER                           [FIRST_PLUS = { minus_sign }]
                               | left_parenthesis <expression> right_parenthesis      [FIRST_PLUS = { left_parenthesis }]

<term_0>                     --> <mulop> <factor> <term_0>                            [FIRST_PLUS = { forward_slash star_sign }]
                               | EPSILON                   [FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign minus_sign plus_sign right_bracket right_parenthesis semicolon }]

<expression_0>               --> <addop> <term> <expression_0>                        [FIRST_PLUS = { minus_sign plus_sign }]
                               | EPSILON                   [FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign right_bracket right_parenthesis semicolon }]

<func_list_0>                --> <func_list>                                 [FIRST_PLUS = { binary decimal int void }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON eof }]

<parameter_list_0>           --> ID <non_empty_list_0>                       [FIRST_PLUS = { ID }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON right_parenthesis }]

<non_empty_list_0>           --> comma <type_name> ID <non_empty_list_0>     [FIRST_PLUS = { comma }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON right_parenthesis }]

<func_2>                     --> <data_decls> <func_3>                       [FIRST_PLUS = { binary decimal int void }]
                               | <statements> right_brace                    [FIRST_PLUS = { ID break continue if print read return while write }]
                               | right_brace                                 [FIRST_PLUS = { right_brace }]

<func_5>                     --> <data_decls> <func_6>                       [FIRST_PLUS = { binary decimal int void }]
                               | <statements> right_brace                    [FIRST_PLUS = { ID break continue if print read return while write }]
                               | right_brace                                 [FIRST_PLUS = { right_brace }]

<func>                       --> <type_name> ID left_parenthesis <func_0>    [FIRST_PLUS = { binary decimal int void }]

<factor_0>                   --> left_bracket <expression> right_bracket     [FIRST_PLUS = { left_bracket }]
                               | left_parenthesis <factor_1>                 [FIRST_PLUS = { left_parenthesis }]
                               | EPSILON                     [FIRST_PLUS = { != < <= == > >= EPSILON comma double_and_sign double_or_sign forward_slash minus_sign plus_sign right_bracket right_parenthesis semicolon star_sign }]

<mulop>                      --> star_sign                                   [FIRST_PLUS = { star_sign }]
                               | forward_slash                               [FIRST_PLUS = { forward_slash }]

<addop>                      --> plus_sign                                   [FIRST_PLUS = { plus_sign }]
                               | minus_sign                                  [FIRST_PLUS = { minus_sign }]

<term>                       --> <factor> <term_0>                           [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<data_decls>                 --> <type_name> <id_list> semicolon <data_decls_0>    [FIRST_PLUS = { binary decimal int void }]

<func_3>                     --> <statements> right_brace                    [FIRST_PLUS = { ID break continue if print read return while write }]
                               | right_brace                                 [FIRST_PLUS = { right_brace }]

<statements>                 --> <statement> <statements_0>                  [FIRST_PLUS = { ID break continue if print read return while write }]

<func_6>                     --> <statements> right_brace                    [FIRST_PLUS = { ID break continue if print read return while write }]
                               | right_brace                                 [FIRST_PLUS = { right_brace }]

<factor_1>                   --> <expr_list> right_parenthesis               [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]
                               | right_parenthesis                           [FIRST_PLUS = { right_parenthesis }]

<id_list>                    --> <id> <id_list_0>                            [FIRST_PLUS = { ID }]

<data_decls_0>               --> <data_decls>                                [FIRST_PLUS = { binary decimal int void }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON ID break continue if print read return right_brace while write }]

<statement>                  --> ID <statement_0>                                                                          [FIRST_PLUS = { ID }]
                               | if left_parenthesis <condition_expression> right_parenthesis <block_statements>           [FIRST_PLUS = { if }]
                               | while left_parenthesis <condition_expression> right_parenthesis <block_statements>        [FIRST_PLUS = { while }]
                               | return <statement_2>                                                                      [FIRST_PLUS = { return }]
                               | break semicolon                                                                           [FIRST_PLUS = { break }]
                               | continue semicolon                                                                        [FIRST_PLUS = { continue }]
                               | read left_parenthesis ID right_parenthesis semicolon                                      [FIRST_PLUS = { read }]
                               | write left_parenthesis <expression> right_parenthesis semicolon                           [FIRST_PLUS = { write }]
                               | print left_parenthesis STRING right_parenthesis semicolon                                 [FIRST_PLUS = { print }]

<statements_0>               --> <statements>                                [FIRST_PLUS = { ID break continue if print read return while write }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON right_brace }]

<expr_list>                  --> <non_empty_expr_list>                       [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<statement_0>                --> equal_sign <expression> semicolon                                                 [FIRST_PLUS = { equal_sign }]
                               | left_bracket <expression> right_bracket equal_sign <expression> semicolon         [FIRST_PLUS = { left_bracket }]
                               | left_parenthesis <statement_1>                                                    [FIRST_PLUS = { left_parenthesis }]

<condition_expression>       --> <condition> <condition_expression_0>        [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<block_statements>           --> left_brace <block_statements_0>             [FIRST_PLUS = { left_brace }]

<statement_2>                --> <expression> semicolon                      [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]
                               | semicolon                                   [FIRST_PLUS = { semicolon }]

<non_empty_expr_list>        --> <expression> <non_empty_expr_list_0>        [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<statement_1>                --> <expr_list> right_parenthesis semicolon     [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]
                               | right_parenthesis semicolon                 [FIRST_PLUS = { right_parenthesis }]

<condition>                  --> <expression> <comparison_op> <expression>   [FIRST_PLUS = { ID NUMBER left_parenthesis minus_sign }]

<condition_expression_0>     --> <condition_op> <condition>                  [FIRST_PLUS = { double_and_sign double_or_sign }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON right_parenthesis }]

<block_statements_0>         --> <statements> right_brace                    [FIRST_PLUS = { ID break continue if print read return while write }]
                               | right_brace                                 [FIRST_PLUS = { right_brace }]

<non_empty_expr_list_0>      --> comma <expression> <non_empty_expr_list_0>  [FIRST_PLUS = { comma }]
                               | EPSILON                                     [FIRST_PLUS = { EPSILON right_parenthesis }]

<comparison_op>              --> ==                     [FIRST_PLUS = { == }]
                               | !=                     [FIRST_PLUS = { != }]
                               | >                      [FIRST_PLUS = { > }]
                               | >=                     [FIRST_PLUS = { >= }]
                               | <                      [FIRST_PLUS = { < }]
                               | <=                     [FIRST_PLUS = { <= }]

<condition_op>               --> double_and_sign        [FIRST_PLUS = { double_and_sign }]
                               | double_or_sign         [FIRST_PLUS = { double_or_sign }]



===============================================================================================

4) Addendum

I wrote a nifty program called ell.pl.  In implements in Perl:

    read input language file
    left recursion removal
    left factoring
    first set generation
    follow set generation
    first plus set generation
    conflict detection
    dead rule removal
    prettified grammar output along with FIRST/FOLLOW/FIRST+

As I mentioned in my notes above, I also wrote a Perl program to generate the first set
tables and the non terminal subroutines.  Have a look at grammar/GOLDEN/generate_code.pl if you
want to see it.



    
    
