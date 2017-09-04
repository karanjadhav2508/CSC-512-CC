/**
 * This is the token class used to check the input against the RE and identify the correct token
 */

/**
 * @author Danny Reinheimer
 *
 */
public class Token {
	
	private RegularExpressions re;
	
	public Token() {
		this.re = new RegularExpressions();
	}
	
	/**
	 * Takes in the parsed input and checks it againts all the regular expressions with priority
	 * @param str the input string to test
	 * @return the name of the token
	 */
	public TokenNames getToken(String str) {
				
		// First check to see if it is meta statement
		if(re.isMetaStatement(str)) {
			return TokenNames.MetaStatements;
		}
		
		if(re.isReservedWord(str)) {
			return TokenNames.ReserveWord;
		}
		
		if(re.isIdentifier(str)) {
			return TokenNames.Identifiers;
		}
		
		if(re.isNumber(str)) {
			return TokenNames.Numbers;
		}
		
		if(re.isString(str)) {
			return TokenNames.String;
		}
		if(re.isSymbol(str)) {
			return TokenNames.Symbol;
		}
		
		if(re.isSpace(str)) {
			return TokenNames.Space;
		}
		
		return TokenNames.None;
	}

}
