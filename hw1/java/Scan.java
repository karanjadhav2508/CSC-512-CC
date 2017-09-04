import java.io.*;

/**
 * This is the scanner class used to scan input for tokens 
 */

/**
 * @author Danny Reinheimer
 *
 */
public class Scan {
	
	//private File file;
	private FileInputStream fileInput;
	private TokenNames lastFoundToken; // stores the last token found when search the string
	private Token token;  // the token object used to check for tokens
	private boolean foundToken;
	private char lastReadChar;  // if we had to back track we save the last character
	private boolean useLastReadChar;
	/**
	 * Constructor that constructs the scanner class 
	 * @param programName the name of the program to be scanned
	 */
	public Scan(String programName) {
		// Try opening the file
		try {
			//file = new File(programName);
			fileInput = new FileInputStream(programName);
		} catch(Exception e){
			System.err.println("Invallid file!");
			System.exit(1);
		}	
		lastFoundToken = TokenNames.None;
		token = new Token();
		foundToken = false;
		useLastReadChar = false;
		
	}
	
	/**
	 * Gets the next character from the file and return it.  Or 
	 * return the end of file character
	 * @return Char from the file
	 */
	private char getNextChar() {
		int chInt;
		try {
			if(useLastReadChar) {
				useLastReadChar = false;
				return lastReadChar;
			}
			// read one character at a time 
			if((chInt = fileInput.read()) != -1) {
				return (char) chInt;				
			}
		} catch (IOException e) {
			System.err.println("Error reading file in getNextChar.  Not my fault though");
			System.exit(1);
		}
		char ch = '\u001a';  //end of file character
		return ch;
	}
	
	/**
	 * Parses the input one character at a time and checks for tokens.
	 * We save the found token in the lastTokenFound variable and keep looking until we 
	 * stop finding tokens.  If we reach the end of the line and have found no tokens then throw an error
	 * @return Token for the given input
	 */
	public Pair<TokenNames,String> getNextToken() {
		lastFoundToken = TokenNames.None;  // reset the token names when we search for new tokens
		TokenNames currentToken = TokenNames.None;
		foundToken = false;
		String characters = "";  // A string containing all the found characters so fare
		char ch;
		// Loop until we reach the end of line or end of file
		while((ch = getNextChar()) != '\n' && ch != '\u001a') {
			characters += "" + ch;			
			// check for token
			currentToken = token.getToken(characters);
			if(currentToken != TokenNames.None) {
				foundToken = true;
				lastFoundToken = currentToken;
			}
			// if we have already found a token and the current token is none
			// then we no the token is the current string minus the last character
			// to avoid erros we ignore // since it will be picked up latter
			if(foundToken && currentToken == TokenNames.None && !characters.contains("//")) {
				String tokenValue = characters.substring(0, characters.length()-1);  // gets the string minus the last char
				lastReadChar = characters.charAt(characters.length() -1); // save the last character to use it to start from
				useLastReadChar = true;
				Pair<TokenNames,String> pair = new Pair<TokenNames, String>(lastFoundToken, tokenValue);
				return pair;
			}
		}
		// check to see if we are at the end of line and if we previously found a token
		// we do this because some tokens require the end of line character
		if(ch == '\n' && lastFoundToken == TokenNames.None) {
			characters += "" + ch;
			currentToken = token.getToken(characters);
			if(currentToken == TokenNames.None) {
				// we have reached the end of the line and found no tokens so throw an error
				System.err.println("Invalid input Not a valid token: " + characters);
				System.exit(1);
			}
			// if we found a token
			if(currentToken != TokenNames.None) {
				Pair<TokenNames,String> pair = new Pair<TokenNames, String>(currentToken, characters);
				useLastReadChar = false;
				return pair;
			}
		}
		// if we have the end of line character then check for tokens 
		if(ch == '\n' ) {
			
			//characters += "" + ch;
			currentToken = token.getToken(characters);
			if(currentToken != TokenNames.None) {
				Pair<TokenNames,String> pair = new Pair<TokenNames, String>(currentToken, characters);
				lastReadChar = ch;
				useLastReadChar = true;
				return pair;
			}
		}
		
		// if we have the end of file character see if there was a token before it
		if(ch == '\u001a' && lastFoundToken != TokenNames.None) {
			Pair<TokenNames,String> pair = new Pair<TokenNames, String>(lastFoundToken, characters);
			return pair;
		}
		
//		System.out.println("Error parsing input");
//		System.exit(0);
		return null;
		
		
		
	}
	
	

}
