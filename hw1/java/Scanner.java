import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;

/**
 * This is the main class for the Scanner
 */

/**
 * @author Danny Reinheimer
 *
 */
public class Scanner {

	/**
	 * @param args The file name 
	 */
	public static void main(String[] args) {
		// checks to see if we are given any arguments
		if(args.length < 1) {
			System.err.println("Please provide an input file to process");
			System.exit(1);
		}
		
		String fileName = args[0];
		Scan scan = new Scan(fileName);
		Pair<TokenNames,String> tokenPair;
		
		try {
			// get the name of the file minus the dot 
			int pos = fileName.lastIndexOf(".");
			String newFileName = fileName.substring(0, pos) + "_gen.c";
			PrintWriter writer = new PrintWriter(newFileName,"UTF-8");
			
			// keep getting the next token until we get a null
			while((tokenPair = scan.getNextToken()) != null) {
				// check to see if the token is an identifer but not main
				if(tokenPair.getKey() == TokenNames.Identifiers && !tokenPair.getValue().equals("main")) {
					String newName = "cs512" + tokenPair.getValue();
					writer.print(newName);
				}
				else {
					// just add it to the output with out modifying the values
					writer.print(tokenPair.getValue());
				}
			}
			writer.close();
		} catch (FileNotFoundException e) {
			System.err.println("Could not create output file");
			System.exit(1);
		} catch (UnsupportedEncodingException e) {
			System.err.println("Error encoding output file.  Not my fault though");
			System.exit(1);
		}
		
		
		

	}

}
