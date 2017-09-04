/**
 * This is a simple key value pair class 
 */

/**
 * @author Danny Reinheimer
 *
 */
public class Pair<T1,T2> {
	
	private T1 key;
	private T2 value;
	
	/**
	 * Constructor that takes the key and value
	 * @param key of type T1
	 * @param value of type T2
	 */
	public Pair(T1 key, T2 value) {
		this.key = key;
		this.value = value;
	}

	/**
	 * Retuns the key
	 * @return the key
	 */
	public T1 getKey() {
		return key;
	}

	/**
	 * Sets the key
	 * @param key the key to set
	 */
	public void setKey(T1 key) {
		this.key = key;
	}

	/**
	 * Retuns the value
	 * @return the value
	 */
	public T2 getValue() {
		return value;
	}

	/**
	 * Sets the value
	 * @param value the value to set
	 */
	public void setValue(T2 value) {
		this.value = value;
	}
	
	

}
