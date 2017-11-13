package neuralNetwork.java.test;

import java.util.*;
import java.io.*;

public class Huffman {
	final static int R = 128;
	int[] ASCII;
	PriorityQueue<Node> nodes = new PriorityQueue<>((o1, o2) -> (o1.freq < o2.freq) ? -1 : 1);
	Map<Character, String> codes = new TreeMap<Character, String>();
	
	public static void main(String[] args) throws IOException
	{
		Scanner sc = new Scanner (System.in);
		Huffman h = new Huffman();
		String encoded = h.encode(sc.nextLine());
		System.out.println(encoded);
		System.out.println(encoded.length());
		System.out.println(encoded.length() > 8 ? encoded.length()%8 : 8 - encoded.length());
		System.out.println(h.decode(encoded));
		sc.close();
	}
	public String encode(String in)
	{
		ASCII = new int[R];
		nodes.clear();
		codes.clear();
		calculateCharIntervals(nodes, in);
		buildTree(nodes);
		generateCodes(nodes.peek(), "");
		
		codes = sortByValue(codes);
		System.out.println(nodes.peek().freq + " " + codes.size());
		codes.forEach((k, v) -> System.out.println("'" + k + "': " + v));
		
		String out = "";
		for (int a = 0; a < in.length(); a++)
			out += codes.get(in.charAt(a));
		return out;
	}
	public String decode(String in)
	{
		String out = "";
		Node node = nodes.peek();
		for (int a = 0; a < in.length(); )
		{
			Node tmp = node;
			while(tmp.left != null && tmp.right != null && a < in.length())
			{
				if (in.charAt(a) == '1')
					tmp = tmp.right;
				else tmp = tmp.left;
				a++;
			}
			if (tmp != null)
				if(tmp.data.length() == 1)
					out += tmp.data;
				else System.out.println("Invalid Input");
		}
		return out;
	}
	public static <K, V extends Comparable<? super V>> Map<K, V> sortByValue(Map<K, V> unsortMap) 
	{
	    List<Map.Entry<K, V>> list =
	            new LinkedList<Map.Entry<K, V>>(unsortMap.entrySet());
	    Collections.sort(list, new Comparator<Map.Entry<K, V>>() 
	    {
	        public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
	            return (o1.getValue()).compareTo(o2.getValue());
	        }
	    });
	    Map<K, V> result = new LinkedHashMap<K, V>();
	    for (Map.Entry<K, V> entry : list) {
	        result.put(entry.getKey(), entry.getValue());
	    }
	    return result;
	}
	private void buildTree(PriorityQueue<Node> nodes)
	{
		while(nodes.size() > 1)
		{
			nodes.add(new Node(nodes.poll(), nodes.poll()));
		}
	}
	private void generateCodes(Node node, String s)
	{
		if (node != null)
		{
			if (node.isLeaf())
				codes.put(node.data.charAt(0), s);
			else {
				generateCodes(node.right, s + "1");
				generateCodes(node.left, s + "0");
			}
		}
	}
	private void calculateCharIntervals(PriorityQueue<Node> nodes, String text)
	{
		for (int a = 0; a < text.length(); a++)
			ASCII[text.charAt(a)]++;
		for (int a = 0; a < ASCII.length; a++)
		{
			if (ASCII[a] > 0)
			{
				nodes.add(new Node(((char)a) + "", ASCII[a], null, null));
			}
		}
	}
}

class Node implements Comparable<Node>
{
	int freq;
	String data;
	Node left, right;
	
	Node (String data, int freq, Node left, Node right)
	{
		this.data = data;
		this.freq = freq;
		this.left = left;
		this.right = right;
	}
	Node(Node left, Node right)
	{
		this.freq = left.freq + right.freq;
		this.data = left.data + right.data;
		if (left.freq > right.freq)
		{
			this.right = left;
			this.left = right;
		}
		else 
		{
			this.right = right;
			this.left = left;
		}
	}
	boolean isLeaf()
	{
		assert (left == null && right == null) || (left != null) && (right != null);
		return left == null && right == null;
	}
	public int compareTo(Node that)
	{
		return this.freq - that.freq;
	}
}

