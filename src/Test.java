
import java.util.*;
import java.io.*;

import NeuralNetwork.*;

public class Test {
	public static void main (String[] args) throws IOException
	{
		float[][] in = new float[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
				target = new float[][] {{0}, {1}, {1}, {0}};
		NeuralNetwork network = new NeuralNetwork(
				new int[] {2, 5, 1}, 
				Activation.sigmoid(), 
				Initialization.randomUniform() 
				);
		for(int a = 0; a < in.length; a++)
		{
			System.out.println("Input data " + a + ":");
			Utils.print(in[a]);
			System.out.println("Target output of input data " + a + ":");
			Utils.print(target[a]);
			System.out.println();
		}
		System.out.println();
		System.out.println("Training neural network 10000 times");
		network.train(10000, in, target, 3);
		System.out.println();
		for(int a = 0; a < in.length; a++)
		{
			System.out.println("Test data " + a + ":");
			float[] out = network.feedForward(in[a]);
			for(int b = 0; b < out.length; b++)
				System.out.print("Output: " + out[b] + " ");
			System.out.println();
			for(int b = 0; b < out.length; b++)
				System.out.print("Expected: " + target[a][b] + " ");
			//System.out.println();
			//network.print();
			System.out.println('\n');
		}
/*		float tmp = sc.nextFloat(), tmp1 = sc.nextFloat();
		float[] out = network.computeOutputs(new float[] {tmp, tmp1});
		for(int b = 0; b < out.length; b++)
			System.out.print("Output " + b + ": " + out[b] + " ");
		System.out.println();*/
	}
	@SuppressWarnings("unused")
	private static void setOutput(String filename)
	{
		try {
			PrintStream out = new PrintStream(new FileOutputStream(filename));
			System.setOut(out);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}