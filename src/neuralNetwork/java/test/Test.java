package neuralNetwork.java.test;

import java.util.*;

import neuralNetwork.java.*;
import neuralNetwork.java.functions.*;
import neuralNetwork.java.networks.FeedforwardNetwork;
import neuralNetwork.java.training.Dataset;

import java.io.*;

public class Test {
	public static void main (String[] args) throws IOException
	{
		float[][] in = new float[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
				target = new float[][] {{0}, {1}, {1}, {0}};
		FeedforwardNetwork network = new FeedforwardNetwork(
				new int[] {2, 5, 1}, 
				Activation.sigmoid(), 
				Initialization.randomUniform() 
				);
		long start = System.nanoTime(), end;
		network.train(10000, new Dataset(in, target), 0.15f, Cost.mst());
		end = System.nanoTime();
		System.out.println(end-start);
		for(int a = 0; a < in.length; a++)
		{
			System.out.println("Test data " + a + ":");
			float[] out = network.feedForward(in[a]);
			for(int b = 0; b < out.length; b++)
				System.out.print("Output: " + out[b] + " ");
			System.out.println();
			for(int b = 0; b < out.length; b++)
				System.out.print("Expected: " + target[a][b] + " ");
			for(int b = 0; b < out.length; b++)
				System.out.print("Error: " + Cost.crossEntropy().f(out, target[a]) + " ");
			System.out.println('\n');
		}
		network.print();
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