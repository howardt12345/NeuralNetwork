package main.java.test;

import java.util.*;

import main.java.neuralNetwork.*;
import main.java.neuralNetwork.functions.*;
import main.java.neuralNetwork.layer.ConnectedLayer;
import main.java.neuralNetwork.networks.supervised.FeedforwardNetwork;
import main.java.neuralNetwork.training.Dataset;
import main.java.neuralNetwork.utils.Utils;

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
		network.train(10000, new Dataset(in, target), null, 0.01f, Cost.crossEntropy());
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
			//for(int b = 0; b < out.length; b++)
				//System.out.print("Error: " + Cost.mst().f(out, target[a]) + " ");
			System.out.println('\n');
		}
		/*Utils.print(network.getTrainingLoss());
		System.out.println();
		Utils.print(network.getTestLoss());*/
		System.out.println(network.getIterations());
		network.evaluate(new Dataset(in, target));
		System.out.println(network.getAccuracy());
		System.out.println(network.getLoss());
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