package main.java.examples.addition;

import java.io.File;
import java.io.IOException;
import java.util.*;

import main.java.neuralNetwork.functions.*;
import main.java.neuralNetwork.networks.supervised.FeedforwardNetwork;
import main.java.neuralNetwork.training.Dataset;
import main.java.neuralNetwork.utils.Utils;

public class Addition {
	public static void main(String[] args) throws IOException
	{
		Dataset training = new Dataset(new File("data/addition/addTraining.txt")),
				test = new Dataset(new File("data/addition/addTest.txt"));
		FeedforwardNetwork network = new FeedforwardNetwork(
				new int[] {2, 100, 1}, 
				Activation.reLU(), 
				Initialization.randomUniform()
				);
		Utils.print(network.feedForward(new float[] {0, 1}));
		network.train(1, 
				training, 
				test, 
				0.01f, 
				Cost.mst());
		List<float[]> data = training.getData();
		for(int a = 0; a < data.size(); a++)
		{
			for(int b = 0; b < data.get(a).length; b++)
				System.out.print(data.get(a)[b] + " ");
			System.out.println();
		}
		for(int a = 0; a < data.size(); a++)
		{
			System.out.println("Test Data " + a + ":");
			float[] out = network.feedForward(data.get(a));
			for(int b = 0; b < out.length; b++)
				System.out.println("Output: " + out[b]);
			for(int b = 0; b < out.length; b++)
				System.out.println("Expected: " + training.get(data.get(a))[b] + '\n');
		}
		network.evaluate(test);
		System.out.println("Training iterations: " + network.getIterations());
		System.out.println("Accuracy: " + network.getAccuracy());
		System.out.println("Loss: " + network.getLoss());
	}
}