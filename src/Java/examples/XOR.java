package Java.examples;

import Java.neuralNetwork.*;

public class XOR {
	public static void main(String[] args)
	{
		float[][] in = new float[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
				target = new float[][] {{0}, {1}, {1}, {0}};
		NeuralNetwork network = new NeuralNetwork(
				new int[] {2, 4, 1}, 
				Activation.sigmoid(), 
				Initialization.randomUniform() 
				);
		network.train(10000, new Dataset(in, target), 3);
		for(int a = 0; a < in.length; a++)
		{
			System.out.println("Test data " + a + ":");
			float[] out = network.feedForward(in[a]);
			for(int b = 0; b < out.length; b++)
				System.out.println("Output: " + out[b]);
			for(int b = 0; b < out.length; b++)
				System.out.println("Expected: " + target[a][b] + '\n');
		}
	}
}
