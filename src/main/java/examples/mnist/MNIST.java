package main.java.examples.mnist;

import java.util.*;

import main.java.neuralNetwork.functions.*;
import main.java.neuralNetwork.networks.supervised.FeedforwardNetwork;
import main.java.neuralNetwork.training.Dataset;
import main.java.neuralNetwork.utils.*;

public class MNIST {
	public static void main(String[] args)
	{
		int[] trainLabels = MnistReader.getLabels("data/mnist/train-labels.idx1-ubyte"),
				testLabels = MnistReader.getLabels("data/mnist/t10k-labels.idx1-ubyte");
		List<int[][]> trainImages = MnistReader.getImages("data/mnist/train-images.idx3-ubyte"),
				testImages = MnistReader.getImages("data/mnist/t10k-images.idx3-ubyte");
				
		assert(trainLabels.length == trainImages.size());
		assert(28 == trainImages.get(0).length);
		assert(28 == trainImages.get(0)[0].length);
		System.out.println("Done reading data");
		FeedforwardNetwork network = new FeedforwardNetwork(
				new int[] {784, 30, 10},
				Activation.reLU(),
				Initialization.randomNormal()
				);
		network.train(30, 
				new Dataset(configureIn(trainImages, 784), configureOut(trainLabels, 10)), 
				new Dataset(configureIn(testImages, 784), configureOut(testLabels, 10)),
				1, 
				Cost.mst());
		network.evaluate(new Dataset(configureIn(testImages, 784), configureOut(testLabels, 10)));
		Utils.print(network.getTrainingLoss());
		System.out.println();
		Utils.print(network.getTestLoss());
		System.out.println("Training iterations: " + network.getIterations());
		System.out.println("Accuracy: " + network.getAccuracy());
		System.out.println("Loss: " + network.getLoss());
	}
	public static float[][] configureIn(List<int[][]> in, int size)
	{
		float[][] tmp = new float[in.size()][size];
		for(int a = 0; a < tmp.length; a++)
			tmp[a] = Utils.divide(Utils.toArray(Utils.asString(in.get(a))), 255);
		return tmp;
	}
	public static float[][] configureOut(int[] values, int size)
	{
		float[][] tmp = new float[values.length][size];
		for(int a = 0; a < tmp.length; a++)
			tmp[a][values[a]] = 1;
		return tmp;
	}
}
