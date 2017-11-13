package neuralNetwork.java.examples.mnist;

import java.util.*;

import neuralNetwork.java.*;
import neuralNetwork.java.utils.*;
import neuralNetwork.java.functions.*;
import neuralNetwork.java.networks.FeedforwardNetwork;
import neuralNetwork.java.training.Dataset;

public class MNIST {
	public static void main(String[] args)
	{
		int[] labels = MnistReader.getLabels("data/mnist/t10k-labels.idx1-ubyte");
		List<int[][]> images = MnistReader.getImages("data/mnist/t10k-images.idx3-ubyte");
		
		assert(labels.length == images.size());
		assert(28 == images.get(0).length);
		assert(28 == images.get(0)[0].length);
		for (int i = 0; i < Math.min(10, labels.length); i++) 
		{
			printf("================= LABEL %d\n", labels[i]);
			printf("%s", MnistReader.renderImage(images.get(i)));
		}
		FeedforwardNetwork network = new FeedforwardNetwork(
				new int[] {784, 30, 10},
				Activation.sigmoid(),
				Initialization.randomUniform()
				);
		network.train(10, new Dataset(configureIn(images, 784), configureOut(labels, 10)), 1, Cost.mst());
		network.print();
	}
	public static float[][] configureIn(List<int[][]> in, int size)
	{
		float[][] tmp = new float[in.size()][size];
		for(int a = 0; a < tmp.length; a++)
			tmp[a] = Utils.divide(Utils.toArray(Utils.asString(in.get(a))), 255);
		return tmp;
	}
	public static void printf(String format, Object... args) {
		System.out.printf(format, args);
	}
	public static float[][] configureOut(int[] values, int size)
	{
		float[][] tmp = new float[values.length][size];
		for(int a = 0; a < tmp.length; a++)
			tmp[a][values[a]] = 1;
		return tmp;
	}
}
