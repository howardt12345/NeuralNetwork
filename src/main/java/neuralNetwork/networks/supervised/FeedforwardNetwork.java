package main.java.neuralNetwork.networks.supervised;

import java.util.*;
import java.util.stream.*;

import main.java.neuralNetwork.functions.*;
import main.java.neuralNetwork.layer.*;
import main.java.neuralNetwork.training.*;
import main.java.neuralNetwork.utils.*;

public class FeedforwardNetwork extends NeuralNetwork {
	private Activation activation;
	
	public FeedforwardNetwork(
			int[] layers, 
			Activation a, 
			Initialization i) 
	{
		this.layers = new ConnectedLayer[layers.length];
		IntStream.range(0, layers()).forEach(n -> this.layers[n] = new ConnectedLayer(layers[n]));
		this.activation = a;
		initialize(i);
	}
	public FeedforwardNetwork(
			ConnectedLayer[] layers,
			Activation a,
			Initialization i)
	{
		this.layers = layers;
		this.activation = a;
		initialize(i);
	}
	private void initialize(Initialization i)
	{
		for(int a = 0; a < layers(); a++) {
			if(layers[a].getActivation() == null)
				layers[a].setActivation(activation);
		}
		setWeights(new Connection[layers()-1]);
		for(int a = 0; a < weights.length; a++) {
			weights[a] = new Connection(layers[a], layers[a+1], i);
		}
	}
	public Matrix feedForward(Matrix in)
	{
		layers[0].setValues(in);
		return calculateOutputs();
	}
	public float[] feedForward(float[] in)
	{
		layers[0].setValues(new Matrix(in));
		return calculateOutputs().matrix();
	}
	public Matrix calculateOutputs()
	{
		for(int a = 0; a < weights.length; a++)
		{
			weights[a].compute();
		}
		return layers[layers()-1].outputs();
	}
	public void train(int epochs, 
			Dataset trainingData, Dataset testData,
			float learnRate,
			Cost cost)
	{
		BackPropagation train = new BackPropagation(this, 
				trainingData, testData, learnRate, cost);
		this.cost = cost;
		train.train(epochs);
	}
	public void evaluate(Dataset testData)
	{
		List<Float> error = new ArrayList<Float>(), loss = new ArrayList<Float>();
		testData.getData().parallelStream().forEach(x -> {
			Matrix out = this.feedForward(new Matrix(x));
			Matrix delta = Matrix.subtract(out, new Matrix(testData.get(x)));
			error.add(delta.sum());
			loss.add(cost.f(out, new Matrix(testData.get(x))));
		});
		this.accuracy = 1-(Math.abs(Utils.sum(error)/error.size()));
		this.loss = Utils.sum(loss)/loss.size();
	}
	public Layer get(int index)
	{
		return layers[index];
	}
	public void set(ConnectedLayer l, int index)
	{
		layers[index] = l;
	}
	public Connection getWeight(int index)
	{
		return weights[index];
	}
	public Connection[] getWeights() 
	{
		return weights;
	}
	public void setWeights(Connection[] weights) 
	{
		this.weights = weights;
	}
	public Activation getActivation()
	{
		return activation;
	}
	public int layers()
	{
		return layers.length;
	}
	public void print()
	{
		System.out.println("Number of layers: " + layers() + '\n');
		IntStream.range(0, layers()).forEach(a -> 
		{
			System.out.println(get(a).getClass().getSimpleName() + " " + a + ":");
			System.out.println("Values:");
			get(a).values().print();
			System.out.println("Outputs:");
			get(a).outputs().print();
			System.out.println("Biases:");
			get(a).bias().print();
			System.out.println();
		});
		IntStream.range(0, layers()-1).forEach(a -> 
		{
			System.out.println("Weight " + a + ":");
			getWeights()[a].getWeights().print();
			System.out.println();
		});
	}
}