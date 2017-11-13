package neuralNetwork.java.networks;

import java.util.stream.*;

import neuralNetwork.java.utils.*;
import neuralNetwork.java.layer.*;
import neuralNetwork.java.functions.*;
import neuralNetwork.java.layer.Layer;
import neuralNetwork.java.training.*;

public class FeedforwardNetwork extends NeuralNetwork {
	protected Layer[] layers;
	private Connection[] weights;
	private Activation activation;
	
	public FeedforwardNetwork(
			int[] layers, 
			Activation a, 
			Initialization i) 
	{
		this.layers = new Layer[layers.length];
		IntStream.range(0, layers()).forEach(n -> this.layers[n] = new Layer(layers[n]));
		this.activation = a;
		initialize(i);
	}
	public FeedforwardNetwork(
			Layer[] layers,
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
	public float[] feedForward(float[] in)
	{
		layers[0].setValues(in);
		for(int a = 0; a < weights.length; a++)
		{
			weights[a].compute();
		}
		return layers[layers()-1].outputs();
	}
	public float[] feedForward(int[] in)
	{
		layers[0].setValues(Utils.toFloat(in));
		for(int a = 0; a < weights.length; a++)
		{
			weights[a].compute();
		}
		return layers[layers()-1].outputs();
	}
	public float[] feedForward(double[] in)
	{
		layers[0].setValues(Utils.toFloat(in));
		for(int a = 0; a < weights.length; a++)
		{
			weights[a].compute();
		}
		return layers[layers()-1].outputs();
	}
	public float train(int epochs, 
			Dataset data,
			float learnRate,
			Cost cost)
	{
		BackPropagation train = new BackPropagation(this, data, learnRate, cost);
		float tmp = train.train(epochs);
		return tmp;
	}
	public Layer get(int index)
	{
		return layers[index];
	}
	public void set(Layer l, int index)
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
		System.out.println("Number of layers: " + layers());
		System.out.println();
		IntStream.range(0, layers()).forEach(a -> 
		{
			System.out.println("Layer " + a + ":");
			System.out.println("Layer:");
			Utils.print(get(a).values());
			System.out.println("Outputs:");
			Utils.print(get(a).outputs());
			System.out.println("Biases:");
			Utils.print(get(a).getBias());
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