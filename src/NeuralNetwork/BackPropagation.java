package neuralNetwork;

import java.util.*;

public class BackPropagation extends Training {
	private NeuralNetwork network;
	private float[][] input, target;
	private float learnRate;
	Cost cost = Cost.quadratic();
	
	public BackPropagation(
			NeuralNetwork n, 
			float[][] in, 
			float[][] target,
			float learnRate)
	{
		this.network = n;
		this.input = in;
		this.target = target;
		this.learnRate = learnRate;
	}
	public void train(int epochs)
	{
		int epoch = 1;
		float error = 0;
		do 
		{
			error += this.iterate();
		} while(epoch++ < epochs);
		setError(error/epochs);
	}
	public float iterate()
	{
		float tmp = 0;
		int n = network.weights.length;
		float[][] nebla_b = new float[n][];
		for(int a = 0; a < nebla_b.length; a++)
		{
			nebla_b[a] = new float[network.get(a+1).size()];
		}
		for(int i = 0; i < input.length; i++)
		{
			float[] x = input[i], y = target[i];
			float[] out = network.feedForward(x);
			float[] delta = cost.delta(out, y, network.layers[n]);
			
			nebla_b[n-1] = Utils.add(nebla_b[n-1], delta);
			network.weights[n-1].deltaWeights = Matrix.add(
					network.weights[n-1].deltaWeights, 
					Matrix.multiply(
							new Matrix(delta), 
							new Matrix(network.get(n-1).outputs
									).transpose()
							).transpose());
			for(int l = 2; l < n; l++)
			{
				delta = Matrix.haramardProduct(
						Matrix.multiply(
								network.weights[n-l].weights.transpose(), 
								new Matrix(delta)
								), 
						new Matrix(network.get(n-l).outputs)).matrix();
				nebla_b[n-l] = Utils.add(nebla_b[n-l], delta);
				network.weights[n-l].deltaWeights = Matrix.add(
						network.weights[n-l].deltaWeights, 
						Matrix.multiply(
								new Matrix(delta), 
								new Matrix(network.get(n-l-1).outputs).transpose())
						);
			}
		}
		for(int a = 0; a < n; a++)
		{
			network.weights[a].adjustWeights(learnRate, input.length);
			network.get(a+1).updateBiases(nebla_b[a], learnRate, input.length);
		}
		return tmp;
	}
}