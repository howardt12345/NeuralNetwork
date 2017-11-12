package Java.neuralNetwork;

import java.util.*;

public class BackPropagation extends Training {
	private NeuralNetwork network;
	private float learnRate;
	Cost cost = Cost.quadratic();
	
	public BackPropagation(
			NeuralNetwork n, 
			Dataset data,
			float learnRate)
	{
		super(data);
		this.network = n;
		this.learnRate = learnRate;
	}
	public float train(int epochs)
	{
		int epoch = 1;
		do 
		{
			setError(iterate());
		} while(epoch++ < epochs);
		return getError();
	}
	public float iterate()
	{
		List<Float> errors = new ArrayList<Float>();
		int n = network.weights.length, size = data.size();
		float[][] nebla_b = new float[n][];
		for(int a = 0; a < nebla_b.length; a++)
		{
			nebla_b[a] = new float[network.get(a+1).size()];
		}
		data.getData().parallelStream().forEach(x -> {
			synchronized(errors)
			{
				float[] out = network.feedForward(x);
				float[] delta = Utils.multiply(
						cost.delta(out, data.get(x)), 
						network.layers[n].derivatives()
						);
				nebla_b[n-1] = Utils.add(nebla_b[n-1], delta);
				network.weights[n-1].deltaWeights = Matrix.add(
						network.weights[n-1].deltaWeights, 
						Matrix.multiply(
								new Matrix(delta), 
								new Matrix(network.get(n-1).outputs
										).transpose()
								).transpose()
						);
				for(int l = 2; l <= n; l++)
				{
					delta = Matrix.haramardProduct(
							Matrix.multiply(
									network.weights[n-l+1].weights, 
									new Matrix(delta)
									), 
							new Matrix(network.get(n-l+1).outputs)).matrix();
					nebla_b[n-l] = Utils.add(nebla_b[n-l], delta);
					network.weights[n-l].deltaWeights = Matrix.add(
							network.weights[n-l].deltaWeights, 
							Matrix.multiply(
									new Matrix(delta), 
									new Matrix(network.get(n-l).outputs
											).transpose()
									).transpose()
							);
				}
				errors.add(cost.f(out, data.get(x)));
			}
		});
		for(int a = 0; a < n; a++)
		{
			network.weights[a].adjustWeights(learnRate, size);
			network.get(a+1).updateBiases(nebla_b[a], learnRate, size);
		}
		return Utils.sum(errors);
	}
}