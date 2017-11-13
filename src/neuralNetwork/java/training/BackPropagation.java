package neuralNetwork.java.training;

import java.util.*;

import neuralNetwork.java.*;
import neuralNetwork.java.feedforward.FeedforwardNeuralNetwork;
import neuralNetwork.java.functions.Cost;

public class BackPropagation extends Training {
	private FeedforwardNeuralNetwork network;
	private float learnRate;
	Cost cost = Cost.mst();
	
	public BackPropagation(
			FeedforwardNeuralNetwork n, 
			Dataset data,
			float learnRate,
			Cost cost)
	{
		super(data);
		this.network = n;
		this.learnRate = learnRate;
		this.cost = cost;
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
		int n = network.getWeights().length, size = trainingSet.size();
		float[][] nebla_b = new float[n][];
		for(int a = 0; a < nebla_b.length; a++)
		{
			nebla_b[a] = new float[network.get(a+1).size()];
		}
		trainingSet.getData().parallelStream().forEach(x -> {
			synchronized(errors)
			{
				float[] out = network.feedForward(x);
				float[] delta = Utils.multiply(
						cost.delta(out, trainingSet.get(x)), 
						network.get(n).derivatives()
						);
				nebla_b[n-1] = Utils.add(nebla_b[n-1], delta);
				network.getWeights()[n-1].setDeltaWeights(Matrix.add(
						network.getWeights()[n-1].getDeltaWeights(), 
						Matrix.multiply(
								new Matrix(delta), 
								new Matrix(network.get(n-1).outputs()
										).transpose()
								).transpose()
						));
				for(int l = 2; l <= n; l++)
				{
					delta = Matrix.haramardProduct(
							Matrix.multiply(
									network.getWeights()[n-l+1].getWeights(), 
									new Matrix(delta)
									), 
							new Matrix(network.get(n-l+1).outputs())).matrix();
					nebla_b[n-l] = Utils.add(nebla_b[n-l], delta);
					network.getWeights()[n-l].setDeltaWeights(Matrix.add(
							network.getWeights()[n-l].getDeltaWeights(), 
							Matrix.multiply(
									new Matrix(delta), 
									new Matrix(network.get(n-l).outputs()
											).transpose()
									).transpose()
							));
				}
				errors.add(cost.f(out, trainingSet.get(x)));
			}
		});
		for(int a = 0; a < n; a++)
		{
			network.getWeights()[a].adjustWeights(learnRate, size);
			network.get(a+1).updateBiases(nebla_b[a], learnRate, size);
		}
		return Utils.sum(errors);
	}
}