package neuralNetwork.java.training;

import java.util.*;

import neuralNetwork.java.utils.*;
import neuralNetwork.java.functions.*;
import neuralNetwork.java.networks.FeedforwardNetwork;

public class BackPropagation extends Training {
	private FeedforwardNetwork network;
	private float eta;
	Cost cost = Cost.mst();
	GradientDescent gd = GradientDescent.SGD();
	
	public BackPropagation(
			FeedforwardNetwork n, 
			Dataset data,
			float eta,
			Cost cost)
	{
		super(data);
		this.network = n;
		this.eta = eta;
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
		int n = network.getWeights().length, size = training.size();
		float[][] nebla_b = new float[n][];
		for(int a = 0; a < nebla_b.length; a++)
		{
			nebla_b[a] = new float[network.get(a+1).size()];
		}
		training.getData().parallelStream().forEach(x -> {
			synchronized(errors)
			{
				float[] out = network.feedForward(x);
				float[] delta = cost.delta(out, 
						training.get(x), 
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
				errors.add(cost.f(out, training.get(x)));
			}
		});
		for(int a = 0; a < n; a++)
		{
			gd.update(network.getWeights()[a], eta, size);
			gd.update(network.get(a+1), nebla_b[a], eta, size);
			//network.getWeights()[a].adjustWeights(eta, size);
			//network.get(a+1).updateBiases(nebla_b[a], eta, size);
		}
		return Utils.sum(errors);
	}
}