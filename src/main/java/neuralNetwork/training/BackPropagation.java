package main.java.neuralNetwork.training;

import java.util.*;

import main.java.neuralNetwork.functions.*;
import main.java.neuralNetwork.networks.supervised.FeedforwardNetwork;
import main.java.neuralNetwork.utils.*;

public class BackPropagation extends Training {
	private FeedforwardNetwork network;
	private float eta;
	Cost cost = Cost.mst();
	GradientDescent gd = GradientDescent.SGD();
	
	public BackPropagation(
			FeedforwardNetwork n, 
			Dataset trainingData, Dataset testData,
			float eta,
			Cost cost)
	{
		super(trainingData, testData);
		this.network = n;
		this.eta = eta;
		this.cost = cost;
	}
	public void train(int epochs)
	{
		trainingLoss = new float[epochs];
		if (testData != null) testLoss = new float[epochs];
		int epoch = 1;
		do 
		{
			iterate(epoch);
			if(Float.compare(trainingLoss[epoch-1], 0.0f) <= 0) 
				break;
		} while(epoch++ < epochs);
		network.setTrainingLoss(trainingLoss);
		if (testData != null) network.setTestLoss(testLoss);
		network.setIterations(epoch);
	}
	public void iterate(int epoch)
	{
		List<Float> loss = new ArrayList<Float>();
		int n = network.getWeights().length, size = trainingData.size();
		trainingData.getData().parallelStream().forEach(x -> {
			synchronized(loss)
			{
				Matrix out = network.feedForward(new Matrix(x));
				Matrix delta = cost.delta(out, 
						new Matrix(trainingData.get(x)), 
							network.get(n).derivatives()
						);
				network.get(n).setDeltaBias(Matrix.add(network.get(n).nebla_b(), delta));;
				network.getWeights()[n-1].setDeltaWeights(Matrix.add(
						network.getWeights()[n-1].getDeltaWeights(), 
						Matrix.multiply(
								delta, 
								network.get(n-1).outputs().transpose()
								).transpose()
						));
				for(int l = 2; l <= n; l++)
				{
					delta = Matrix.haramardProduct(
							Matrix.multiply(
									network.getWeights()[n-l+1].getWeights(), 
									delta
									), 
							network.get(n-l+1).outputs());
					network.get(n-l+1).setDeltaBias(Matrix.add(network.get(n-l+1).nebla_b(), delta));;
					network.getWeights()[n-l].setDeltaWeights(Matrix.add(
							network.getWeights()[n-l].getDeltaWeights(), 
							Matrix.multiply(
									delta, 
									network.get(n-l).outputs().transpose()
									).transpose()
							));
				}
				loss.add(cost.f(out, new Matrix(trainingData.get(x))));
			}
		});
		for(int a = 0; a < n; a++)
		{
			gd.update(network.getWeights()[a], eta, size);
			gd.update(network.get(a+1), eta, size);
		}
		trainingLoss[epoch-1] = Utils.sum(loss)/loss.size();
		if(testData != null)
		{
			loss.clear();
			testData.getData().stream().forEach(x -> {
				synchronized(loss)
				{
					Matrix out = network.feedForward(new Matrix(x));
					loss.add(cost.f(out, new Matrix(testData.get(x))));
				}
			});
			testLoss[epoch-1] = Utils.sum(loss)/loss.size();
		}
	}
}