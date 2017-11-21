package main.java.neuralNetwork.training;

public abstract class Training {
	protected float[] trainingLoss, testLoss;
	protected Dataset trainingData = null, testData = null;
	
	protected Training(Dataset training, Dataset test)
	{
		this.trainingData = training;
		this.testData = test;
	}
	public abstract void train(int epochs);
	public Dataset getTrainingData()
	{
		return trainingData;
	}
	public float[] getTrainingLoss()
	{
		return trainingLoss;
	}
	public float[] getTestLoss()
	{
		return testLoss;
	}
}
