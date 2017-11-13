package neuralNetwork.java.training;

import neuralNetwork.java.*;

public class Training {
	private float error = 1;
	protected Dataset trainingSet;
	
	protected Training(Dataset data)
	{
		this.trainingSet = data;
	}
	public void train()
	{
		
	}
	public Dataset getData()
	{
		return trainingSet;
	}
	public float getError()
	{
		return error;
	}
	protected void setError(float x)
	{
		error = x;
	}
}
