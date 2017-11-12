package Java.neuralNetwork.training;

import Java.neuralNetwork.Dataset;

public class Training {
	private float error = 1;
	protected Dataset data;
	
	protected Training(Dataset data)
	{
		this.data = data;
	}
	public void train()
	{
		
	}
	public Dataset getData()
	{
		return data;
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
