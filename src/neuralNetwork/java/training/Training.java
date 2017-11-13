package neuralNetwork.java.training;

public class Training {
	private float error = 1;
	protected Dataset training, test;
	
	protected Training(Dataset data)
	{
		this.training = data;
	}
	public void train()
	{
		
	}
	public Dataset getData()
	{
		return training;
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
