package Java.neuralNetwork;

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
	public float getError()
	{
		return error;
	}
	protected void setError(float x)
	{
		error = x;
	}
}
