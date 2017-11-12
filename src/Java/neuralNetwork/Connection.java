package Java.neuralNetwork;

public class Connection {
	protected Matrix weights, deltaWeights;
	private Layer input, output;
	
	public Connection(Layer input, Layer output, Initialization i)
	{
		this.input = input;
		this.output = output;
		initialize(i);
	}
	private void initialize(Initialization i)
	{
		weights = new Matrix(input.size(), output.size());
		deltaWeights = Matrix.zero(input.size(), output.size());
		i.initialize(this);
	}
	public void compute()
	{
		output.layer = Matrix.multiply(new Matrix(input.outputs).transpose(),
				weights).matrix;
		output.updateOutputs();
	}
	public void adjustWeights(float eta, int length)
	{		
		weights = Matrix.subtract(weights, Matrix.scalarMultiply(deltaWeights, eta/length));
	}
	public int size()
	{
		return weights.matrix.length;
	}
	public Layer getInput()
	{
		return input;
	}
	public Layer getOutput()
	{
		return output;
	}
}