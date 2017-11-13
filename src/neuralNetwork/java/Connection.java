package neuralNetwork.java;

import neuralNetwork.java.*;
import neuralNetwork.java.functions.Initialization;

public class Connection {
	private Matrix weights;
	private Matrix deltaWeights;
	private Layer input, output;
	
	public Connection(Layer input, Layer output, Initialization i)
	{
		this.input = input;
		this.output = output;
		initialize(i);
	}
	private void initialize(Initialization i)
	{
		setWeights(new Matrix(input.size(), output.size()));
		setDeltaWeights(Matrix.zero(input.size(), output.size()));
		i.initialize(this);
	}
	public void compute()
	{
		output.setLayer(Matrix.multiply(new Matrix(input.outputs).transpose(),
				getWeights()).matrix);
		output.updateOutputs();
	}
	public void adjustWeights(float eta, int length)
	{		
		setWeights(Matrix.subtract(getWeights(), Matrix.scalarMultiply(getDeltaWeights(), eta/length)));
	}
	public int size()
	{
		return getWeights().matrix.length;
	}
	public Layer getInput()
	{
		return input;
	}
	public Layer getOutput()
	{
		return output;
	}
	public Matrix getWeights() 
	{
		return weights;
	}
	public void setWeights(Matrix weights) 
	{
		this.weights = weights;
	}
	public Matrix getDeltaWeights()
	{
		return deltaWeights;
	}
	public void setDeltaWeights(Matrix deltaWeights) 
	{
		this.deltaWeights = deltaWeights;
	}
}