package Java.neuralNetwork.layer;

import Java.neuralNetwork.functions.Activation;
import Java.neuralNetwork.utils.*;

public abstract class Layer {
	protected Matrix layer, outputs, bias, nebla_b;
	protected Activation a;
	public Layer(Matrix layer, Matrix outputs, Matrix bias)
	{
		this.layer = layer;
		this.outputs = outputs;
		this.bias = bias;
		this.nebla_b = new Matrix(bias.rows(), bias.columns());
	}
	public abstract Matrix calculateOutputs();
	public int size()
	{
		assert((outputs.length() | bias.length()) == layer.length());
		return layer.length();
	}
	public void setValues(Matrix values)
	{
		assert(values.columns() == this.layer.columns()) : values.columns() + " != " + this.layer.columns();
		this.outputs = values;
	}
	public Matrix outputs()
	{
		return outputs;
	}
	public Matrix values()
	{
		return layer;
	}
	public Matrix bias() 
	{
		return bias;
	}
	public Matrix nebla_b()
	{
		return nebla_b;
	}
	public void setDeltaBias(Matrix nebla_b)
	{
		this.nebla_b = nebla_b;
	}
	public Matrix derivatives()
	{
		return getActivation().der(outputs);
	}
	public float get(int index)
	{
		return layer.get(index);
	}
	public void setBias(Matrix bias) 
	{
		this.bias = bias;
	}
	public void setLayer(Matrix layer) 
	{
		this.layer = layer;
	}
	public Activation getActivation()
	{
		return a;
	}
	public void setActivation(Activation a)
	{
		this.a = a;
	}
}
