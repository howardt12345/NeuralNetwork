package neuralNetwork;

import java.util.*;
import java.util.stream.*;
import java.util.function.*;
import java.io.*;

public class Layer {
	protected float[] layer, outputs, bias;
	private Activation activation;
	
	public Layer(int neurons)
	{
		this.layer = new float[neurons];
		this.outputs = new float[neurons];
		this.bias = new float[neurons];
		Arrays.fill(bias, 0.1f);
	}
	public Layer(int neurons, Activation a)
	{
		this.layer = new float[neurons];
		this.outputs = new float[neurons];
		this.bias = new float[neurons];
		Arrays.fill(bias, 0.1f);
		this.activation = a;
	}
	public Layer(float[] layer)
	{
		this.layer = layer;
		this.outputs = new float[layer.length];
		this.bias = new float[layer.length];
		Arrays.fill(bias, 0.1f);
	}
	public Layer(float[] layer, Activation a)
	{
		this.layer = layer;
		this.outputs = new float[layer.length];
		this.bias = new float[layer.length];
		Arrays.fill(bias, 0.1f);
		this.activation = a;
	}
	public void updateOutputs()
	{
		outputs = getActivation().f(Utils.add(layer, bias));
	}
	public void updateBiases(float[] nebla_b, float eta, float length)
	{
		bias = Utils.subtract(bias, Utils.multiply(nebla_b, eta/length));
	}
	public int size()
	{
		assert((outputs.length | bias.length) == layer.length);
		return layer.length;
	}
	public float[] outputs()
	{
		return outputs;
	}
	public float[] values()
	{
		return layer;
	}
	public float[] derivatives()
	{
		return getActivation().der(layer);
	}
	protected void setValues(float[] values)
	{
		assert(values.length == this.layer.length) : values.length + " != " + this.layer.length;
		this.outputs = values;
	}
	public void setActivation(Activation a)
	{
		this.activation = a;
	}
	public Activation getActivation()
	{
		return activation;
	}
	public float get(int index)
	{
		return layer[index];
	}
}