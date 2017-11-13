package neuralNetwork.java.layer;

import java.util.*;
import neuralNetwork.java.utils.*;
import neuralNetwork.java.functions.Activation;

public class Layer {
	private float[] layer;
	protected float[] outputs;
	private float[] bias;
	private Activation activation;
	
	public Layer(int neurons)
	{
		this.setLayer(new float[neurons]);
		this.outputs = new float[neurons];
		this.setBias(new float[neurons]);
		Arrays.fill(biases(), 0.1f);
	}
	public Layer(int neurons, Activation a)
	{
		this.setLayer(new float[neurons]);
		this.outputs = new float[neurons];
		this.setBias(new float[neurons]);
		Arrays.fill(biases(), 0.1f);
		this.activation = a;
	}
	public Layer(float[] layer)
	{
		this.setLayer(layer);
		this.outputs = new float[layer.length];
		this.setBias(new float[layer.length]);
		Arrays.fill(biases(), 0.1f);
	}
	public Layer(float[] layer, Activation a)
	{
		this.setLayer(layer);
		this.outputs = new float[layer.length];
		this.setBias(new float[layer.length]);
		Arrays.fill(biases(), 0.1f);
		this.activation = a;
	}
	public void updateOutputs()
	{
		outputs = getActivation().f(Utils.add(layer, biases()));
	}
	public void updateBiases(float[] nebla_b, float eta, float length)
	{
		setBias(Utils.subtract(biases(), Utils.multiply(nebla_b, eta/length)));
	}
	public int size()
	{
		assert((outputs.length | biases().length) == layer.length);
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
	public void setValues(float[] values)
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
	public float[] biases() {
		return bias;
	}
	public void setBias(float[] bias) {
		this.bias = bias;
	}
	public void setLayer(float[] layer) {
		this.layer = layer;
	}
}