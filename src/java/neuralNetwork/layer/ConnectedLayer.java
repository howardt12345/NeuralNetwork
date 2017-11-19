package Java.neuralNetwork.layer;

import java.util.*;

import Java.neuralNetwork.functions.Activation;
import Java.neuralNetwork.utils.*;

public class ConnectedLayer extends Layer {
	private Activation activation;
	
	public ConnectedLayer(int neurons)
	{
		super(new Matrix(neurons, 1), new Matrix(neurons, 1), Matrix.ofValue(0.1f, neurons, 1));
	}
	public ConnectedLayer(int neurons, Activation a)
	{
		super(new Matrix(neurons, 1), new Matrix(neurons, 1), Matrix.ofValue(0.1f, neurons, 1));
		this.activation = a;
	}
	public ConnectedLayer(float[] layer)
	{
		super(new Matrix(layer), new Matrix(layer.length), Matrix.ofValue(0.1f, layer.length, 1));
	}
	public ConnectedLayer(float[] layer, Activation a)
	{
		super(new Matrix(layer), new Matrix(layer.length), Matrix.ofValue(0.1f, layer.length, 1));
		this.activation = a;
	}
	public void updateBiases(float eta, float length)
	{
		setBias(Matrix.subtract(bias, Matrix.multiply(nebla_b, eta/length)));
	}
	public void setActivation(Activation a)
	{
		this.activation = a;
	}
	public Activation getActivation()
	{
		return activation;
	}
	@Override
	public Matrix calculateOutputs() {
		outputs = getActivation().f(Matrix.add(layer, bias));
		return outputs;
	}
}