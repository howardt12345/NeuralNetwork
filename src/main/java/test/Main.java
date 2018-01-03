package main.java.test;

import java.util.*;
import java.io.*;
import main.java.neuralNetwork.utils.*;

public class Main {

	public static void main(String[] args) 
	{
		Function sum = new Function() {
			@Override
			public float f(float[] x) {
				float tmp = 0;
				for(int a = 0; a < x.length; a++)
					tmp += x[a];
				return tmp;
			}
			@Override
			public float der(float[] x) {
				return 0;
			}
		},
		sigmoid = new Function() {
			@Override
			public float f(float[] x) {
				return (float) (1/(1+Math.exp(-x[0])));
			}
			@Override
			public float der(float[] x) {
				return 0;
			}
		};
		Neuron n1 = new Neuron() {
			@Override
			public void calculate(float[] inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				this.output[0] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		}, 
		n2 = new Neuron() {
			@Override
			public void calculate(float[] inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				this.output[0] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		},
		n3 = new Neuron() {
			@Override
			public void calculate(float[] inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				this.output[0] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		},
		n4 = new Neuron() {
			@Override
			public void calculate(float[] inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				this.output[0] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		};
	}
}
abstract class Function
{
	public abstract float f(float[] x);
	public abstract float der(float[] x);
}
abstract class Neuron
{
	int inputs = 1, outputs = 1;
	float value = 0.0f;
	float[] output = new float[1];
	public Neuron() {}
	public Neuron(int inputs, int outputs)
	{
		this.inputs = inputs;
		this.outputs = outputs;
	}
	public float getValue()
	{
		return value;
	}
	public float[] getOutput()
	{
		return output;
	}
	public int getInputs()
	{
		return inputs;
	}
	public int getOutputs()
	{
		return outputs;
	}
	public abstract void calculate(float[] inputs);
}
class Connection
{
	Neuron[] inputs = new Neuron[1], outputs = new Neuron[1];
	Matrix weights;
	Connection(Neuron[] inputs, Neuron[] outputs)
	{
		this.inputs = inputs;
		this.outputs = outputs;
		initialize();
	}
	Connection(Neuron input, Neuron[] outputs)
	{
		inputs[0] = input;
		this.outputs = outputs;
		initialize();
	}
	Connection(Neuron[] inputs, Neuron output)
	{
		this.inputs = inputs;
		outputs[0] = output;
		initialize();
	}
	Connection(Neuron input, Neuron output)
	{
		inputs[0] = input;
		outputs[0] = output;
		initialize();
	}
	private void initialize()
	{
		weights = new Matrix(getInputSize(), getOutputSize());
		weights.randomize();
	}
	public void calculate()
	{
		float[] inputs = new float[getInputSize()];
		int i = 0;
		for(int a = 0; a < getInputCount(); a++)
			for(int b = 0; b < this.inputs[a].getOutputs(); b++)
			{
				inputs[i] = this.inputs[a].getOutput()[b];
				i++;
			}
		Matrix result = Matrix.multiply(new Matrix(inputs), weights);
		int k = 0;
		for(int a = 0; a < getOutputCount(); a++)
		{
			float[] tmp = new float[this.outputs[a].getInputs()];
			for(int b = 0; b < this.outputs[a].getInputs(); b++)
			{
				tmp[b] = result.get(k);
				k++;
			}
			this.outputs[a].calculate(tmp);
		}
	}
	public int getInputCount()
	{
		return inputs.length;
	}
	public int getOutputCount()
	{
		return outputs.length;
	}
	public int getInputSize()
	{
		int i = 0;
		for(int a = 0; a < inputs.length; a++)
			i+= inputs[a].getInputs();
		return i;
	}
	public int getOutputSize()
	{
		int i = 0;
		for(int a = 0; a < outputs.length; a++)
			i+= outputs[a].getOutputs();
		return i;
	}
}