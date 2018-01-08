package main.java.test;

import java.util.*;
import java.io.*;
import main.java.neuralNetwork.utils.*;

public class Test {

	public static void main(String[] args) 
	{
		Function sum = new Function() {
			@Override
			public float f(float... x) {
				float tmp = 0;
				for(int a = 0; a < x.length; a++)
					tmp += x[a];
				return tmp;
			}
			@Override
			public float der(float... x) {
				return 0;
			}
		},
		sigmoid = new Function() {
			@Override
			public float f(float... x) {
				return (float) (1/(1+Math.exp(-x[0])));
			}
			@Override
			public float der(float... x) {
				return x[0]*(1-x[0]);
			}
		};
		Neuron n1 = new Neuron(1, 2) {
			@Override
			public void calculate(float... inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				for(int a = 0; a < output.length; a++)
					this.output[a] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		},
		n2 = new Neuron() {
			@Override
			public void calculate(float... inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				for(int a = 0; a < output.length; a++)
					this.output[a] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		},
		n3 = new Neuron(3, 1) {
			@Override
			public void calculate(float... inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				for(int a = 0; a < output.length; a++)
					this.output[a] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		},
		n4 = new Neuron() {
			@Override
			public void calculate(float... inputs) 
			{
				this.value = sigmoid.f(new float[] {sum.f(inputs)});
				for(int a = 0; a < output.length; a++)
					this.output[a] = sigmoid.f(new float[] {sum.f(inputs)});
			}
		};
		n1.output[0] = 1;
		n1.output[1] = 0;
		n2.output[0] = -1;
		Neuron[] inputs = new Neuron[2], outputs = new Neuron[2];
		inputs[0] = n1;
		inputs[1] = n2;
		outputs[0] = n3;
		outputs[1] = n4;
		Connection c = new Connection(inputs, outputs);
		c.calculate();
		for(int a = 0; a < outputs.length; a++)
			for(int b = 0; b < outputs[a].outputs; b++)
			{
				System.out.println(outputs[a].value);
			}
	}
}
abstract class Function
{
	public abstract float f(float... x);
	public abstract float der(float... x);
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
		this.output = new float[outputs];
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
	public abstract void calculate(float... inputs);
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
		Matrix result = Matrix.multiply(new Matrix(inputs).transpose(), weights);
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
			i+= inputs[a].getOutputs();
		return i;
	}
	public int getOutputSize()
	{
		int i = 0;
		for(int a = 0; a < outputs.length; a++)
			i+= outputs[a].getInputs();
		return i;
	}
}