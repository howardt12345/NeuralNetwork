package main.java.neuralNetwork.networks.supervised;

import java.util.*;

import main.java.neuralNetwork.functions.Cost;
import main.java.neuralNetwork.layer.*;
import main.java.neuralNetwork.training.Dataset;
import main.java.neuralNetwork.utils.Matrix;

import java.io.*;

public abstract class NeuralNetwork {
	protected Layer[] layers;
	protected Connection[] weights;
	protected Cost cost;
	private float[] trainingLoss = new float[1], testLoss = new float[1];
	protected float accuracy = 0, loss = 1;
	private int iterations = 0;
	public abstract void train(int epochs, 
			Dataset trainingData, Dataset testData,
			float learnRate,
			Cost cost);
	protected abstract void evaluate(Dataset testData);
	public float[] getTrainingLoss()
	{
		return trainingLoss;
	}
	public float[] getTestLoss()
	{
		return testLoss;
	}
	public void setTrainingLoss(float[] trainingLoss)
	{
		this.trainingLoss = trainingLoss;
	}
	public void setTestLoss(float[] testLoss)
	{
		this.testLoss = testLoss;
	}
	public int getIterations()
	{
		return iterations;
	}
	public void setIterations(int iterations)
	{
		this.iterations = iterations;
	}
	public float getAccuracy()
	{
		return accuracy;
	}
	public float getLoss()
	{
		return loss;
	}
}