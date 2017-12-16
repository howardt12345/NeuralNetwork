#include "stdafx.h"
#include "networks.supervised.h"

using namespace neuralNetwork::networks::supervised;


FeedforwardNeuralNetwork::FeedforwardNeuralNetwork
	(int * layers, int size,
	activation::Activation & a, 
	initialization::Initialization * i)
{
	for (int i = 0; i < size; i++)
	{
		this->layers.push_back(layer::ConnectedLayer(layers[i], a));
	}
	initialize(i);
}
FeedforwardNeuralNetwork::FeedforwardNeuralNetwork
	(vector<layer::Layer> layers, 
	initialization::Initialization * i)
{
	this->layers = layers;
	initialize(i);
}

void FeedforwardNeuralNetwork::initialize(initialization::Initialization * i)
{
	for (int a = 0; a < layers.size()-2; a++)
	{
		weights.push_back(layer::Connection(&layers[a], &layers[a+1], i));
	}
}
MatrixXf FeedforwardNeuralNetwork::feedForward(MatrixXf in)
{
	layers[0].setValues(in);
	return calculateOutputs();
}
MatrixXf FeedforwardNeuralNetwork::calculateOutputs()
{
	for (int a = 0; a < weights.size(); a++)
	{
		weights[a].compute();
	}
	return layers.back().outputs();
}
void FeedforwardNeuralNetwork::train(
	data::Data trainingData, 
	data::Data testData, 
	float eta, 
	functions::cost::Cost cost)
{

}