#include "stdafx.h"
#include "lib.h"

using namespace neuralNetwork::lib;
using namespace std;

void ConnectedLayer::updateBiases(float eta, int length)
{
	bias = bias.array() - (nebla_b.array() * eta / length);
}
MatrixXf ConnectedLayer::calculateOutputs()
{
	output = a.f(layer.array() + bias.array());
	return output;
}