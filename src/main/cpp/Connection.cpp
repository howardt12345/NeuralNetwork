#include "stdafx.h"
#include "lib.h"

using namespace neuralNetwork::lib;
using namespace std;

Connection::Connection(Layer * input, Layer * output, initialization::Initialization *i)
{
	this->input = input;
	this->output = output;
	initialize(i);
}

void Connection::initialize(Initialization * i)
{
	weights = MatrixXf(input->size(), output->size());
	nebla_w = MatrixXf::Zero(input->size(), output->size());
	i->initialize(this);
}
void Connection::compute()
{
	output->setLayer(input->outputs().transpose() * weights.transpose());
	output->calculateOutputs();
}
void Connection::adjustWeights(float eta, int length)
{
	weights = weights.array() - (nebla_w.array() * eta / length);
}