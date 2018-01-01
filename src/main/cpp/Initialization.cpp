#include "stdafx.h"
#include "functions.initialization.h"
#include "layer.h"
#include <Eigen/Dense>

using namespace Eigen;

using namespace neuralNetwork::functions::initialization;

void Zeros::initialize(layer::Connection * c)
{
	c->getWeights().setZero();
}
void Ones::initialize(layer::Connection * c)
{
	c->getWeights().setOnes();
}
void Constant::initialize(layer::Connection * c)
{
	c->getWeights().setConstant(x);
}
void Random::initialize(layer::Connection * c)
{
	c->getWeights().setRandom();
}