#include "stdafx.h"
#include "functions.initialization.h"
#include "layer.h"
#include <Eigen/Dense>

using namespace Eigen;

using namespace neuralNetwork::functions::initialization;



void RandomUniform::initialize(layer::Connection * c)
{
	c->getWeights().setRandom();
}