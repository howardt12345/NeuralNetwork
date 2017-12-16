
#pragma once
#include "stdafx.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

namespace neuralNetwork
{
	namespace data
	{
	}
	namespace functions 
	{
		namespace activation
		{
			class Activation;
			class Identity;
			class Binary;
			class Sigmoid;
			class ReLu;
			class LeakyReLu;
			class PReLu;
			class TanH;
			class SoftMax;
			class SoftPlus;
			class Swish;
		}
		namespace cost
		{
			class Cost;
		}
		namespace initialization
		{
			class Initialization;
		}
		namespace gradientDescent
		{
			class GradientDescent;
		}
	}
	namespace layer 
	{  
		class Layer;
		class Connection;
		class ConnectedLayer;
	}
	namespace networks
	{
		namespace supervised 
		{
			class NeuralNetwork;
			class FeedforwardNeuralNetwork;
		}
	}
	namespace training 
	{
		class Training;
		class BackPropagation;
	}
	class Utils 
	{ 
	};
}
