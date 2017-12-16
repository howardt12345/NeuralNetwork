#ifndef LIB_H
#define LIB_H
#pragma once
#include "stdafx.h"
#include "neuralNetwork.h"
#include "data.h"
#include "functions.activation.h"
#include "functions.cost.h"
#include "functions.gradientDescent.h"
#include "functions.initialization.h"
#include "training.training.h"
#include "layer.h"

namespace neuralNetwork
{
	namespace lib
	{
		using namespace data;
		using namespace functions::activation;
		using namespace functions::cost;
		using namespace functions::gradientDescent;
		using namespace functions::initialization;
		using namespace layer;
		using namespace networks;
		using namespace networks::supervised;
		using namespace training;
		using neuralNetwork::Utils;
	}
	namespace feedforward
	{
		using namespace data;
		using namespace functions::activation;
		using namespace functions::cost;
		using namespace functions::gradientDescent;
		using namespace functions::initialization;
		using namespace layer;
		using namespace networks::supervised;
		using namespace training;
		using neuralNetwork::Utils;
	}
}
#endif // !LIB_H
