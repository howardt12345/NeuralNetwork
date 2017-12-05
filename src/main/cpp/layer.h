#pragma once
#include "functions.activation.h"
#include "functions.initialization.h"
#include <Eigen/Dense>

using namespace Eigen;
using namespace neuralNetwork::functions;

namespace neuralNetwork
{
	namespace layer
	{
		class Layer
		{
		protected:
			MatrixXf layer, output, bias, nebla_b;
			functions::activation::Activation &a = functions::activation::Identity();
		public:
			Layer(MatrixXf layer,
				MatrixXf outputs,
				MatrixXf bias);
			Layer(MatrixXf layer,
				MatrixXf outputs,
				MatrixXf bias,
				functions::activation::Activation &a);
			virtual MatrixXf calculateOutputs() = 0;
			auto size()
			{
				return layer.size();
			}
			void setValues(MatrixXf values)
			{
				this->output = values;
			}
			MatrixXf values()
			{
				return layer;
			}
			MatrixXf outputs()
			{
				return output;
			}
		};
		class Connection
		{
		private:
			MatrixXf weights, nebla_w;
			Layer *input, *output;
			void initialize(functions::initialization::Initialization *i);
		public:
			Connection(Layer * input, Layer * output, functions::initialization::Initialization *i)
			{
				this->input = input;
				this->output = output;
			}
		};
	}
}